#!/usr/bin/env python3
"""RankEF Adaptive / Hybrid OOD 回放(审计 P1)。

规则(全部 IID 冻结, 零样本迁移 RTLLM):
  rankef_adaptive@4: thr_cascade(rk, B=4, tau_R[model#fold]) — tau_R 来自
    IID validation 冻结的 results/rankef_adaptive_thresholds.json;
  hybrid@4: RecProbe 门控停止(P1<0.5 and isnan(P2) or P2>0.5), 候选选择
    改用 RankEF argmax(与 IID hybrid 同定义)。
既有臂(alone/sc/q1only_tok/prompt_only_tok/q2lite_tok/recprobe/oracle_stop/
oracle/q2lite)直接聚合 decisions_q2v2.csv 的 OOD 行。
RankEF 分数必须来自修正版 5fold 重推(rtllm_rankek_scores_5fold.npz)。
输出 results/rtllm_policy_replay_v2.json
"""
import csv, json, os, sys, collections
import numpy as np

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']
BS = 4

def thr_cascade(score, B, tau):
    kept = [0]
    for i in range(1, B):
        last = kept[-1]
        if score[last] < tau and len(kept) < B:
            kept.append(i)
        else:
            break
    return kept

def rp_cascade(p1, p2, B):
    kept = [0]
    for i in range(1, B):
        last = kept[-1]
        g = p2[last]
        cont = p1[last] < 0.5 and (np.isnan(g) or g > 0.5)
        if not cont or len(kept) >= B:
            break
        kept.append(i)
    return kept

def main():
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(folds['ood_rtllm_tasks'])
    thr = json.load(open(f'{OUT}/results/rankef_adaptive_thresholds.json'))
    rkz = np.load(f'{OUT}/results/rtllm_rankek_scores_5fold.npz')
    rankef = {}
    for key in rkz.files:
        m, k, t = key.split('#', 2)
        rankef[(m, int(k), t)] = rkz[key].tolist()

    # 标签 + token
    ymap, glen = {}, {}
    for m in MODELS:
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm:
                ymap[(m, r['task'], int(r['generation_id']))] = int(bool(r['simulation_pass']))
        for line in open(f'{STEP2}/{m}/labels.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm:
                glen[(m, r['task'], int(r['generation_id']))] = int(r['gen_len'])
    # OOD P1/P2(池)
    pools = {}
    for m in MODELS:
        z = np.load(f'{OUT}/predictions/q2v2/{m}_pool.npz')
        for key in z.files:
            base, arr = key.rsplit('|', 1)
            if not base.startswith('OOD|'):
                continue
            _, k, t = base.split('|', 2)
            pools.setdefault((m, int(k), t), {})[arr] = z[key][:BS]

    per_task = collections.defaultdict(dict)  # (m, fold, task) -> arm -> (pass, tokens, used)
    # 既有臂: decisions_q2v2.csv OOD
    for r in csv.DictReader(open(f'{OUT}/predictions/decisions_q2v2.csv')):
        if r['domain'] != 'OOD':
            continue
        key = (r['model'], r['fold'], r['task'])
        arm = f"{r['method']}@{r['B']}"
        per_task[key][arm] = (int(r['submitted_pass']), int(r['gen_tokens']), int(r['used']))

    # 新臂: rankef_adaptive@4 / hybrid@4(逐折)
    n_missing_tau = 0
    for m in MODELS:
        for k in range(5):
            tau_R = thr.get(f'{m}#fold{k}', {}).get('tau_R')
            tasks_m = sorted(set(t for (mm, kk, t) in rankef if mm == m and kk == k))
            for t in tasks_m:
                key = (m, f'fold{k}', t)
                rk = rankef.get((m, k, t))
                if rk is None:
                    continue
                y = [ymap.get((m, t, g), 0) for g in range(BS)]
                toks = [glen.get((m, t, g), 0) for g in range(BS)]
                if tau_R is not None:
                    kept = thr_cascade(rk, BS, tau_R)
                else:
                    n_missing_tau += 1
                    kept = thr_cascade(rk, BS, 0.5)
                sub = int(np.argmax([rk[i] for i in kept]))
                per_task[key]['rankef_adaptive@4'] = (
                    int(y[sub]), int(sum(toks[i] for i in kept)), len(kept))
                pool = pools.get((m, k, t), {})
                if 'P1' in pool and 'P2' in pool:
                    p1 = pool['P1'].astype(np.float64)
                    p2 = pool['P2'].astype(np.float64)
                    kept_h = rp_cascade(p1, p2, BS)
                    sub_h = int(np.argmax([rk[i] for i in kept_h]))
                    per_task[key]['hybrid@4'] = (
                        int(y[sub_h]), int(sum(toks[i] for i in kept_h)), len(kept_h))

    # ===== 聚合(macro=模型均值) + task-cluster bootstrap =====
    ARMS = ['alone@4', 'sc@4', 'q1only@4', 'q1only_tok@4', 'prompt_only_tok@4',
            'q2lite@4', 'q2lite_tok@4', 'recprobe@4', 'hybrid@4',
            'rankef_adaptive@4', 'oracle_stop@4', 'oracle@4']
    summary, task_pass = {}, {}
    for arm in ARMS:
        per_model = collections.defaultdict(lambda: ([], []))
        tp = collections.defaultdict(list)
        for (mm, fd, t), arms in per_task.items():
            if arm not in arms:
                continue
            p, tk, _ = arms[arm]
            per_model[mm][0].append(p); per_model[mm][1].append(tk)
            tp[(mm, t)].append(p)
        if not per_model:
            continue
        macro_pass = float(np.mean([np.mean(v[0]) for v in per_model.values()]))
        mean_tok = float(np.mean([np.mean(v[1]) for v in per_model.values()]))
        summary[arm] = {'pass': round(macro_pass, 4), 'mean_tokens': round(mean_tok, 0),
                        'n_task_model': sum(len(v[0]) for v in per_model.values()),
                        'per_model': {m: {'pass': round(float(np.mean(v[0])), 4),
                                          'tokens': round(float(np.mean(v[1])), 0)}
                                      for m, v in per_model.items()}}
        task_pass[arm] = {kk: float(np.mean(v)) for kk, v in tp.items()}

    def cluster_boot(A, B, n_boot=10000, seed=42):
        tasks = sorted(set(t for (_, t) in A) & set(t for (_, t) in B))
        rng = np.random.RandomState(seed)
        av, bv = collections.defaultdict(list), collections.defaultdict(list)
        for (m, t), v in A.items():
            av[t].append(v)
        for (m, t), v in B.items():
            bv[t].append(v)
        diffs = []
        for _ in range(n_boot):
            idx = rng.randint(0, len(tasks), len(tasks))
            a = [x for i in idx for x in av[tasks[i]]]
            b = [x for i in idx for x in bv[tasks[i]]]
            diffs.append(np.mean(a) - np.mean(b))
        return np.percentile(diffs, [2.5, 97.5])

    def macro(d):
        bym = collections.defaultdict(list)
        for (m, t), v in d.items():
            bym[m].append(v)
        return float(np.mean([np.mean(v) for v in bym.values()]))

    deltas = {}
    base = task_pass.get('alone@4', {})
    rec = task_pass.get('recprobe@4', {})
    for arm, tp in task_pass.items():
        if arm == 'alone@4':
            continue
        d = {}
        if base:
            lo, hi = cluster_boot(tp, base)
            d['vs_alone_pp'] = {'mean': round(100*(macro(tp)-macro(base)), 2),
                                'lo': round(100*float(lo), 2), 'hi': round(100*float(hi), 2)}
        if arm != 'recprobe@4' and rec:
            lo, hi = cluster_boot(tp, rec)
            d['vs_recprobe_pp'] = {'mean': round(100*(macro(tp)-macro(rec)), 2),
                                   'lo': round(100*float(lo), 2), 'hi': round(100*float(hi), 2)}
        deltas[arm] = d

    out = {'mode': 'rtllm_ood_replay_v2',
           'ranker_source': 'rtllm_rankek_scores_5fold.npz (corrected fine-tuned encoder)',
           'thresholds': 'IID-val frozen (rankef_adaptive_thresholds.json)',
           'missing_tau_fallback_05': n_missing_tau,
           'arms': summary, 'deltas': deltas}
    json.dump(out, open(f'{OUT}/results/rtllm_policy_replay_v2.json', 'w'), indent=1)
    for arm in ARMS:
        if arm in summary:
            s = summary[arm]; dl = deltas.get(arm, {})
            va = dl.get('vs_alone_pp', {}); vr = dl.get('vs_recprobe_pp', {})
            print(f"{arm}: pass={s['pass']:.4f} tok={s['mean_tokens']:.0f}"
                  f" | vsAlone {va.get('mean','-')} [{va.get('lo','-')},{va.get('hi','-')}]"
                  f" | vsRec {vr.get('mean','-')} [{vr.get('lo','-')},{vr.get('hi','-')}]", flush=True)
    print('saved results/rtllm_policy_replay_v2.json')

if __name__ == '__main__':
    main()
