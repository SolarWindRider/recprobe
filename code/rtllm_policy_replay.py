#!/usr/bin/env python3
"""RTLLM OOD 策略回放聚合 + 配对 bootstrap。

臂: alone / retry@4 / sc@4 / q1only@4 / recprobe@4 / rankef_tok@4 / oracle@4
  前6臂来自 decisions_trackA.csv 的 OOD 行(离线回放, fold 特异 P1/P2);
  rankef_tok@4 用 RTLLM RankEF 分数 + τ=0.5 门控(零样本迁移, IID 校准规则)。
输出 results/rtllm_policy_replay.json
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

def main():
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(folds['ood_rtllm_tasks'])

    # ===== 1. 从 decisions_trackA.csv 取 OOD 行(fold 特异, 排除 ens) =====
    per_task = collections.defaultdict(dict)  # (m, fold, task) -> {arm: (pass, tokens, used)}
    for r in csv.DictReader(open(f'{OUT}/predictions/decisions_trackA.csv')):
        if r['domain'] != 'OOD':
            continue
        key = (r['model'], r['fold'], r['task'])
        arm = f"{r['method']}@{r['B']}" if r['B'] != '-' else f"{r['method']}@1"
        per_task[key][arm] = (int(r['submitted_pass']), int(r['gen_tokens']), int(r['used']))

    # ===== 2. rankef_tok@4: RankEF 分数 τ=0.5 门控(零样本) =====
    rk = np.load(f'{OUT}/results/rtllm_rankek_scores.npz')
    rankef = {}
    for key in rk.files:
        m, t = key.split('#', 1)
        rankef[(m, t)] = rk[key].tolist()
    glen = {}
    for m in MODELS:
        for line in open(f'{STEP2}/{m}/labels.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm:
                glen[(m, r['task'], int(r['generation_id']))] = int(r['gen_len'])
    ymap = {}
    for m in MODELS:
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm and int(r['generation_id']) < BS:
                ymap[(m, r['task'], int(r['generation_id']))] = int(bool(r['simulation_pass']))

    for m in MODELS:
        tasks_m = sorted(set(t for (mm, t) in rankef if mm == m))
        for t in tasks_m:
            scores = rankef[(m, t)]
            y = [ymap.get((m, t, g), 0) for g in range(BS)]
            toks = [glen.get((m, t, g), 0) for g in range(BS)]
            # τ=0.5 门控级联: 沿生成序采样, 当前样本分数≥τ 则停
            kept = [0]
            for i in range(1, BS):
                if scores[kept[-1]] >= 0.5:
                    break
                kept.append(i)
            sub = int(np.argmax([scores[i] for i in kept]))
            key = (m, 'fold0', t)
            per_task.setdefault(key, {})['rankef_tok@4'] = (
                int(y[sub]), int(sum(toks[i] for i in kept)), len(kept))

    # ===== 3. 聚合: macro(模型均值) =====
    ARMS = ['alone@1', 'retry@4', 'sc@4', 'q1only@4', 'recprobe@4',
            'rankef_tok@4', 'oracle@4']
    summary = {}
    task_pass = {}  # arm -> {(m, task): pass}
    for arm in ARMS:
        per_model = {}
        for m in MODELS:
            ps, tks = [], []
            for (mm, fd, t), arms in per_task.items():
                if mm == m and arm in arms:
                    ps.append(arms[arm][0]); tks.append(arms[arm][1])
            if ps:
                per_model[m] = (float(np.mean(ps)), float(np.mean(tks)), len(ps))
        if per_model:
            macro_pass = float(np.mean([v[0] for v in per_model.values()]))
            mean_tok = float(np.mean([v[1] for v in per_model.values()]))
            summary[arm] = {'pass': round(macro_pass, 4),
                            'mean_tokens': round(mean_tok, 0),
                            'n_task_model': sum(v[2] for v in per_model.values()),
                            'per_model': {m: {'pass': round(v[0], 4),
                                              'tokens': round(v[1], 0)}
                                          for m, v in per_model.items()}}
        # 供 bootstrap
        tp = {}
        for (mm, fd, t), arms in per_task.items():
            if arm in arms:
                # fold 平均(同任务多折取均值, 保持任务粒度)
                tp.setdefault((mm, t), []).append(arms[arm][0])
        task_pass[arm] = {k: float(np.mean(v)) for k, v in tp.items()}

    # ===== 4. 配对 bootstrap(任务级重采样, 模型内) vs alone / vs recprobe@4 =====
    def paired_boot(A, B, n_boot=10000, seed=42):
        keys = sorted(set(A) & set(B))
        by_model = collections.defaultdict(list)
        for k in keys:
            by_model[k[0]].append(k)
        rng = np.random.RandomState(seed)
        diffs = []
        for _ in range(n_boot):
            ds = []
            for m, ks in by_model.items():
                idx = rng.choice(len(ks), len(ks), replace=True)
                a = np.mean([A[ks[i]] for i in idx])
                b = np.mean([B[ks[i]] for i in idx])
                ds.append(a - b)
            diffs.append(np.mean(ds))
        return np.percentile(diffs, [2.5, 97.5])

    deltas = {}
    base = task_pass.get('alone@1', {})
    rec = task_pass.get('recprobe@4', {})
    for arm in ARMS:
        if arm in ('alone@1',):
            continue
        tp = task_pass.get(arm, {})
        if not tp:
            continue
        d = {}
        if base:
            lo, hi = paired_boot(tp, base)
            d['vs_alone_pp'] = {'mean': round(100*float(np.mean(list(tp.values()))
                                                     - np.mean(list(base.values()))), 2),
                                'lo': round(100*float(lo), 2), 'hi': round(100*float(hi), 2)}
        if arm != 'recprobe@4' and rec:
            lo, hi = paired_boot(tp, rec)
            d['vs_recprobe_pp'] = {'mean': round(100*float(np.mean(list(tp.values()))
                                                         - np.mean(list(rec.values()))), 2),
                                   'lo': round(100*float(lo), 2), 'hi': round(100*float(hi), 2)}
        deltas[arm] = d

    out = {'mode': 'rtllm_ood_replay', 'n_tasks': len(rtllm), 'B': BS,
           'rankef_tok_rule': 'tau=0.5 gate, zero-shot from IID rule',
           'arms': summary, 'deltas': deltas}
    json.dump(out, open(f'{OUT}/results/rtllm_policy_replay.json', 'w'), indent=1)
    for arm in ARMS:
        if arm in summary:
            s = summary[arm]
            dl = deltas.get(arm, {})
            va = dl.get('vs_alone_pp', {})
            vr = dl.get('vs_recprobe_pp', {})
            print(f"{arm}: pass={s['pass']:.4f} tok={s['mean_tokens']:.0f}"
                  f" | vs alone {va.get('mean','--'):+.2f} [{va.get('lo','')},{va.get('hi','')}]"
                  f" | vs recprobe {vr.get('mean','--'):+.2f} [{vr.get('lo','')},{vr.get('hi','')}]",
                  flush=True)
    print('saved results/rtllm_policy_replay.json')

if __name__ == '__main__':
    main()
