#!/usr/bin/env python3
"""RTLLM OOD 五折排名对照(审计修正版)。

修正三点:
  1. tie-break: np.argsort(-s, kind='stable') — 分数并列时最早 gid 优先
  2. 五折聚合: CPP/LAT/RankEF 各折分别算池指标后按折平均(SLM 零样本折不变)
  3. task-cluster bootstrap: 整簇抽 50 任务 ID, 同任务 12 生成器同进同出,
     先任务内(跨生成器)聚合, 再宏平均
输出 results/rtllm_ranking_compare_5fold.json
"""
import json, os, sys, csv, glob, collections
import numpy as np
from sklearn.metrics import roc_auc_score

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']
BS = 4

def load_labels():
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(folds['ood_rtllm_tasks'])
    labels = {}
    for m in MODELS:
        gm = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm and int(r['generation_id']) < BS:
                gm[r['task']][int(r['generation_id'])] = int(bool(r.get('simulation_pass')))
        for t, gd in gm.items():
            if len(gd) == BS:
                labels[(m, t)] = [gd[g] for g in range(BS)]
    return labels

def load_scores():
    """{scorer: {(m, k, t): [4 scores]}} — 保留折维"""
    scores = collections.defaultdict(dict)
    # CPP: OOD|{k}|{t} P1
    for m in MODELS:
        pz = f'{OUT}/predictions/q2v2/{m}_pool.npz'
        if not os.path.exists(pz):
            continue
        z = np.load(pz)
        for key in z.files:
            base, arr = key.rsplit('|', 1)
            if not (base.startswith('OOD|') and arr == 'P1'):
                continue
            _, k, t = base.split('|', 2)
            scores['cpp'][(m, int(k), t)] = z[key][:BS].tolist()
    # LAT
    lp = f'{OUT}/results/rtllm_lat_scores.npz'
    if os.path.exists(lp):
        z = np.load(lp)
        for key in z.files:
            m, k, t = key.split('#', 2)
            scores['lat_l27'][(m, int(k), t)] = z[key].tolist()
    # SLM(零样本, 折不变 — 复制到每折以走同一聚合路径)
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(folds['ood_rtllm_tasks'])
    sm = collections.defaultdict(dict)
    for fp in glob.glob(f'{OUT}/predictions/slm_judge_scores_*.csv'):
        for r in csv.DictReader(open(fp)):
            if r['task'] in rtllm:
                sm[(r['model'], r['task'])][int(r['gid'])] = float(r['score'])
    for (m, t), gd in sm.items():
        if len(gd) == BS:
            for k in range(5):
                scores['slm_judge'][(m, k, t)] = [gd[g] for g in range(BS)]
    # RankEF 5fold(修正版)
    rp = f'{OUT}/results/rtllm_rankek_scores_5fold.npz'
    if os.path.exists(rp):
        z = np.load(rp)
        for key in z.files:
            m, k, t = key.split('#', 2)
            scores['rankef'][(m, int(k), t)] = z[key].tolist()
    return scores

def pool_pass(scores4, labels4):
    """稳定排序 Pass@rank-1/2; 并列 → 最早 gid 优先(kind='stable')"""
    s = np.array(scores4[:BS]); y = np.array(labels4[:BS])
    order = np.argsort(-s, kind='stable')
    return int(y[order[0]]), int(y[order[:2]].max())

def fold_metrics(scores, labels, scorer):
    """每折池级指标 → {k: (pr1_arr, pr2_arr, auroc_arr, keys)}"""
    out = {}
    for k in range(5):
        pr1, pr2, au, keys = [], [], [], []
        for (m, kk, t), s4 in scores[scorer].items():
            if kk != k or (m, t) not in labels:
                continue
            y = labels[(m, t)]
            p1, p2 = pool_pass(s4, y)
            pr1.append(p1); pr2.append(p2); keys.append((m, t))
            yarr = np.array(y)
            if len(np.unique(yarr)) == 2:
                try:
                    au.append(roc_auc_score(yarr, np.array(s4)))
                except Exception:
                    pass
        if pr1:
            out[k] = (np.array(pr1), np.array(pr2),
                      np.array(au) if au else None, keys)
    return out

def task_cluster_bootstrap(per_task_A, per_task_B, n_boot=10000, seed=42):
    """per_task_X: {(m, t): pass}; 整簇抽任务, 先任务内跨生成器聚合, 再宏平均"""
    tasks = sorted(set(t for (_, t) in per_task_A) & set(t for (_, t) in per_task_B))
    by_task = collections.defaultdict(list)
    for t in tasks:
        by_task[t].append(t)
    task_ids = sorted(by_task)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(task_ids), len(task_ids))
        # 任务簇 → 任务内生成器均值 → 宏平均(等价于先按簇聚合所有生成器记录)
        a_vals, b_vals = [], []
        for i in idx:
            t = task_ids[i]
            a_vals.extend(v for (m, tt), v in per_task_A.items() if tt == t)
            b_vals.extend(v for (m, tt), v in per_task_B.items() if tt == t)
        diffs.append(np.mean(a_vals) - np.mean(b_vals))
    return np.percentile(diffs, [2.5, 97.5])

def main():
    labels = load_labels()
    scores = load_scores()
    print(f'labels: {len(labels)} pools', flush=True)
    for s, d in scores.items():
        ks = sorted(set(k[1] for k in d))
        print(f'  {s}: {len(d)} pools, folds={ks}', flush=True)

    report, per_task_pass = {}, {}
    for sname in ['cpp', 'lat_l27', 'slm_judge', 'rankef']:
        if sname not in scores:
            continue
        fm = fold_metrics(scores, labels, sname)
        if not fm:
            continue
        pr1_all = np.concatenate([fm[k][0] for k in fm])
        pr2_all = np.concatenate([fm[k][1] for k in fm])
        au_all = [fm[k][2] for k in fm if fm[k][2] is not None and len(fm[k][2])]
        au_mean = float(np.concatenate(au_all).mean()) if au_all else None
        # 池级记录 → 任务簇(跨折平均同一 (m,t))
        tp1 = collections.defaultdict(list)
        keys = []
        for k in fm:
            for i, key in enumerate(fm[k][3]):
                tp1[key].append(int(fm[k][0][i])); keys.append(key)
        per_task_pass[sname] = {kk: float(np.mean(v)) for kk, v in tp1.items()}
        entry = {
            'pass@rank1': round(float(pr1_all.mean())*100, 2),
            'pass@rank2': round(float(pr2_all.mean())*100, 2),
            'candidate_auroc': round(au_mean, 4) if au_mean is not None else None,
            'n_pool_fold_obs': int(len(pr1_all)),
            'n_folds': len(fm),
            'per_fold_pass1': {k: round(float(fm[k][0].mean())*100, 2) for k in sorted(fm)}}
        report[sname] = entry

    # task-cluster bootstrap vs cpp
    if 'cpp' in per_task_pass:
        base = per_task_pass['cpp']
        for sname, tp in per_task_pass.items():
            if sname == 'cpp':
                continue
            lo, hi = task_cluster_bootstrap(tp, base)
            ma = np.mean([np.mean([v for (m, t), v in tp.items() if True])])
            mb = np.mean([np.mean([v for (m, t), v in base.items() if True])])
            # 宏平均 = 先任务簇均值? 口径: 先任务(跨折)均值, 再模型均值
            def macro(d):
                bym = collections.defaultdict(list)
                for (m, t), v in d.items():
                    bym[m].append(v)
                return float(np.mean([np.mean(v) for v in bym.values()]))
            report[sname]['delta_vs_cpp_pp'] = {
                'mean': round(100*(macro(tp) - macro(base)), 2),
                'lo': round(100*float(lo), 2), 'hi': round(100*float(hi), 2),
                'method': 'task-cluster bootstrap, 10000 draws, cluster=task(12 generators)'}

    # tie-break 并列率(审计 P1 证据)
    tie = {}
    for sname in scores:
        n_tie = n_all = 0
        for key, s4 in scores[sname].items():
            if key[1] != 0:
                continue
            s = np.array(s4[:BS]); n_all += 1
            if int((s == s.max()).sum()) > 1:
                n_tie += 1
        if n_all:
            tie[sname] = round(n_tie/n_all, 4)
    report['_tie_rate_at_max_fold0'] = tie

    json.dump(report, open(f'{OUT}/results/rtllm_ranking_compare_5fold.json', 'w'), indent=1)
    print(json.dumps(report, indent=1), flush=True)

if __name__ == '__main__':
    main()
