#!/usr/bin/env python3
"""标量特征档 Q1/Q2(快速): [T_out, mean_entropy, mean_chosen_logprob]。

只读 samples/*.lp.npz(无残差流),秒级 CV。目的:
  1) 立即给出 Q1/Q2 的廉价基线
  2) 检验 Step1.5 预言: 标量档 vs 残差流档谁赢
协议与 step2_analysis.probe 一致: 设计分组 KFold(10, seed=42)。
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = 'experiments/step2'
MODELS = ['CodeV-QW-7B', 'CodeV-DS-6.7B', 'VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B', 'CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1', 'DeepSeek-Coder-6.7B']


def load_scalar_rows(m):
    tasks = {t['task_id']: t for t in json.load(open(f'{ROOT}/tasks_step2.json'))}
    ev = {}
    for r in map(json.loads, open(f'{ROOT}/{m}/eval.jsonl')):
        ev[(r['task'], r['generation_id'])] = bool(r['simulation_pass'])
    task_pass = defaultdict(bool)
    for (t, g), p in ev.items():
        task_pass[t] |= p
    rows = []
    for f in sorted(glob.glob(f'{ROOT}/{m}/samples/*.lp.npz')):
        base = os.path.basename(f)[:-len('.lp.npz')]
        task, _, gid = base.rpartition('__')
        key = (task, int(gid))
        if key not in ev:
            continue
        z = np.load(f)
        try:
            ent = float(z['entropy'].mean())
            clp = float(z['chosen_logprob'].mean())
            T = int(len(z['chosen_ids']))
        except Exception:
            continue
        rows.append({'task': task, 'group': tasks[task]['design_group'],
                     'passed': ev[key], 'recoverable': task_pass[task],
                     'x': np.array([T, ent, clp], dtype=np.float32)})
    return rows


def group_kfold(rows, k=10, seed=42):
    rng = np.random.RandomState(seed)
    groups = sorted({r['group'] for r in rows})
    rng.shuffle(groups)
    buckets = [[] for _ in range(k)]
    for i, g in enumerate(groups):
        buckets[i % k].append(g)
    folds = []
    for b in buckets:
        bs = set(b)
        te = [i for i, r in enumerate(rows) if r['group'] in bs]
        tr = [i for i, r in enumerate(rows) if r['group'] not in bs]
        if te and tr:
            folds.append((tr, te))
    return folds


def cv_auc(rows, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    folds = group_kfold(rows)
    oof = np.full(len(rows), np.nan)
    for tr, te in folds:
        Xtr = np.stack([rows[i]['x'] for i in tr])
        Xte = np.stack([rows[i]['x'] for i in te])
        ytr = np.array([int(rows[i][label]) for i in tr])
        yte = np.array([int(rows[i][label]) for i in te])
        if len(set(ytr)) < 2 or len(set(yte)) < 2:
            continue
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(C=1, solver='liblinear', max_iter=2000)
        clf.fit(sc.transform(Xtr), ytr)
        oof[te] = clf.predict_proba(sc.transform(Xte))[:, 1]
    mask = ~np.isnan(oof)
    y = np.array([int(r[label]) for r in rows])
    if mask.sum() == 0 or len(set(y[mask])) < 2:
        return float('nan')
    return float(roc_auc_score(y[mask], oof[mask]))


def main():
    out = {}
    print(f"{'模型':28s} {'Q1标量':>7s} {'Q2标量':>7s} {'n':>6s}")
    for m in MODELS:
        rows = load_scalar_rows(m)
        if len(rows) < 100:
            print(f'{m:28s} 数据不足 {len(rows)}')
            continue
        a1 = cv_auc(rows, 'passed')
        q2rows = [r for r in rows if not r['passed']]
        a2 = cv_auc(q2rows, 'recoverable') if len(q2rows) > 50 else float('nan')
        out[m] = {'Q1_scalar_LR': round(a1, 4), 'Q2_scalar_LR': round(a2, 4),
                  'n_rows': len(rows), 'n_q2': len(q2rows)}
        print(f"{m:28s} {a1:7.3f} {a2:7.3f} {len(rows):6d}", flush=True)
    json.dump(out, open(f'{ROOT}/probe_scalar_results.json', 'w'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
