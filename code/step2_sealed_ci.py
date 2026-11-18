#!/usr/bin/env python3
"""密封 Q2 AUC 的任务级自助置信区间(scalar + residual, iid/shift)。"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step2_sealed_eval import MODELS, load_rows, make_split  # noqa: E402

ROOT = 'experiments/step2'
split = make_split()
calib_g = set(split['calib'])
seals = {'iid': set(split['sealed_iid']), 'shift': set(split['sealed_shift'])}


def main():
    out = {}
    for tier in ['scalar', 'residual']:
        for m in MODELS:
            rows = load_rows(m, tier)
            cal_fail = [r for r in rows
                        if r['group'] in calib_g and not r['passed']]
            rng = np.random.RandomState(7)
            groups = sorted({r['group'] for r in cal_fail})
            pick = set(rng.choice(groups, size=min(50, len(groups)),
                                  replace=False))
            tr = [r for r in cal_fail if r['group'] in pick]
            for seal, ss in seals.items():
                te = [r for r in rows if r['group'] in ss and not r['passed']]
                by_task = defaultdict(list)
                # 用固定模型给密封行打分
                from sklearn.linear_model import LogisticRegression
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import roc_auc_score
                Xtr = np.stack([r['x'] for r in tr])
                sc = StandardScaler().fit(Xtr)
                clf = LogisticRegression(C=1, solver='liblinear',
                                         max_iter=2000)
                clf.fit(sc.transform(Xtr), np.array([int(r['rec'])
                                                     for r in tr]))
                for r in te:
                    p = float(clf.predict_proba(
                        sc.transform(r['x'][None]))[0, 1])
                    by_task[r['task']].append((int(r['rec']), p))
                tasks = list(by_task)
                y = np.array([v[0] for t in tasks for v in by_task[t]])
                s = np.array([v[1] for t in tasks for v in by_task[t]])
                if len(set(y)) < 2:
                    continue
                point = roc_auc_score(y, s)
                brng = np.random.RandomState(0)
                bs = []
                for _ in range(1000):
                    idx = brng.randint(len(tasks), size=len(tasks))
                    yy = np.array([v[0] for i in idx for v in by_task[tasks[i]]])
                    pp = np.array([v[1] for i in idx for v in by_task[tasks[i]]])
                    if len(set(yy)) < 2:
                        continue
                    bs.append(roc_auc_score(yy, pp))
                lo, hi = np.percentile(bs, [2.5, 97.5])
                out.setdefault(m, {})[f'{tier}_{seal}'] = {
                    'auc': round(float(point), 3), 'n_tasks': len(tasks),
                    'ci95': [round(float(lo), 3), round(float(hi), 3)]}
                print(f"{m:28s} {tier:8s} {seal:5s} AUC={point:.3f} "
                      f"[{lo:.3f},{hi:.3f}] ({len(tasks)} 任务)", flush=True)
    json.dump(out, open(f'{ROOT}/sealed_ci.json', 'w'),
              ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
