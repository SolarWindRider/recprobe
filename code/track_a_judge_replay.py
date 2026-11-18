#!/usr/bin/env python3
"""Track A 扩展:LLM-as-judge(self-eval)基线回放。

策略与 scripts/step2_baselines_sim.py 的 judge@B 逐行一致:
  门控: 当前保留候选的 judge 分数 < 0.5 → 续采下一条(预算 B)
  提交: kept 中 judge 分数最高者(平局取最早)
judge 分数与折无关(IID 各折与 OOD 各折模型用同一分数,同 sc 基线口径)。

输出: 追加 judge@2/4/8 行到 predictions/decisions_extra.csv
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, OUT, canonical_rows
from track_a_scalar import load_meta_light

BS = [2, 4, 8]


def judge_cascade(js, B):
    """step2_baselines_sim judge@B 策略的精确复制(候选按 gid 升序)。"""
    kept = [0]
    for i in range(1, min(B, len(js))):
        if js[kept[-1]] < 0.5:
            kept.append(i)
        else:
            break
    sub = max(kept, key=lambda i: js[i])
    return kept, sub


def stage_replay(models):
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rows = canonical_rows()
    out = []
    for m in models:
        jp = f'{OUT}/judge_scores/{m}.json'
        if not os.path.exists(jp):
            print('skip (no judge scores):', m); continue
        judge = json.load(open(jp))
        meta = load_meta_light(m, rows)
        task_rows = defaultdict(list)
        for i, t in enumerate(meta['task']):
            task_rows[t].append(i)
        for t in task_rows:
            task_rows[t].sort(key=lambda i: meta['gid'][i])

        def run(tasks, fold, domain):
            for t in tasks:
                idx = task_rows[t]
                if not idx:
                    continue
                passed = meta['passed'][idx]; glen = meta['gen_len'][idx]
                js = [judge.get(t, {}).get(str(int(meta['gid'][idx[i]])), 0.5)
                      for i in range(len(idx))]
                for B in BS:
                    kept, sub = judge_cascade(js, B)
                    out.append({'model': m, 'domain': domain, 'fold': fold, 'task': t,
                                'group': meta['group'][idx[0]], 'method': 'judge', 'B': str(B),
                                'submitted_pass': int(passed[sub]), 'used': len(kept),
                                'gen_tokens': int(sum(glen[i] for i in kept))})
        for k in range(5):
            run(folds['folds'][f'fold{k}']['test_tasks'], f'fold{k}', 'IID')
            run(folds['ood_rtllm_tasks'], f'fold{k}', 'OOD')
        print(m, 'judge decision rows:', len(out), flush=True)
    p = f'{OUT}/predictions/decisions_extra.csv'
    write_header = not os.path.exists(p)
    with open(p, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'domain', 'fold', 'task', 'group',
                                          'method', 'B', 'submitted_pass', 'used', 'gen_tokens'])
        if write_header:
            w.writeheader()
        w.writerows(out)
    print('appended', p, len(out), 'rows')


if __name__ == '__main__':
    ms = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    stage_replay(ms)
