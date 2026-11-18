#!/usr/bin/env python3
"""密封 OOD 协议(最终评估专用)。

任务分割(按 design_group 冻结, 分割表落盘):
  CALIB   : VE/CC 设计组 120/156 (240 任务) —— 校准/调参/特征选择随意用
  SEALED_IID : VE/CC 设计组 36/156 (72 任务) —— 同分布密封集, 只许最终评估
  SEALED_SHIFT: RTLLM 全部 44 任务 —— 分布偏移密封集(RTLLM 风格 ≠ VE)

流程: 每模型在 CALIB 上做 k-shot 校准(k=5..100, 纯目标协议)→
      在两个密封集上各评估一次(k=50 校准), 绝不回头调参。
特征: 标量 3 特征 + 残差流 w100@L27(LR)。
"""
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = 'experiments/step2'
MODELS = ['CodeV-QW-7B', 'CodeV-DS-6.7B', 'VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B', 'CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1', 'DeepSeek-Coder-6.7B']
KS = [5, 10, 20, 50, 100]
SPLIT_FILE = f'{ROOT}/sealed_split.json'


def make_split(seed=42):
    if os.path.exists(SPLIT_FILE):
        return json.load(open(SPLIT_FILE))
    tasks = json.load(open(f'{ROOT}/tasks_step2.json'))
    ve_groups = sorted({t['design_group'] for t in tasks
                        if t['design'].startswith('spec') or
                        t['design'].startswith('code')})
    rtllm_groups = sorted({t['design_group'] for t in tasks
                           if t['design_group'].startswith('rtllm')})
    rng = np.random.RandomState(seed)
    rng.shuffle(ve_groups)
    n_seal = len(ve_groups) * 36 // 156
    split = {'calib': ve_groups[n_seal:],
             'sealed_iid': ve_groups[:n_seal],
             'sealed_shift': rtllm_groups}
    json.dump(split, open(SPLIT_FILE, 'w'), indent=1)
    return split


def load_rows(m, tier):
    """tier: 'scalar' 或 'residual'。"""
    tasks = {t['task_id']: t for t in json.load(open(f'{ROOT}/tasks_step2.json'))}
    ev = {}
    for r in map(json.loads, open(f'{ROOT}/{m}/eval.jsonl')):
        ev[(r['task'], r['generation_id'])] = bool(r['simulation_pass'])
    task_pass = defaultdict(bool)
    for (t, g), p in ev.items():
        task_pass[t] |= p
    rows = []
    if tier == 'scalar':
        for f in sorted(glob.glob(f'{ROOT}/{m}/samples/*.lp.npz')):
            base = os.path.basename(f)[:-len('.lp.npz')]
            task, _, gid = base.rpartition('__')
            key = (task, int(gid))
            if key not in ev:
                continue
            z = np.load(f)
            try:
                x = np.array([int(len(z['chosen_ids'])),
                              float(z['entropy'].mean()),
                              float(z['chosen_logprob'].mean())],
                             dtype=np.float32)
            except Exception:
                continue
            rows.append({'task': task,
                         'group': tasks[task]['design_group'],
                         'passed': ev[key], 'rec': task_pass[task], 'x': x})
    else:
        for f in sorted(glob.glob(f'{ROOT}/{m}/samples/*.json')):
            if f.endswith('.lp.npz'):
                continue
            base = os.path.basename(f)[:-len('.json')]
            task, _, gid = base.rpartition('__')
            key = (task, int(gid))
            if key not in ev:
                continue
            h = hashlib.md5((m + task + gid).encode()).hexdigest()[:16]
            zf = f'{ROOT}/hidden_stream/{m}/{h}.npz'
            if not os.path.exists(zf):
                continue
            z = np.load(zf, allow_pickle=False)
            man = json.loads(str(z['pos_manifest']))
            tk = {tg: k for k, tags in man.items() for tg in tags}
            if 'w100' not in tk:
                continue
            rows.append({'task': task,
                         'group': tasks[task]['design_group'],
                         'passed': ev[key], 'rec': task_pass[task],
                         'x': z[f"{tk['w100']}@L27"].astype(np.float32)})
    return rows


def fit_eval(tr, te, label):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    Xtr = np.stack([r['x'] for r in tr])
    Xte = np.stack([r['x'] for r in te])
    ytr = np.array([int(r[label]) for r in tr])
    yte = np.array([int(r[label]) for r in te])
    if len(set(ytr)) < 2 or len(set(yte)) < 2:
        return float('nan')
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1, solver='liblinear', max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def main():
    split = make_split()
    calib_g = set(split['calib'])
    seal_sets = {'iid': set(split['sealed_iid']),
                 'shift': set(split['sealed_shift'])}
    print(f"分割: CALIB {len(calib_g)} 组 | SEALED_IID {len(seal_sets['iid'])} 组"
          f" | SEALED_SHIFT {len(seal_sets['shift'])} 组\n")
    out = {}
    for tier in ['scalar', 'residual']:
        print(f'===== 特征档: {tier} =====')
        for m in MODELS:
            rows = load_rows(m, tier)
            cal = [r for r in rows if r['group'] in calib_g]
            cal_fail = [r for r in cal if not r['passed']]
            res = {}
            for label in ['passed', 'rec']:
                src = rows if label == 'passed' else cal_fail
                # k-shot 曲线: 在 CALIB 内部留出评估
                rng = np.random.RandomState(42)
                groups = sorted({r['group'] for r in src})
                curve = {}
                for k in KS:
                    aucs = []
                    for _ in range(5):
                        pick = set(rng.choice(groups,
                                              size=min(k, len(groups)),
                                              replace=False))
                        tr = [r for r in src if r['group'] in pick]
                        te = [r for r in src if r['group'] not in pick]
                        aucs.append(fit_eval(tr, te, label))
                    curve[k] = round(float(np.nanmean(aucs)), 3)
                res[f'{label}_calib_curve'] = curve
                # 密封终评: k=50 校准(纯目标), 密封集各评一次
                rng = np.random.RandomState(7)
                pick = set(rng.choice(groups, size=min(50, len(groups)),
                                      replace=False))
                tr = [r for r in src if r['group'] in pick]
                for name, ss in seal_sets.items():
                    te = [r for r in rows if r['group'] in ss]
                    if label == 'rec':
                        te = [r for r in te if not r['passed']]
                    res[f'{label}_sealed_{name}'] = round(
                        fit_eval(tr, te, label), 3)
            out.setdefault(m, {})[tier] = res
            print(f"{m:28s} Q2 calib(k50)={res['rec_calib_curve'][50]:.3f} "
                  f"sealed_iid={res['rec_sealed_iid']:.3f} "
                  f"sealed_shift={res['rec_sealed_shift']:.3f} | "
                  f"Q1 sealed_iid={res['passed_sealed_iid']:.3f} "
                  f"sealed_shift={res['passed_sealed_shift']:.3f}", flush=True)
    json.dump(out, open(f'{ROOT}/sealed_eval_results.json', 'w'),
              ensure_ascii=False, indent=1)
    print('->', f'{ROOT}/sealed_eval_results.json')


if __name__ == '__main__':
    main()
