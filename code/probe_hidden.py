#!/usr/bin/env python3
"""残差流探针:位置 × 层 AUROC 网格(LOTO),含置换对照。

对 7 模型各自训练线性探针(4096→1,标准化,LogisticRegression C=1),
留一任务切分;输出每 (位置, 层) 的 AUROC mean±std。
置换对照:在 (w10, L27) 做 5 次标签置换的完整 LOTO,必须回落 0.5。
输出: experiments/hidden_stream/probe_auroc.json
"""
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODELS = ['CodeV-QW-7B', 'VeriCoder_Qwen14B', 'CodeV-R1-Distill-Qwen-7B',
          'CodeV-R1-RL-Qwen-7B', 'RTLCoder-Deepseek-v1.1',
          'CodeV-DS-6.7B', 'DeepSeek-Coder-6.7B']
POS = ['pre', 'first'] + [f's{s}' for s in
       (100, 200, 300, 400, 500, 600, 700, 800, 900, 1000)] + \
      ['w02', 'w05', 'w07', 'w08', 'w10']
LAYERS = [7, 14, 21, 27]


def load_model(m):
    """每模型: hash -> (task, label, {tag: {layer: vec}})"""
    ev_dir = f'experiments/uncertainty_sampling/{m}'
    ev = {(r['task'], r['generation_id']): r['hallucination_type']
          for r in map(json.loads, open(f'{ev_dir}/eval.jsonl'))}
    hash2task = {}
    for jf in glob.glob(f'{ev_dir}/samples/*.json'):
        j = json.load(open(jf))
        h = hashlib.md5((m + j['task'] + str(j['generation_id'])).encode()
                        ).hexdigest()[:16]
        hash2task[h] = j['task']
    rows = []
    for f in glob.glob(f'experiments/hidden_stream/{m}/*.npz'):
        h = os.path.basename(f)[:-4]
        task = hash2task.get(h)
        if task is None:
            continue
        z = np.load(f, allow_pickle=False)
        man = json.loads(str(z['pos_manifest']))
        tag2key = {}
        for key, tags in man.items():
            for tg in tags:
                tag2key[tg] = key
        sig = {f'{tg}@L{L}': z[f'{tag2key[tg]}@L{L}'].astype(np.float32)
               for tg in tag2key for L in LAYERS
               if f'{tag2key[tg]}@L{L}' in z}
        rows.append({'task': task,
                     'label': int(z['label'][0]),
                     'sig': sig})
    return rows


def eval_cell(args):
    rows, tag, L, perms = args
    from sklearn.linear_model import LogisticRegression
    key = f'{tag}@L{L}'
    data = [(r['task'], r['label'], r['sig'][key]) for r in rows
            if key in r['sig']]
    if len(data) < 50:
        return (tag, L, len(data), None, None)
    tasks = sorted({t for t, _, _ in data})
    folds = []
    for g in tasks:
        tr = [d for d in data if d[0] != g]
        te = [d for d in data if d[0] == g]
        yte = {lab for _, lab, _ in te}
        if len(tr) == 0 or len(yte) < 2:
            continue
        Xtr = np.stack([s for _, _, s in tr])
        Xte = np.stack([s for _, _, s in te])
        ytr = np.array([lab for _, lab, _ in tr])
        yte = np.array([lab for _, lab, _ in te])
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        folds.append((Xtr, ytr, Xte, yte))

    def run_folds(seed=None):
        rng = np.random.RandomState(seed) if seed is not None else None
        aucs = []
        for Xtr, ytr, Xte, yte in folds:
            yt = ytr if rng is None else rng.permutation(ytr)
            clf = LogisticRegression(C=1.0, max_iter=500)
            clf.fit(Xtr, yt)
            from fh_predict import auroc
            aucs.append(auroc(yte, clf.predict_proba(Xte)[:, 1]))
        return float(np.nanmean(aucs)), float(np.nanstd(aucs)), len(aucs)

    m0, s0, k = run_folds()
    perm_means = []
    if perms:
        for i in range(perms):
            m1, _, _ = run_folds(seed=1000 + i)
            perm_means.append(m1)
    return (tag, L, len(data), {'mean': round(m0, 3), 'std': round(s0, 3),
                                'folds': k},
            {'perm_mean': round(float(np.mean(perm_means)), 3)} if perms else None)


def main():
    perms = '--perms' in sys.argv
    all_out = {}
    for m in MODELS:
        rows = load_model(m)
        if not rows:
            print(f'{m}: 无数据,跳过', flush=True)
            continue
        jobs = [(rows, tag, L, 5 if (perms and tag == 'w10' and L == 27) else 0)
                for tag in POS for L in LAYERS]
        cell_out = {}
        with ProcessPoolExecutor(max_workers=8) as ex:
            for tag, L, n, r, pr in ex.map(eval_cell, jobs):
                cell_out[f'{tag}@L{L}'] = {'n': n, 'auroc': r, 'perm': pr}
        all_out[m] = cell_out
        good = {k: v for k, v in cell_out.items() if v['auroc']}
        best = max(good, key=lambda k: good[k]['auroc']['mean'])
        print(f'{m}: best={best} {good[best]["auroc"]} (n={good[best]["n"]})',
              flush=True)
    json.dump(all_out, open('experiments/hidden_stream/probe_auroc.json',
                            'w'), indent=1)
    print('saved probe_auroc.json')


if __name__ == '__main__':
    main()
