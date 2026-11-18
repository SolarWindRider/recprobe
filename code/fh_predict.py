#!/usr/bin/env python3
"""V1-E2 Step 2+3: uncertainty-signature features + FH early prediction.

Corpus: experiments/uncertainty_sampling (3,080 gens, T=0.8, npz per token).
Task: among generations that COMPILE (labels functional/none), predict from
the first w fraction of tokens whether the final verdict is a functional
hallucination (FH, label=functional) vs pass (label=none).

Features (computed on the token prefix only — no leakage):
  entropy:   mean / std / max / max 8-token jump / argmax position (frac)
  chosen_logprob: mean / min
  topk margin (p1-p2): mean / min
  chosen_rank: mean / frac(rank>1)
  structure: code-start token fraction (first 'module'/'always'/'assign'
             via approximate per-token decode lengths), entropy_mean after
             code start / before
Windows w in {0.2, 0.5, 0.7, 0.8} (features from prefix only) + 1.0.
Splits: leave-one-model-out (7 folds), leave-one-task-out (44 folds).
Classifier: standardized logistic regression. Baseline: gen_len only.

Output: experiments/fh_predict/features.jsonl + auroc.json + console table.
"""
import glob
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
MODELS = ['CodeV-QW-7B', 'VeriCoder_Qwen14B', 'CodeV-R1-Distill-Qwen-7B',
          'CodeV-R1-RL-Qwen-7B', 'RTLCoder-Deepseek-v1.1',
          'CodeV-DS-6.7B', 'DeepSeek-Coder-6.7B']
WINDOWS = [0.2, 0.5, 0.7, 0.8, 1.0]
OUT = 'experiments/fh_predict'
CODE_MARK = re.compile(r'\b(module|always|assign)\b')


def approx_token_len(tok, tid):
    return max(1, len(tok.decode([int(tid)])))


def code_start_frac(tok, ids, text):
    m = CODE_MARK.search(text)
    if not m:
        return 1.0
    # approximate token index via cumulative single-token decode lengths
    acc = 0
    for i, t in enumerate(ids):
        acc += approx_token_len(tok, t)
        if acc >= m.start():
            return i / max(len(ids), 1)
    return 1.0


def feats_at(n, ent, clp, margin, rank, csf):
    k = max(int(n), 1)
    e = ent[:k]
    c = clp[:k]
    m = margin[:k]
    r = rank[:k]
    p1 = np.exp(m if False else 0)  # placeholder no-op
    jumps = np.abs(np.diff(e)) if len(e) > 1 else np.zeros(1)
    return {
        'H_mean': float(np.mean(e)), 'H_std': float(np.std(e)),
        'H_max': float(np.max(e)),
        'H_p90': float(np.percentile(e, 90)),
        'H_jump': float(np.max(jumps)),
        'spike_density': float(np.mean(jumps > 1.0)),
        'H_argmax_pos': float(np.argmax(e)) / k,
        'clp_mean': float(np.mean(c)), 'clp_min': float(np.min(c)),
        'margin_mean': float(np.mean(m)), 'margin_min': float(np.min(m)),
        'margin_p10': float(np.percentile(m, 10)),
        'rank_mean': float(np.mean(r)),
        'rank_gt1_frac': float(np.mean(r > 1)),
        'csf': csf,
        'H_after_code_mean': float(np.mean(e[int(csf * k):])) if int(csf * k) < k else float(np.mean(e)),
        'H_before_code_mean': float(np.mean(e[:int(csf * k)])) if int(csf * k) > 0 else 0.0,
        'H_code_contrast': float(np.mean(e[int(csf * k):]) - np.mean(e[:int(csf * k)])) if 0 < int(csf * k) < k else 0.0,
        'len_tokens': n,
    }


def build_dataset():
    from transformers import AutoTokenizer
    rows = []
    for m in MODELS:
        tok = AutoTokenizer.from_pretrained(f'models/{m}', trust_remote_code=True)
        ev = {(r['task'], r['generation_id']): r['hallucination_type']
              for r in map(json.loads,
                           open(f'experiments/uncertainty_sampling/{m}/eval.jsonl'))}
        for jf in sorted(glob.glob(
                f'experiments/uncertainty_sampling/{m}/samples/*.json')):
            j = json.load(open(jf))
            lab = ev.get((j['task'], j['generation_id']))
            if lab not in ('functional', 'none'):
                continue
            ids = j['output_token_ids']
            if not ids:
                continue
            nf = jf[:-5] + '.lp.npz'
            z = np.load(nf)
            ent = z['entropy'].astype(np.float64)
            clp = z['chosen_logprob'].astype(np.float64)
            tk_lp = z['topk_logprobs']
            margin = np.exp(tk_lp[:, 0]) - np.exp(tk_lp[:, 1]) \
                if tk_lp.shape[1] >= 2 else np.zeros(len(ent))
            rank = z['chosen_rank']
            text = j['output_text']
            csf = code_start_frac(tok, ids, text)
            n = len(ids)
            row = {'model': m, 'task': j['task'],
                   'gid': j['generation_id'],
                   'hash': hashlib.md5((m + text).encode()).hexdigest(),
                   'label': 1 if lab == 'functional' else 0}
            for w in WINDOWS:
                f = feats_at(max(1, int(n * w)), ent, clp, margin, rank, csf)
                for k2, v in f.items():
                    row[f'{k2}@{w}'] = v
            rows.append(row)
        print(m, 'done', flush=True)
    # content dedup (identical output text within a model)
    seen, out = set(), []
    for r in rows:
        if r['hash'] in seen:
            continue
        seen.add(r['hash'])
        out.append(r)
    return out


def auroc(y, s):
    y = np.asarray(y)
    s = np.asarray(s)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    order = np.argsort(np.concatenate([neg, pos]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    # average ranks for ties
    allv = np.concatenate([neg, pos])
    for v in np.unique(allv):
        mask = allv == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    rp = ranks[len(neg):].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2) /
                 (len(pos) * len(neg)))


def main():
    os.makedirs(OUT, exist_ok=True)
    fpath = os.path.join(OUT, 'features.jsonl')
    if os.path.exists(fpath):
        rows = [json.loads(l) for l in open(fpath)]
    else:
        rows = build_dataset()
        with open(fpath, 'w') as f:
            for r in rows:
                f.write(json.dumps(r) + '\n')
    print(f'dataset: {len(rows)} unique rows, '
          f'L2={sum(r["label"] for r in rows)}, '
          f'pass={sum(1-r["label"] for r in rows)}')

    feat_cols = [c for c in rows[0]
                 if re.search(r'@\d|\@1\.0$|@0\.', c)] if False else \
        [c for c in rows[0] if '@' in c]
    # prefix-only feature sets per window
    results = {}
    for scheme in ['LOMO', 'LOTO']:
        groups = ({r['model'] for r in rows} if scheme == 'LOMO' else
                  {r['task'] for r in rows})
        per_w = {w: [] for w in WINDOWS}
        for g in groups:
            if scheme == 'LOMO':
                tr = [r for r in rows if r['model'] != g]
                te = [r for r in rows if r['model'] == g]
            else:
                tr = [r for r in rows if r['task'] != g]
                te = [r for r in rows if r['task'] == g]
            if not te or len({r['label'] for r in te}) < 2:
                continue
            ytr = np.array([r['label'] for r in tr])
            yte = np.array([r['label'] for r in te])
            for w in WINDOWS:
                cols = [c for c in feat_cols if c.endswith(f'@{w}')]
                Xtr = np.array([[r[c] for c in cols] for r in tr])
                Xte = np.array([[r[c] for c in cols] for r in te])
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
                Xtr = (Xtr - mu) / sd
                Xte = (Xte - mu) / sd
                s = Xte.mean(1)  # logistic via simple linear score
                # proper LR
                from numpy.linalg import inv
                Xb = np.hstack([Xtr, np.ones((len(Xtr), 1))])
                beta = np.zeros(Xb.shape[1])
                for _ in range(200):
                    p = 1 / (1 + np.exp(-Xb @ beta))
                    W = p * (1 - p)
                    H = Xb.T @ (Xb * W[:, None]) + 1e-6 * np.eye(Xb.shape[1])
                    beta = beta + inv(H) @ (Xb.T @ (ytr - p))
                Xbe = np.hstack([Xte, np.ones((len(Xte), 1))])
                score = Xbe @ beta
                per_w[w].append(auroc(yte, score))
        results[scheme] = {str(w): {
            'mean': float(np.nanmean(per_w[w])),
            'std': float(np.nanstd(per_w[w])), 'folds': len(per_w[w])}
            for w in WINDOWS}
    # length-only baseline at w=1.0 (window-independent)
    y = np.array([r['label'] for r in rows])
    s = np.array([r['len_tokens@1.0'] for r in rows])
    results['length_baseline_auroc'] = auroc(y, s)

    # 非线性对照:HistGradientBoosting,LOMO 切分
    from sklearn.ensemble import HistGradientBoostingClassifier
    per_w = {w: [] for w in WINDOWS}
    for g in sorted({r['model'] for r in rows}):
        tr = [r for r in rows if r['model'] != g]
        te = [r for r in rows if r['model'] == g]
        if len({r['label'] for r in te}) < 2:
            continue
        ytr = np.array([r['label'] for r in tr])
        yte = np.array([r['label'] for r in te])
        for w in WINDOWS:
            cols = [c for c in feat_cols if c.endswith(f'@{w}')]
            Xtr = np.array([[r[c] for c in cols] for r in tr])
            Xte = np.array([[r[c] for c in cols] for r in te])
            clf = HistGradientBoostingClassifier(max_iter=200, random_state=0)
            clf.fit(Xtr, ytr)
            per_w[w].append(auroc(yte, clf.predict_proba(Xte)[:, 1]))
    results['LOMO_GBM'] = {str(w): {
        'mean': float(np.nanmean(per_w[w])), 'std': float(np.nanstd(per_w[w]))}
        for w in WINDOWS}

    json.dump(results, open(os.path.join(OUT, 'auroc.json'), 'w'), indent=1)
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
