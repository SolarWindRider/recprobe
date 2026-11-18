#!/usr/bin/env python3
"""10 随机任务切分 × 7 模型:PCA-128 + LR 的 A→B 验证,给出误差棒。"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_hidden import load_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from fh_predict import auroc

MODELS = ['CodeV-QW-7B', 'VeriCoder_Qwen14B', 'Qwen2.5-14B-Instruct', 'CodeV-R1-Distill-Qwen-7B',
          'CodeV-R1-RL-Qwen-7B', 'RTLCoder-Deepseek-v1.1',
          'CodeV-DS-6.7B', 'DeepSeek-Coder-6.7B']
SEEDS = list(range(10))
out = {}
def main():
    for m in MODELS:
        rows = load_model(m)
        tasks = sorted({r['task'] for r in rows})
        key = 'w100@L27'
        aucs = []
        for seed in SEEDS:
            rng = np.random.RandomState(seed)
            rng.shuffle(tasks)
            A, B = set(tasks[:len(tasks)//2]), set(tasks[len(tasks)//2:])
            dataA = [(r['task'], r['label'], r['sig'][key]) for r in rows if r['task'] in A and key in r['sig']]
            dataB = [(r['task'], r['label'], r['sig'][key]) for r in rows if r['task'] in B and key in r['sig']]
            if len(dataA) == 0 or len(dataB) == 0:
                continue
            XA = np.stack([s for _, _, s in dataA]); yA = np.array([l for _, l, _ in dataA])
            XB = np.stack([s for _, _, s in dataB]); yB = np.array([l for _, l, _ in dataB])
            pca = PCA(n_components=min(128, XA.shape[0]-1), random_state=seed).fit(XA)
            clf = LogisticRegression(C=1.0, max_iter=500).fit(pca.transform(XA), yA)
            aucs.append(auroc(yB, clf.predict_proba(pca.transform(XB))[:, 1]))
        out[m] = {'mean': round(float(np.mean(aucs)), 3),
                  'std': round(float(np.std(aucs)), 3),
                  'min': round(float(np.min(aucs)), 3), 'max': round(float(np.max(aucs)), 3)}
        print(m, out[m], flush=True)
    json.dump(out, open('experiments/hidden_stream/probe_seeds.json', 'w'), indent=1)

if __name__ == '__main__':
    main()
