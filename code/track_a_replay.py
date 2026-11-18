#!/usr/bin/env python3
"""Track A 阶段3:离线回放完整 RecProbe 控制器(IID OOF + RTLLM OOD)。

输出 predictions/decisions_trackA.csv:每行 = model×domain×fold×task×method×B 的
提交决策(submitted_gid, submitted_pass, used, gen_tokens)。真实执行标签只用于
事后判定提交结果,不进入任何决策。
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_a_common import MODELS, OUT, canonical_rows, tasks_dict, load_texts

BS = [2, 4, 8]
SCB = [2, 4, 8]

def load_meta_light(m, rows):
    tasks = tasks_dict()
    from collections import defaultdict
    ev = {}
    for r in map(json.loads, open(f'./experiments/step2/{m}/eval.jsonl')):
        ev[(r['task'], r['generation_id'])] = bool(r['simulation_pass'])
    glen = {}
    for r in map(json.loads, open(f'./experiments/step2/{m}/labels.jsonl')):
        glen[(r['task'], r['generation_id'])] = int(r['gen_len'])
    task_pass = defaultdict(bool)
    for (t, g), p in ev.items():
        task_pass[t] |= p
    meta = {'task': [], 'gid': [], 'group': [], 'passed': [], 'rec': [], 'gen_len': []}
    for t, g in rows:
        meta['task'].append(t); meta['gid'].append(g)
        meta['group'].append(tasks[t]['design_group'])
        meta['passed'].append(bool(ev[(t, g)])); meta['rec'].append(bool(task_pass[t]))
        meta['gen_len'].append(glen[(t, g)])
    return {k: np.array(v) for k, v in meta.items()}

def jaccard(a, b):
    A, B = set(a.split()), set(b.split())
    if not A and not B:
        return 1.0
    return len(A & B) / max(len(A | B), 1)

def cascade(p1v, p2v, B, q1_only=False):
    """原 sim_task cascade 的精确复制(任务局部向量)。

    p1v/p2v = 按 gid 排序的该任务样本分数;p2v=None 时为 Q1-only 消融。
    返回 kept(生成候选的位置列表,len<=B)。"""
    kept = [0]
    for i in range(1, len(p1v)):
        last = kept[-1]
        if q1_only:
            cont = p1v[last] < 0.5
        else:
            g = p2v[last]
            cont = p1v[last] < 0.5 and (np.isnan(g) or g > 0.5)
        if not cont:
            break
        kept.append(i)
        if len(kept) >= B:
            break
    return kept

def sc_pick(cands, texts):
    best_i, best_s = 0, -1.0
    for i in range(len(cands)):
        others = [j for j in range(len(cands)) if j != i]
        s = sum(jaccard(texts[cands[i]], texts[cands[j]]) for j in others) / max(len(others), 1)
        if s > best_s:
            best_s, best_i = s, i
    return best_i

def replay_model(m, folds, rows):
    z = np.load(f'{OUT}/predictions/{m}.npz')
    P1, P2 = z['p1'], z['p2']
    meta = load_meta_light(m, rows)
    texts = load_texts(m, rows)
    task_rows = defaultdict(list)
    for i, t in enumerate(meta['task']):
        task_rows[t].append(i)
    for t in task_rows:
        task_rows[t].sort(key=lambda i: meta['gid'][i])
    out = []
    def run(tasks, fold, domain, p1, p2):
        for t in tasks:
            idx = task_rows[t]
            if not idx:
                continue
            passed = meta['passed'][idx]; glen = meta['gen_len'][idx]
            recs = [('alone', '-', int(passed[0]), 1, int(glen[0]))]
            for B in BS:
                recs.append((f'retry@{B}', '-', int(passed[0]), B, int(glen[:B].sum())))
                recs.append((f'oracle@{B}', '-', int(any(passed[:B])), B, int(glen[:B].sum())))
            for B in SCB:
                cand = list(range(min(B, len(idx))))
                tlist = [texts[(t, int(meta['gid'][idx[i]]))]
                         for i in range(min(max(BS), len(idx)))]
                sub = sc_pick(cand, tlist)
                recs.append((f'sc@{B}', '-', int(passed[sub]), B, int(glen[:B].sum())))
            p1v = p1[idx].astype(np.float64); p2v = p2[idx].astype(np.float64)
            for B in BS:
                kept = cascade(p1v, p2v, B, q1_only=False)
                sub = int(np.argmax(p1v[kept]))
                recs.append((f'recprobe@{B}', '-', int(passed[sub]), len(kept),
                             int(sum(glen[i] for i in kept))))
                kept = cascade(p1v, None, B, q1_only=True)
                sub = int(np.argmax(p1v[kept]))
                recs.append((f'q1only@{B}', '-', int(passed[sub]), len(kept),
                             int(sum(glen[i] for i in kept))))
            for meth, _, sp, used, tok in recs:
                out.append({'model': m, 'domain': domain, 'fold': fold, 'task': t,
                            'group': meta['group'][idx[0]], 'method': meth.split('@')[0],
                            'B': meth.split('@')[1] if '@' in meth else '-',
                            'submitted_pass': sp, 'used': used, 'gen_tokens': tok})
    for k in range(5):
        p1, p2 = P1[k].astype(np.float64), P2[k].astype(np.float64)
        run(folds['folds'][f'fold{k}']['test_tasks'], f'fold{k}', 'IID', p1, p2)
        run(folds['ood_rtllm_tasks'], f'fold{k}', 'OOD', p1, p2)
    p1e, p2e = P1.mean(0).astype(np.float64), P2.mean(0).astype(np.float64)
    run(folds['ood_rtllm_tasks'], 'ens', 'OOD_ENS', p1e, p2e)
    return out

def main():
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rows = canonical_rows()
    allrows = []
    for m in MODELS:
        z = f'{OUT}/predictions/{m}.npz'
        if not os.path.exists(z):
            print('skip (no scores yet):', m); continue
        rr = replay_model(m, folds, rows)
        allrows += rr
        print(m, 'decision rows:', len(rr), flush=True)
    p = f'{OUT}/predictions/decisions_trackA.csv'
    with open(p, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'domain', 'fold', 'task', 'group',
                                          'method', 'B', 'submitted_pass', 'used', 'gen_tokens'])
        w.writeheader(); w.writerows(allrows)
    print('wrote', p, len(allrows), 'rows')

if __name__ == '__main__':
    main()
