#!/usr/bin/env python3
"""Track A 扩展:标量特征档 RecProbe(黑盒两特征档中的廉价档)。

与残差档完全同协议,仅特征不同:
  特征  X = [T_out, mean_entropy, mean_chosen_logprob](来自 samples/*.lp.npz,零推理开销)
  探针  每折重训 LR(C=1, liblinear, max_iter=2000, seed42)+ StandardScaler(折内拟合)
        Q1: 折训练设计全部样本 label=passed;Q2: 折训练设计失败样本 label=recoverable
  回放  与 track_a_replay.cascade 逐行一致的控制器;方法名 recprobe_s / q1only_s

输出:
  predictions/scalar_features/<m>.npz   (X 11584x3 缓存)
  predictions/scalar/<m>.npz            (p1,p2 = 5x11584, NaN=不可训练折)
  models/<m>_fold<k>_scalar_q{1,2}.joblib + models/<m>_scalar_train_info.json
  decisions 行由 --replay 追加到 predictions/decisions_extra.csv
"""
import csv, json, os, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, OUT, canonical_rows, tasks_dict
from track_a_score import fit_lr

STEP2 = './experiments/step2'
BS = [2, 4, 8]


def load_meta_light(m, rows):
    """与 track_a_replay.load_meta_light 一致(只读标签,不进决策)。"""
    ev = {}
    for r in map(json.loads, open(f'{STEP2}/{m}/eval.jsonl')):
        ev[(r['task'], r['generation_id'])] = bool(r['simulation_pass'])
    glen = {}
    for r in map(json.loads, open(f'{STEP2}/{m}/labels.jsonl')):
        glen[(r['task'], r['generation_id'])] = int(r['gen_len'])
    task_pass = defaultdict(bool)
    for (t, g), p in ev.items():
        task_pass[t] |= p
    tasks = tasks_dict()
    meta = {'task': [], 'gid': [], 'group': [], 'passed': [], 'rec': [], 'gen_len': []}
    for t, g in rows:
        meta['task'].append(t); meta['gid'].append(g)
        meta['group'].append(tasks[t]['design_group'])
        meta['passed'].append(bool(ev[(t, g)]))
        meta['rec'].append(bool(task_pass[t]))
        meta['gen_len'].append(glen[(t, g)])
    return {k: np.array(v) for k, v in meta.items()}


def scalar_features(m, rows, cache_dir=None):
    """X[i] = [len(chosen_ids), mean(entropy), mean(chosen_logprob)],行序=rows。"""
    cache = cache_dir or f'{OUT}/predictions/scalar_features'
    os.makedirs(cache, exist_ok=True)
    p = f'{cache}/{m}.npz'
    if os.path.exists(p):
        z = np.load(p)
        return z['X']
    key2i = {(t, g): i for i, (t, g) in enumerate(rows)}
    X = np.full((len(rows), 3), np.nan, dtype=np.float32)
    n = 0
    for f in sorted(os.listdir(f'{STEP2}/{m}/samples')):
        if not f.endswith('.lp.npz'):
            continue
        base = f[:-len('.lp.npz')]
        task, _, gid = base.rpartition('__')
        i = key2i.get((task, int(gid)))
        if i is None:
            continue
        z = np.load(f'{STEP2}/{m}/samples/{f}')
        try:
            X[i] = (len(z['chosen_ids']), float(z['entropy'].mean()),
                    float(z['chosen_logprob'].mean()))
            n += 1
        except Exception:
            pass
    assert n == len(rows), f'{m}: scalar features missing {len(rows)-n}'
    np.savez_compressed(p, X=X)
    return X


def cascade(p1v, p2v, B, q1_only=False):
    """track_a_replay.cascade 的精确复制。"""
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


def stage_score(models):
    """每折重训标量探针并给全量样本打分(与 track_a_score.stage_score 同协议)。"""
    import joblib
    from sklearn.metrics import roc_auc_score
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rows = canonical_rows()
    os.makedirs(f'{OUT}/predictions/scalar', exist_ok=True)
    os.makedirs(f'{OUT}/models', exist_ok=True)
    for m in models:
        t0 = time.time()
        X = scalar_features(m, rows)
        meta = load_meta_light(m, rows)
        group = meta['group']; passed = meta['passed'].astype(int)
        rec = meta['rec'].astype(int)
        P1 = np.full((5, 11584), np.nan, dtype=np.float32)
        P2 = np.full((5, 11584), np.nan, dtype=np.float32)
        info = {'model': m, 'tier': 'scalar [T_out, mean_entropy, mean_chosen_logprob]',
                'folds': {}}
        for k in range(5):
            tr_groups = set(folds['folds'][f'fold{k}']['train_groups'])
            trm = np.array([g in tr_groups for g in group])
            r1 = fit_lr(X[trm], passed[trm])
            fq = trm & (passed == 0)
            if r1 is None:
                print(f'{m} fold{k}: UNTRAINABLE (single-class)', flush=True)
                info['folds'][f'fold{k}'] = {'untrainable': True}
                continue
            sc, clf = r1
            P1[k] = clf.predict_proba(sc.transform(X))[:, 1]
            auc1 = roc_auc_score(passed[trm], P1[k][trm])
            joblib.dump({'scaler': sc, 'clf': clf, 'label': 'passed'},
                        f'{OUT}/models/{m}_fold{k}_scalar_q1.joblib')
            r2 = fit_lr(X[fq], rec[fq])
            if r2 is not None:
                sc2, clf2 = r2
                P2[k] = clf2.predict_proba(sc2.transform(X))[:, 1]
                auc2 = roc_auc_score(rec[fq], P2[k][fq])
                joblib.dump({'scaler': sc2, 'clf': clf2, 'label': 'recoverable'},
                            f'{OUT}/models/{m}_fold{k}_scalar_q2.joblib')
            else:
                auc2 = None
            info['folds'][f'fold{k}'] = {
                'n_train_rows_q1': int(trm.sum()), 'n_train_rows_q2': int(fq.sum()),
                'train_auroc_q1': round(float(auc1), 4),
                'train_auroc_q2': None if auc2 is None else round(float(auc2), 4)}
            print(f'{m} fold{k}: scalar q1_tr_auc={auc1:.4f} '
                  f"q2_tr_auc={auc2 if auc2 is None else round(auc2,4)}", flush=True)
        np.savez_compressed(f'{OUT}/predictions/scalar/{m}.npz', p1=P1, p2=P2)
        json.dump(info, open(f'{OUT}/models/{m}_scalar_train_info.json', 'w'), indent=1)
        print(f'{m} done in {time.time()-t0:.0f}s -> predictions/scalar/{m}.npz', flush=True)


def stage_replay(models):
    """回放 recprobe_s / q1only_s(与 track_a_replay 同折叠结构),追加 decisions_extra.csv。"""
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rows = canonical_rows()
    out = []
    for m in models:
        p = f'{OUT}/predictions/scalar/{m}.npz'
        if not os.path.exists(p):
            print('skip (no scalar scores):', m); continue
        z = np.load(p)
        P1, P2 = z['p1'], z['p2']
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
                p1v = P1[k][idx].astype(np.float64)
                p2v = P2[k][idx].astype(np.float64)
                for B in BS:
                    for meth, pv in [('recprobe_s', (p1v, p2v)), ('q1only_s', (p1v, None))]:
                        kept = cascade(pv[0], pv[1], B, q1_only=(pv[1] is None))
                        sub = int(np.nanargmax(pv[0][kept]))
                        out.append({'model': m, 'domain': domain, 'fold': fold, 'task': t,
                                    'group': meta['group'][idx[0]], 'method': meth, 'B': str(B),
                                    'submitted_pass': int(passed[sub]), 'used': len(kept),
                                    'gen_tokens': int(sum(glen[i] for i in kept))})
        for k in range(5):
            run(folds['folds'][f'fold{k}']['test_tasks'], f'fold{k}', 'IID')
            run(folds['ood_rtllm_tasks'], f'fold{k}', 'OOD')
        print(m, 'scalar decision rows:', len(out), flush=True)
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
    ms = sys.argv[2].split(',') if len(sys.argv) > 2 else MODELS
    if sys.argv[1] == 'score':
        stage_score(ms)
    elif sys.argv[1] == 'replay':
        stage_replay(ms)
    else:
        print(__doc__)
