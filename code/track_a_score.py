#!/usr/bin/env python3
"""Track A 阶段1:冻结折分配;阶段2:每折重训 Q1/Q2 并给全部样本打分。

用法:
  python3 track_a_score.py folds    # 生成 fold_assignments.json
  python3 track_a_score.py score [model ...]   # 逐模型:5 折训练+全量打分
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_a_common import (MODELS, OUT, build_folds, canonical_rows, tasks_dict,
                            load_model_matrix)

def stage_folds():
    folds, rtllm = build_folds(seed=42, k=5)
    tasks = tasks_dict()
    out = {"seed": 42, "k": 5,
           "method": "sorted 156 VE design_groups; RandomState(42).permutation; round-robin i%5",
           "folds": {}, "ood_rtllm_tasks": []}
    all_test = []
    for k in range(5):
        fk = f'fold{k}'
        gs = folds[fk]['groups']
        train_g = sorted(set().union(*[set(folds[f'fold{j}']['groups'])
                                       for j in range(5) if j != k]))
        test_tasks = sorted(t for t, v in tasks.items()
                            if v['design_group'] in set(gs))
        train_tasks = sorted(t for t, v in tasks.items()
                             if v['design_group'] in set(train_g))
        out['folds'][fk] = {"test_groups": gs, "train_groups": train_g,
                            "test_tasks": test_tasks, "train_tasks": train_tasks,
                            "n_test_tasks": len(test_tasks),
                            "n_train_tasks": len(train_tasks)}
        all_test += test_tasks
    out['ood_rtllm_tasks'] = sorted(v['task_id'] for v in tasks.values()
                                    if not v['task_id'].startswith(('ve_', 'cc_')))
    assert len(all_test) == 312 == len(set(all_test)), '每任务恰一次 OOF'
    assert not (set(all_test) & set(out['ood_rtllm_tasks']))
    p = f'{OUT}/fold_assignments.json'
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p)
    for k in range(5):
        f = out['folds'][f'fold{k}']
        print(f"fold{k}: test_groups={len(f['test_groups'])} "
              f"test_tasks={f['n_test_tasks']} train_tasks={f['n_train_tasks']}")

def fit_lr(Xtr, ytr):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    if len(np.unique(ytr)) < 2:
        return None  # 训练集单类(弱模型全败折): 无法训练, 调用方降级为 NaN→single-shot
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1, solver='liblinear', max_iter=2000,
                             random_state=42).fit(sc.transform(Xtr), ytr)
    return sc, clf

def stage_score(models):
    import joblib
    from sklearn.metrics import roc_auc_score
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rows = canonical_rows()
    cfg = json.load(open(f'{OUT}/config/track_a_config.json'))
    os.makedirs(f'{OUT}/predictions', exist_ok=True)
    os.makedirs(f'{OUT}/models', exist_ok=True)
    for m in models:
        t0 = time.time()
        X, meta, nmiss = load_model_matrix(m, rows)
        assert X.shape[0] == 11584 and nmiss == 0, (m, X.shape, nmiss)
        group = meta['group']; passed = meta['passed'].astype(int)
        rec = meta['rec'].astype(int)
        P1 = np.full((5, 11584), np.nan, dtype=np.float32)
        P2 = np.full((5, 11584), np.nan, dtype=np.float32)
        info = {'model': m, 'folds': {}}
        for k in range(5):
            tr_groups = set(folds['folds'][f'fold{k}']['train_groups'])
            trm = np.array([g in tr_groups for g in group])
            assert trm.sum() > 0 and (~trm).sum() > 0
            fq = trm & (passed == 0)
            # Q1: 折内训练设计全部样本
            r1 = fit_lr(X[trm], passed[trm])
            if r1 is None:
                print(f'{m} fold{k}: UNTRAINABLE (train labels single-class; '
                      f'fold degrades to single-shot)', flush=True)
                info['folds'][f'fold{k}'] = {
                    'untrainable': True,
                    'n_train_rows_q1': int(trm.sum()), 'n_train_rows_q2': int(fq.sum()),
                    'train_task_pass_rate': round(float(passed[trm].mean()), 4)}
                continue
            sc, clf = r1
            P1[k] = clf.predict_proba(sc.transform(X))[:, 1]
            auc1 = roc_auc_score(passed[trm], P1[k][trm])
            joblib.dump({'scaler': sc, 'clf': clf, 'label': 'passed'},
                        f'{OUT}/models/{m}_fold{k}_q1.joblib')
            # Q2: 折内训练设计的失败样本
            fq = trm & (passed == 0)
            r2 = fit_lr(X[fq], rec[fq])
            if r2 is None:
                print(f'{m} fold{k}: Q2 untrainable; P2 stays NaN', flush=True)
                info['folds'][f'fold{k}'] = {
                    'n_train_rows_q1': int(trm.sum()), 'n_train_rows_q2': int(fq.sum()),
                    'train_auroc_q1': round(float(auc1), 4), 'q2_untrainable': True,
                    'train_task_pass_rate': round(float(passed[trm].mean()), 4)}
                continue
            sc2, clf2 = r2
            P2[k] = clf2.predict_proba(sc2.transform(X))[:, 1]
            auc2 = roc_auc_score(rec[fq], P2[k][fq])
            joblib.dump({'scaler': sc2, 'clf': clf2, 'label': 'recoverable'},
                        f'{OUT}/models/{m}_fold{k}_q2.joblib')
            info['folds'][f'fold{k}'] = {
                'n_train_rows_q1': int(trm.sum()), 'n_train_rows_q2': int(fq.sum()),
                'train_auroc_q1': round(float(auc1), 4),
                'train_auroc_q2': round(float(auc2), 4),
                'train_task_pass_rate': round(float(passed[trm].mean()), 4)}
            print(f'{m} fold{k}: q1_tr_auc={auc1:.4f} q2_tr_auc={auc2:.4f} '
                  f'(rows {int(trm.sum())}/{int(fq.sum())})', flush=True)
        np.savez_compressed(f'{OUT}/predictions/{m}.npz', p1=P1, p2=P2)
        info['config'] = cfg['probe']
        json.dump(info, open(f'{OUT}/models/{m}_train_info.json', 'w'), indent=1)
        print(f'{m} done in {time.time()-t0:.0f}s -> predictions/{m}.npz', flush=True)

if __name__ == '__main__':
    if sys.argv[1] == 'folds':
        stage_folds()
    elif sys.argv[1] == 'score':
        ms = sys.argv[2].split(',') if len(sys.argv) > 2 else MODELS
        stage_score(ms)
