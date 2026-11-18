#!/usr/bin/env python3
"""LAT-L27-adapted(零成本先行, 伦理标注为 adapted baseline)。

与完整 LAT 的差异(必须如实标注):
  - 激励: 使用生成时原始 chat prompt 的最后 token 残差(已保存), 非 LAT 论文的
    neutral-fitting / explicit-correctness 两种新 stimulus(后者需全层新 forward)。
  - 层选择: 仅 layer 27(与 RecProbe Q1 同层同状态), 无 layer-wise 选择。
与 RecProbe Q1 的对照点(忠实保留):
  - 方向提取 = 同任务 pass/fail 候选成对差分 → 中心化 → 第一主成分(PCA),
    符号按训练对校准(正确侧得分更高)——与 Q1 的 logistic 探针同数据同层,
    直接回答"成对差分 PCA 方向 vs LR 探针"的方法问题。

协议: 每外折 k、每生成器: 训练折(≠k)中"既有 pass 又有 fail"的任务, 取最早
pass gid 与最早 fail gid 成对(确定性); 测试折全部任务的前 B=4 候选打分排序。
输出: predictions/lat_l27_scores.csv + results/lat_l27_ranking.json
"""
import csv, json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, LEGACY7, OUT, canonical_rows, load_model_matrix

BS_MAIN = 4


def main(models):
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rows = canonical_rows()
    t2k = {}
    for k in range(5):
        for t in folds['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    decisions, score_rows, fold_info = [], [], {}
    for m in models:
        X, meta, nmiss = load_model_matrix(m, rows)
        assert nmiss == 0
        passed = meta['passed'].astype(int)
        task_rows = collections.defaultdict(list)
        for i, t in enumerate(meta['task']):
            task_rows[t].append(i)
        scores_row = {}
        for k in range(5):
            # ---- 拟合: 训练折成对差分 → PCA 第一主成分
            pairs = []
            for t, idxs in task_rows.items():
                if t2k.get(t) == k:
                    continue
                yy = passed[idxs]
                if yy.sum() == 0 or yy.sum() == len(yy):
                    continue
                a_pass = a_fail = None
                for j, i in enumerate(sorted(idxs, key=lambda i: meta['gid'][i])):
                    if yy[j] and a_pass is None:
                        a_pass = i
                    if not yy[j] and a_fail is None:
                        a_fail = i
                if a_pass is not None and a_fail is not None:
                    pairs.append(X[a_pass] - X[a_fail])
            if len(pairs) < 20:
                fold_info[(m, k)] = {'n_pairs': len(pairs), 'skip': True}
                continue
            D = np.array(pairs, dtype=np.float64)
            Dc = D - D.mean(0, keepdims=True)
            U, S, Vt = np.linalg.svd(Dc, full_matrices=False)
            v = Vt[0]
            if float((D @ v).mean()) < 0:      # 符号: 正确侧投影更高
                v = -v
            fold_info[(m, k)] = {'n_pairs': len(pairs),
                                 'train_pair_margin': round(float((D @ v).mean()), 3)}
            # ---- 测试: 本折测试任务前 B=4 候选打分
            for t in folds['folds'][f'fold{k}']['test_tasks']:
                idxs = sorted(task_rows[t], key=lambda i: meta['gid'][i])[:BS_MAIN]
                s = X[idxs] @ v
                order = np.argsort(-s)
                scores_row[t] = (idxs, s, order)
        # ---- 逐任务产出决策(pass@rank 与 Q1 同池同序)
        for k in range(5):
            if (m, k) not in fold_info or fold_info[(m, k)].get('skip'):
                continue
            for t in folds['folds'][f'fold{k}']['test_tasks']:
                if t not in scores_row:
                    continue
                idxs, s, order = scores_row[t]
                yy = passed[idxs]
                for n in (1, 2, BS_MAIN):
                    top = order[:n]
                    sp = int(yy[top].max())
                    decisions.append(dict(model=m, domain='IID', fold=f'fold{k}',
                                          task=t, group=meta['group'][idxs[0]],
                                          method='lat_l27', B=str(n),
                                          submitted_pass=sp, used=0,
                                          gen_tokens=0, n_pool=len(idxs)))
                score_rows.extend(
                    dict(model=m, fold=f'fold{k}', task=t,
                         gid=int(meta['gid'][i]), score=float(ss), passed=int(pp))
                    for i, ss, pp in zip(idxs, s, yy))
        print(m, 'fold pairs:', {k2: fold_info[(m, k2)].get('n_pairs')
                                 for k2 in range(5) if (m, k2) in fold_info}, flush=True)
    with open(f'{OUT}/predictions/lat_l27_ranking.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['model', 'domain', 'fold', 'task', 'group',
                                           'method', 'B', 'submitted_pass', 'used',
                                           'gen_tokens', 'n_pool'])
        w.writeheader(); w.writerows(decisions)
    with open(f'{OUT}/predictions/lat_l27_scores.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['model', 'fold', 'task', 'gid', 'score',
                                           'passed'])
        w.writeheader(); w.writerows(score_rows)
    json.dump({f'{a}#fold{k}': v for (a, k), v in fold_info.items()},
              open(f'{OUT}/results/lat_l27_fit_info.json', 'w'), indent=1)
    print('decisions:', len(decisions))


if __name__ == '__main__':
    ms = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    main(ms)
