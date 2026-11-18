#!/usr/bin/env python3
"""Q2-v2-Lite v4:候选数与 token 双口径的预算感知对照(二轮审计修复 + P1)。

审计修复(P0):
  - token_pareto 数据源分流(共享方法=trackA;新臂=q2v2), 本文件不再写重复行。
  - decisions_q2v2.csv 的 gen_tokens = 采用前缀 len 实测求和(此前为 0)。

P1(审计任务 2/3):
  - token 目标 = 训练 B=4 池上 rec 级联(Q2v1 门,内层 cross-fitted 分数)的
    实测平均输出 token;不再用候选数。
  - 三条 token-budget 匹配臂(τ/τd/τL 仅训练 B=4 池扫描后冻结):
      q1only_tok@4      门控 P1<τ, 提交 argmax(P1)
      prompt_only_tok@4 门控 P1<0.5 AND d(t)>τd(d(t)=TF-IDF 难度, 每折拟合)
      q2lite_tok@4      门控 U_i=(q⁺−q⁻)/L̂_i > τL, L̂=Ridge 预测的下一候选
                        输出 token 数(在线不可知真值, 用预测)
  - 原 count 口径臂(q2lite/tau_tuned/prompt_diff/exist/oracle_stop)保留作对照。

输出: predictions/decisions_q2v2.csv, results/q2v2_lite.json(冻结参数+诊断),
      results/q2v2_lite_agg.json(四 scope 聚合+设计组配对 bootstrap)。
"""
import csv, json, os, sys, collections, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, LEGACY7, OUT, canonical_rows, tasks_dict, load_texts
from track_a_score import fit_lr
from q2v2_states import prefix_state

LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
TAUS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
TAU_Q1_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
TAU_L_GRID = [0.000001, 0.000002, 0.000005, 0.00001, 0.00002, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
TAU_D_FRACS = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
BS_MAIN = 4
FEAT = ['i', 'B', 'Bmi', 'i_over_B', 'p_best', 'p_latest', 'p_second', 'gap12',
        'p_mean', 'p_std', 'p_min', 'p_range', 'n_improve', 'mag_last_improve',
        'len_best', 'len_latest', 'len_mean', 'ent_best', 'ent_latest', 'ent_mean',
        'lp_best', 'lp_latest', 'lp_mean']


def head_proba(r, X):
    if r is None:
        return np.zeros(len(X))
    sc, clf = r
    return clf.predict_proba(sc.transform(X))[:, 1]


def reg_predict(r, X):
    if r is None:
        return np.zeros(len(X))
    sc, mdl = r
    return mdl.predict(sc.transform(X))


def fit_heads(X, delta, exist):
    yg = (delta == 1).astype(int)
    yh = (delta == -1).astype(int)
    return (fit_lr(X, yg) if len(np.unique(yg)) > 1 else None,
            fit_lr(X, yh) if len(np.unique(yh)) > 1 else None,
            fit_lr(X, exist) if len(np.unique(exist)) > 1 else None)


def jaccard_pick(texts, t, gids, B):
    toks = [set(texts[(t, int(g))].split()) for g in gids[:B]]
    best_i, best_s = 0, -1.0
    for i in range(B):
        s = np.mean([len(toks[i] & toks[j]) / max(len(toks[i] | toks[j]), 1)
                     for j in range(B) if j != i])
        if s > best_s:
            best_s, best_i = s, i
    return best_i


def cascade_indices(p, p2, B, mode, dgate=None, tau1=None):
    kept = [0]
    for i in range(1, B):
        last = kept[-1]
        if mode == 'q1only':
            cont = p[last] < tau1
        elif mode == 'q1only_default':
            cont = p[last] < 0.5
        elif mode == 'prompt_only_tok':
            cont = p[last] < 0.5 and dgate
        else:
            g = p2[last]
            cont = p[last] < 0.5 and (np.isnan(g) or g > 0.5)
        if not cont or len(kept) >= B:
            break
        kept.append(i)
    return kept


def main(models):
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rows = canonical_rows()
    tasks = tasks_dict()
    t2g = {t: tasks[t]['design_group'] for t in tasks}
    decisions, per_model_summary = [], {}
    scorer_in_acc = {}
    for m in models:
        t0 = time.time()
        z = np.load(f'{OUT}/predictions/q2v2/{m}_pool.npz')
        pool = {}
        for key in z.files:
            base, arr = key.rsplit('|', 1)
            pool.setdefault(base, {})[arr] = z[key]
        st = list(csv.DictReader(open(f'{OUT}/predictions/q2v2/{m}_states.csv')))
        texts = load_texts(m, rows)
        # RankEF 分数(审计 P1: 自适应停止臂): 测试分 + inner-val 分(τ 选择)
        rankef_test, rankef_val, rankef_in, rankef_in_test = {}, {}, {}, {}
        rf = f'{OUT}/predictions/rankek_scores_fold{0}_init.csv'
        for k2 in range(5):
            fp = f'{OUT}/predictions/rankek_scores_fold{k2}.csv'
            if os.path.exists(fp):
                for r in csv.DictReader(open(fp)):
                    rankef_test[(int(k2), r['task'], int(r['gid']))] = float(r['score'])
                    rankef_in_test[(int(k2), r['task'], int(r['gid']))] = float(r.get('in_tokens', 0))
            fv = f'{OUT}/predictions/rankek_scores_val_fold{k2}.csv'
            if os.path.exists(fv):
                for r in csv.DictReader(open(fv)):
                    rankef_val[(k2, r['task'], int(r['gid']))] = float(r['score'])
                    rankef_in[(k2, r['task'], int(r['gid']))] = float(r.get('in_tokens', 0))

        # ===== 训练折目标: 候选数(旧)与 token(P1 新), 均为训练 B=4 池 rec 级联
        fold_target, fold_target_tokens = {}, {}
        for k in range(5):
            # 审计统一: 仅用有 RankEF val 分数的任务(inner-val 子集, 与 rankef_adaptive_replay 同源)
            tp_all = [v for key, v in pool.items() if key.startswith(f'TRAIN|{k}|')]
            tp = [v for key, v in pool.items()
                  if key.startswith(f'TRAIN|{k}|') and (k, key.split('|')[2], 0) in rankef_val]
            if not tp:
                tp = tp_all  # fallback: 无 val 分数时用全量
            if not tp:
                continue
            us, ts = [], []
            for d in tp:
                n = len(d['y'])
                if n < 1:
                    continue
                kept = [0]
                for i in range(1, min(4, n)):
                    last = kept[-1]
                    g = d['P2'][last]
                    cont = d['P1'][last] < 0.5 and (np.isnan(g) or g > 0.5)
                    if not cont or len(kept) >= 4:
                        break
                    kept.append(i)
                us.append(len(kept))
                ts.append(float(np.sum(d['len'][kept])))
            fold_target[k] = float(np.mean(us))
            fold_target_tokens[k] = float(np.mean(ts))

        # ===== 每外折拟合三头 + 长度回归头; 训练态预测缓存
        heads, lam_star, tau_star, tauq1_star, tau_d, tau_d_orig = {}, {}, {}, {}, {}, {}
        dtask, tau_q1_tok, tau_d_tok, tau_L_tok = {}, {}, {}, {}
        tok_diag, q1_diag_store, diag = {}, {}, []
        lhat_fit, mgf_fit, mhf_fit = {}, {}, {}
        t2k = {}
        for k in range(5):
            for t in folds['folds'][f'fold{k}']['test_tasks']:
                t2k[t] = k
        ve_all = set(t2k)
        ve_rec = {}
        for key, d in pool.items():
            if key.startswith(('IID|', 'TRAIN|')):
                tt = key.split('|')[2]
                if tt not in ve_rec:
                    ve_rec[tt] = int(d['y'].any())
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import roc_auc_score

        for k in range(5):
            fit = [r for r in st if int(float(r['fold'])) == k]
            if len(fit) < 50:
                continue
            Xf = np.array([[float(r[c]) for c in FEAT] for r in fit])
            rg, rh, re = fit_heads(Xf, np.array([int(r['delta']) for r in fit]),
                                   np.array([int(r['exist']) for r in fit]))
            # 长度回归头: 目标 = 下一候选(索引 i)的输出 len, 来自训练池
            lnext = np.full(len(fit), np.nan)
            for ix, r in enumerate(fit):
                dlen = pool.get(f"TRAIN|{int(float(r['fold']))}|{r['task']}")
                ii = int(float(r['i']))
                if dlen is not None and ii < len(dlen['len']):
                    lnext[ix] = dlen['len'][ii]
            okm = ~np.isnan(lnext)
            rl = None
            if okm.sum() > 50:
                sc_l = StandardScaler().fit(Xf[okm])
                rl = (sc_l, Ridge(alpha=1.0).fit(sc_l.transform(Xf[okm]), lnext[okm]))
            heads[k] = (rg, rh, re, rl)
            mg = head_proba(rg, Xf); mh = head_proba(rh, Xf); me = head_proba(re, Xf)
            lh = (reg_predict(rl, Xf) if rl else np.zeros(len(Xf)))
            lhat_fit[k], mgf_fit[k], mhf_fit[k] = lh, mg, mh
            by_task = collections.defaultdict(list)
            for idx, r in enumerate(fit):
                by_task[(int(float(r['B'])), r['task'])].append(idx)
            by_task_b4 = {key: by_task[key] for key in by_task if key[0] == BS_MAIN}

            def replay_used(pred, thr, src=None):
                us = []
                for (B, t), idxs in by_task_b4.items():
                    idxs = sorted(idxs, key=lambda ix: int(float(fit[ix]['i'])))
                    used = 1
                    for ix in idxs:
                        if pred[ix] > thr and used < B:
                            used += 1
                        else:
                            break
                    us.append(used)
                return float(np.mean(us))

            tgt_k = fold_target.get(k, 1.62)
            lam_star[k] = min(LAMBDAS, key=lambda l: abs(replay_used(mg - mh, l) - tgt_k))
            tau_star[k] = min(TAUS, key=lambda v: abs(replay_used(me, v) - tgt_k))
            # Q1-only 阈值调参(训练 B=4; 命名 tau_tuned; 网格含 0)
            q1_pred = {}
            for ix, r in enumerate(fit):
                q1_pred[ix] = float(r['p_best'])
            tauq1_star[k], best_err, achieved_train = None, None, {}
            for tq in TAU_Q1_GRID:
                us = []
                for (B, t), idxs in by_task_b4.items():
                    idxs = sorted(idxs, key=lambda ix: int(float(fit[ix]['i'])))
                    used = 1
                    for ix in idxs:
                        if q1_pred[ix] < tq and used < B:
                            used += 1
                        else:
                            break
                    us.append(used)
                mu = float(np.mean(us))
                achieved_train[tq] = round(mu, 3)
                err = abs(mu - tgt_k)
                if best_err is None or err < best_err:
                    best_err, tauq1_star[k] = err, tq
            q1_diag = {'target_used_train': round(tgt_k, 3),
                       'achieved_used_train_tau': achieved_train,
                       'achieved_used_train_selected': achieved_train.get(tauq1_star[k]),
                       'note': '阈值调参非成本匹配; 见 token 臂'}
            q1_diag_store[k] = q1_diag
            # 头质量诊断
            for t in folds['folds'][f'fold{k}']['test_tasks']:
                base = f'IID|{k}|{t}'
                d = pool.get(base)
                if d is None or len(d['y']) < BS_MAIN:
                    continue
                Xs = np.array([prefix_state(d['P1'], d['len'], d['ent'], d['lp'], BS_MAIN, i)
                               for i in range(1, BS_MAIN)])
                yy = d['y'][:BS_MAIN]
                dt = np.array([int(yy[int(np.argmax(d['P1'][:i+1]))]) - int(yy[int(np.argmax(d['P1'][:i]))])
                               for i in range(1, BS_MAIN)])
                try:
                    if (dt == 1).any() and not (dt == 1).all():
                        diag.append(dict(fold=k, task=t,
                                         auroc_gain=round(float(roc_auc_score(dt == 1, head_proba(rg, Xs))), 4),
                                         auroc_harm=round(float(roc_auc_score(dt == -1, head_proba(rh, Xs))), 4)))
                except Exception:
                    pass

        # ===== prompt 难度分数(每外折) + τd / 原门控 τd_orig / token 臂 τ 冻结
        for k in range(5):
            if k not in heads:
                continue
            tr = [t for t in ve_all if t2k.get(t) != k]
            pr = [tasks[t].get('prompt', '') for t in tr]
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            Xtr = vec.fit_transform(pr)
            ytr = [ve_rec[t] for t in tr]
            clf_d = LogisticRegression(C=1, solver='liblinear', max_iter=2000,
                                       random_state=42).fit(Xtr, ytr)
            need = list(ve_all | set(folds['ood_rtllm_tasks']))
            dp = clf_d.predict_proba(vec.transform([tasks[t].get('prompt', '') for t in need]))[:, 1]
            dtask[k] = dict(zip(need, dp))
            tr_pools = {key: v for key, v in pool.items() if key.startswith(f'TRAIN|{k}|')}
            tgt_tok = fold_target_tokens.get(k)

            def tok_used(gate_fn, pools=tr_pools):
                ts = []
                for key, d in pools.items():
                    t = key.split('|')[2]
                    n = min(BS_MAIN, len(d['y']))
                    kept = [0]
                    for i in range(1, n):
                        last = kept[-1]
                        if not gate_fn(d, i, last, t) or len(kept) >= BS_MAIN:
                            break
                        kept.append(i)
                    ts.append(float(np.sum(d['len'][kept])))
                return float(np.mean(ts))

            # 原论文 prompt-only 门控(count 口径, 保留对照): τd_orig 匹配候选数
            best = None
            for frac in TAU_D_FRACS:
                td = float(np.quantile([dtask[k][key.split('|')[2]] for key in tr_pools],
                                       1.0 - frac))
                mu = tok_used(lambda d, i, last, t, td=td:
                              d['P1'][last] < 0.5 and dtask[k][t] > td)
                us_cnt = []
                for key, d in tr_pools.items():
                    t = key.split('|')[2]
                    gate_t = dtask[k][t] > td
                    kept = [0]
                    for i in range(1, min(BS_MAIN, len(d['y']))):
                        last = kept[-1]
                        if not (d['P1'][last] < 0.5 and gate_t) or len(kept) >= BS_MAIN:
                            break
                        kept.append(i)
                    us_cnt.append(len(kept))
                err = abs(float(np.mean(us_cnt)) - fold_target.get(k, 1.62))
                if best is None or err < best[0]:
                    best = (err, td)
            tau_d_orig[k] = best[1]
            tau_d[k] = float(np.quantile([dtask[k][key.split('|')[2]] for key in tr_pools],
                                         0.8))
            # P1 token-budget 三臂(τ 扫描匹配 token 目标)
            best = None
            for tq in TAU_Q1_GRID:
                mu = tok_used(lambda d, i, last, t, tq=tq: d['P1'][last] < tq)
                if best is None or abs(mu - tgt_tok) < best[0]:
                    best = (abs(mu - tgt_tok), tq, mu)
            tau_q1_tok[k], a_q1 = best[1], best[2]
            best = None
            for frac in TAU_D_FRACS:
                td = float(np.quantile([dtask[k][key.split('|')[2]] for key in tr_pools],
                                       1.0 - frac))
                mu = tok_used(lambda d, i, last, t, td=td:
                              d['P1'][last] < 0.5 and dtask[k][t] > td)
                if best is None or abs(mu - tgt_tok) < best[0]:
                    best = (abs(mu - tgt_tok), td, mu)
            tau_d_tok[k], a_pd = best[1], best[2]
            mgf, mhf, lhf = lhat_fit.get(k), mgf_fit.get(k), {}
            lhf = lhat_fit.get(k)
            mgf = mgf_fit.get(k); mhf = mhf_fit.get(k)
            fit = [r for r in st if int(float(r['fold'])) == k]   # 按折重建, 防外层残留
            fitb4 = [(ix, r) for ix, r in enumerate(fit) if int(float(r['B'])) == BS_MAIN]
            byt4 = collections.defaultdict(list)
            for ix, r in fitb4:
                byt4[r['task']].append(ix)

            def tok_used_q2(tl):
                ts = []
                for t, idxs in byt4.items():
                    idxs = sorted(idxs, key=lambda ix: int(float(st[ix]['i'])))
                    used = 1
                    for ix in idxs:
                        U = (mgf[ix] - mhf[ix]) / max(lhf[ix], 1.0)
                        if U > tl and used < BS_MAIN:
                            used += 1
                        else:
                            break
                    d = tr_pools[f'TRAIN|{k}|{t}']
                    ts.append(float(np.sum(d['len'][:used])))
                return float(np.mean(ts))
            best = None
            for tl in TAU_L_GRID:
                mu = tok_used_q2(tl)
                if best is None or abs(mu - tgt_tok) < best[0]:
                    best = (abs(mu - tgt_tok), tl, mu)
            tau_L_tok[k], a_q2 = best[1], best[2]
            tok_diag[k] = {'target_tokens_train': round(tgt_tok, 1),
                           'achieved_train': {'q1only_tok': round(a_q1, 1),
                                              'prompt_only_tok': round(a_pd, 1),
                                              'q2lite_tok': round(a_q2, 1)},
                           'tau': {'q1': tau_q1_tok[k], 'd_frac_matched': round(tau_d_tok[k], 4),
                                   'L': tau_L_tok[k]}}
        # P1: RankEF 自适应停止 τ_R(inner-val 上匹配 token 目标, 审计规定)
        tau_R, tok_R_diag = {}, {}
        for k in range(5):
            if k not in heads or fold_target_tokens.get(k) is None:
                continue
            # 审计修复: τ 扫描仅用 inner-val 任务(有 RankEF 分数), 池与目标同源
            val_task_set = {t for (kk, t, g) in rankef_val if kk == k}
            val_pairs = {key: v for key, v in pool.items()
                         if key.startswith(f'TRAIN|{k}|') and key.split('|')[2] in val_task_set}
            val_sc = sorted(rankef_val.get((k, t, g), 0.0)
                            for (kk, t, g) in rankef_val if kk == k
                            for g in range(BS_MAIN))
            if len(val_sc) < 100:
                continue
            def tok_val(tau):
                ts = []
                for key, d in val_pairs.items():
                    t = key.split('|')[2]
                    n = min(BS_MAIN, len(d['y']))
                    kept = [0]
                    for i in range(1, n):
                        last = kept[-1]
                        sv = rankef_val.get((k, t, last), 0.0)
                        if sv < tau and len(kept) < BS_MAIN:
                            kept.append(i)
                        else:
                            break
                    ts.append(float(np.sum(d['len'][kept])))
                return float(np.mean(ts))
            cands = []
            for frac in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
                tau = float(np.quantile(val_sc, frac))
                cands.append((abs(tok_val(tau) - fold_target_tokens[k]), tau))
            tau_R[k], tok_R_diag[k] = min(cands)[1], {
                'target': round(fold_target_tokens[k], 1),
                'achieved_val': round(min(cands)[2] if False else tok_val(min(cands)[1]), 1)}
        per_model_summary[m] = {
            'lambda_star': {f'fold{k}': lam_star.get(k) for k in range(5)},
            'tau_R': {f'fold{k}': tau_R.get(k) for k in range(5)},
            'tau_R_diag': {f'fold{k}': tok_R_diag.get(k) for k in range(5)},
            'scorer_in': {'rankef_tok_mean_per_task': None},
            'tau_star': {f'fold{k}': tau_star.get(k) for k in range(5)},
            'tauq1_star': {f'fold{k}': tauq1_star.get(k) for k in range(5)},
            'tau_d_orig': {f'fold{k}': tau_d_orig.get(k) for k in range(5)},
            'token_budget': {f'fold{k}': tok_diag.get(k) for k in range(5)},
            'target_tokens_train': {f'fold{k}': fold_target_tokens.get(k) for k in range(5)},
            'q1_tau_diag': {f'fold{k}': q1_diag_store.get(k) for k in range(5)},
            'diag': diag}

        # ===== 逐任务回放(IID 测试 + OOD 每折模型)
        def replay(base, k):
            d = pool.get(base)
            if d is None or len(d['y']) < BS_MAIN:
                return None
            p, p2, y = d['P1'][:BS_MAIN], d['P2'][:BS_MAIN], d['y'][:BS_MAIN]
            lens, ents, lps = d['len'], d['ent'], d['lp']
            t = base.split('|')[2]
            out = {}
            out['alone@1'] = (int(y[0]), 1, int(lens[0]))
            bi = jaccard_pick(texts, t, d.get('gids', list(range(BS_MAIN))), BS_MAIN)
            out['sc@4'] = (int(y[bi]), BS_MAIN, int(lens[:BS_MAIN].sum()))
            k1 = cascade_indices(p, p2, BS_MAIN, 'q1only_default')
            out['q1only@4'] = (int(y[int(np.argmax(p[k1]))]), len(k1), int(sum(lens[i] for i in k1)))
            kr = cascade_indices(p, p2, BS_MAIN, 'recprobe')
            out['recprobe@4'] = (int(y[int(np.argmax(p[kr]))]), len(kr), int(sum(lens[i] for i in kr)))
            rg, rh, re, rl = heads[k]
            Xs = np.array([prefix_state(p, lens, ents, lps, BS_MAIN, i)
                           for i in range(1, BS_MAIN)])
            mg = head_proba(rg, Xs); mh = head_proba(rh, Xs); me = head_proba(re, Xs)
            lh = (reg_predict(rl, Xs) if rl else np.ones(len(Xs)))
            # count 口径对照臂(保留)
            kept = [0]
            for i in range(1, BS_MAIN):
                if p[kept[-1]] < tauq1_star[k] and len(kept) < BS_MAIN:
                    kept.append(i)
                else:
                    break
            out['q1only_tau_tuned@4'] = (int(y[int(np.argmax(p[kept]))]), len(kept),
                                         int(sum(lens[i] for i in kept)))
            dt = dtask[k].get(t, 0.0)
            nu = BS_MAIN if dt > tau_d[k] else 1
            out['prompt_diff@4'] = (int(y[int(np.argmax(p[:nu]))]), nu, int(lens[:nu].sum()))
            d_gate_orig = dt > tau_d_orig[k]
            ko = cascade_indices(p, p2, BS_MAIN, 'prompt_only_tok', dgate=d_gate_orig)
            out['prompt_only_orig@4'] = (int(y[int(np.argmax(p[ko]))]), len(ko),
                                         int(sum(lens[i] for i in ko)))
            # P1: RankEF + 自适应停止(τ_R 于 inner-val 冻结; 适配声明见 SUMMARY)
            rt = [rankef_test.get((k, t, g)) for g in range(BS_MAIN)]
            if all(v is not None for v in rt):
                used_r = 1
                for i in range(1, BS_MAIN):
                    if rt[used_r - 1] < tau_R[k] and used_r < BS_MAIN:
                        used_r += 1
                    else:
                        break
                a_r = int(np.argmax(np.array(rt[:used_r])))
                s_in = float(sum(rankef_in_test.get((k, t, g), 0.0)
                                 for g in range(used_r)))
                out['rankef_tok@4'] = (int(y[a_r]), used_r,
                                       int(sum(lens[:used_r])), s_in)
            for name, gate in [('exist@4', me > tau_star[k]),
                               ('q2lite@4', (mg - mh) > lam_star[k])]:
                used = 1
                for idx in range(len(Xs)):
                    if bool(gate[idx]) and used < BS_MAIN:
                        used += 1
                    else:
                        break
                out[name] = (int(y[int(np.argmax(p[:used]))]), used,
                             int(sum(lens[:used])))
            # P1 token-budget 三臂
            d_gate_tok = dt > tau_d_tok[k]
            kt = cascade_indices(p, p2, BS_MAIN, 'q1only', tau1=tau_q1_tok[k])
            out['q1only_tok@4'] = (int(y[int(np.argmax(p[kt]))]), len(kt),
                                   int(sum(lens[i] for i in kt)))
            kp = cascade_indices(p, p2, BS_MAIN, 'prompt_only_tok', dgate=d_gate_tok)
            out['prompt_only_tok@4'] = (int(y[int(np.argmax(p[kp]))]), len(kp),
                                        int(sum(lens[i] for i in kp)))
            used = 1
            for idx in range(len(Xs)):
                U = (mg[idx] - mh[idx]) / max(lh[idx], 1.0)
                if U > tau_L_tok[k] and used < BS_MAIN:
                    used += 1
                else:
                    break
            out['q2lite_tok@4'] = (int(y[int(np.argmax(p[:used]))]), used,
                                   int(sum(lens[:used])))
            # oracle 臂
            used = 1
            for i in range(1, BS_MAIN):
                if int(y[int(np.argmax(p[:i]))]) == 1:
                    break
                used = i + 1
            out['oracle_stop@4'] = (int(y[int(np.argmax(p[:used]))]), used,
                                    int(sum(lens[:used])))
            out['oracle@4'] = (int(y[:BS_MAIN].any()), BS_MAIN, int(lens[:BS_MAIN].sum()))
            return out

        for k in range(5):
            if k not in heads:
                continue
            for t in folds['folds'][f'fold{k}']['test_tasks']:
                r = replay(f'IID|{k}|{t}', k)
                if r:
                    for name, r4 in r.items():
                        sp, u, tok = r4[0], r4[1], r4[2]
                        s_in = r4[3] if len(r4) > 3 else 0.0
                        prev = scorer_in_acc.get((m, name), (0.0, 0))
                        scorer_in_acc[(m, name)] = (prev[0] + s_in, prev[1] + 1)
                        decisions.append(dict(model=m, domain='IID', fold=f'fold{k}',
                                              task=t, group=t2g[t],
                                              method=name.split('@')[0], B='4',
                                              submitted_pass=sp, used=u, gen_tokens=tok,
                                              scorer_in_tokens=s_in))
            for t in folds['ood_rtllm_tasks']:
                r = replay(f'OOD|{k}|{t}', k)
                if r:
                    for name, r4 in r.items():
                        sp, u, tok = r4[0], r4[1], r4[2]
                        s_in = r4[3] if len(r4) > 3 else 0.0
                        decisions.append(dict(model=m, domain='OOD', fold=f'fold{k}',
                                              task=t, group=t2g[t],
                                              method=name.split('@')[0], B='4',
                                              submitted_pass=sp, used=u, gen_tokens=tok,
                                              scorer_in_tokens=s_in))
        per_model_summary[m]['n_decisions'] = sum(1 for d in decisions if d['model'] == m)
        print(f'{m}: done ({time.time()-t0:.0f}s) '
              f'tok_target={ {k: round(v,1) for k,v in fold_target_tokens.items()} }',
              flush=True)

    with open(f'{OUT}/predictions/decisions_q2v2.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['model', 'domain', 'fold', 'task', 'group',
                                           'method', 'B', 'submitted_pass', 'used',
                                           'gen_tokens', 'scorer_in_tokens'])
        w.writeheader(); w.writerows(decisions)
    # P1: scorer 输入 token 会计(RankEF 每 forward 的输入 token)
    s_in = collections.defaultdict(lambda: [0.0, 0])
    for d in decisions:
        if d['method'] == 'rankef_tok':
            s_in[d['model']][0] += float(d.get('scorer_in_tokens') or 0)
            s_in[d['model']][1] += 1
    for mm, (tot, cnt) in s_in.items():
        per_model_summary[mm]['scorer_in_mean_per_task'] = round(tot / max(cnt, 1), 1)
    json.dump(per_model_summary, open(f'{OUT}/results/q2v2_lite.json', 'w'), indent=1)

    def boot(domain, scope_models):
        recs = collections.defaultdict(list)
        for d in decisions:
            if d['domain'] == domain and d['model'] in scope_models:
                recs[(d['model'], d['method'])].append(
                    (d['group'], int(d['submitted_pass']), float(d['used']),
                     float(d['gen_tokens'])))
        methods = sorted({mm for (_, mm) in recs})
        mods = list(scope_models)
        groups = sorted({g for (mo, mm) in recs for (g, _, _, _) in recs[(mo, mm)]})
        gidx = {g: i for i, g in enumerate(groups)}
        S = {mm: {mo: np.zeros(len(groups)) for mo in mods} for mm in methods}
        C = {mm: {mo: np.zeros(len(groups)) for mo in mods} for mm in methods}
        U = {mm: {mo: np.zeros(len(groups)) for mo in mods} for mm in methods}
        T = {mm: {mo: np.zeros(len(groups)) for mo in mods} for mm in methods}
        for (mo, mm), lst in recs.items():
            for g, sp, u, tk in lst:
                gi = gidx[g]
                C[mm][mo][gi] += 1; S[mm][mo][gi] += sp
                U[mm][mo][gi] += u; T[mm][mo][gi] += tk
        rng = np.random.RandomState(42)
        idxs = rng.randint(0, len(groups), (10000, len(groups)))
        lv, uv, tv = {}, {}, {}
        for mm in methods:
            lv[mm] = np.mean([S[mm][mo][idxs].sum(1)/np.maximum(C[mm][mo][idxs].sum(1), 1)
                              for mo in mods], axis=0)
            uv[mm] = float(np.mean([np.sum(U[mm][mo])/np.sum(C[mm][mo]) for mo in mods]))
            tv[mm] = float(np.mean([np.sum(T[mm][mo])/np.sum(C[mm][mo]) for mo in mods]))
        res = {}
        for mm in methods:
            point = float(np.mean([S[mm][mo].sum()/max(C[mm][mo].sum(), 1) for mo in mods]))
            res[mm] = {'pass': round(point, 4), 'used': round(uv[mm], 3),
                       'mean_tokens': round(tv[mm], 1)}
            for ref in ['q1only', 'recprobe', 'q2lite', 'q1only_tok', 'sc']:
                if mm == ref:
                    continue
                if ref not in lv:
                    continue
                dser = lv[mm] - lv[ref]
                res[mm][f'delta_vs_{ref}_pt'] = {
                    'mean': round(100*float(dser.mean()), 2),
                    'lo': round(100*float(np.percentile(dser, 2.5)), 2),
                    'hi': round(100*float(np.percentile(dser, 97.5)), 2),
                    'p_le0': round(float((dser <= 0).mean()), 4)}
            for ref in ['sc', 'q1only_tok', 'recprobe']:
                if mm == ref:
                    continue
                if ref not in uv:
                    continue
                dtok = np.mean([T[mm][mo][idxs].sum(1)/np.maximum(C[mm][mo][idxs].sum(1), 1)
                                - T[ref][mo][idxs].sum(1)/np.maximum(C[ref][mo][idxs].sum(1), 1)
                                for mo in mods], axis=0)
                res[mm][f'tokens_minus_{ref}'] = {
                    'mean': round(float(dtok.mean()), 1),
                    'lo': round(float(np.percentile(dtok, 2.5)), 1),
                    'hi': round(float(np.percentile(dtok, 97.5)), 1)}
        return res

    out = {'lambda_grid': LAMBDAS, 'tau_q1_grid': TAU_Q1_GRID, 'tau_L_grid': TAU_L_GRID,
           'per_model': per_model_summary}
    for domain, scope, tag in [('IID', MODELS, 'iid12'), ('OOD', MODELS, 'ood12'),
                               ('IID', LEGACY7, 'iid7'), ('OOD', LEGACY7, 'ood7')]:
        out[tag] = boot(domain, scope)
    json.dump(out, open(f'{OUT}/results/q2v2_lite_agg.json', 'w'), indent=1)
    for tag in ['iid12', 'ood12']:
        print(f'--- {tag} ---')
        for mm in ['alone', 'sc', 'q1only', 'q1only_tok', 'recprobe', 'prompt_only_orig',
                   'prompt_only_tok', 'q2lite', 'q2lite_tok', 'oracle_stop', 'oracle']:
            v = out[tag].get(mm)
            if v:
                dt = v.get('tokens_minus_recprobe'); dq = v.get('delta_vs_q1only_pt')
                print(f"  {mm:18s} pass={v['pass']:.4f} used={v['used']:.2f} tok={v['mean_tokens']:7.1f} "
                      f"ΔQ1only={dq['mean'] if dq else '—'} tok−Rec={dt['mean'] if dt else '—'}")


if __name__ == '__main__':
    ms = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    main(ms)
