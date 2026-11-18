#!/usr/bin/env python3
"""RankEF Adaptive/Hybrid/Fixed 全臂回放 + 统一指标(审计补齐 + Hybrid 实验)。

四臂(B=4, 同池同序, 零新训练——RankEF 五折 cls 模型不变):
  recprobe         RecProbe 原臂: Q2v1 门控(P1<0.5 AND p2>0.5)+ Q1 选择
  hybrid           §五首选: 同 RecProbe 门控(停止不变), 已生成候选改由 RankEF
                   分数选择(隔离"选择器"贡献; 生成 token 与 recprobe 完全相同)
  rankef_adaptive  RankEF 阈值停止: 分数<τ_R 续采; 提交 kept 中 RankEF 最高
  rankef_fixed     固定 B=4, RankEF 选择
  q1only_tok       Q1+统一阈值停止(τ 与 token 目标同法扫描)
  alone@1 / oracle@4(上界)

τ 冻结(审计问题 1): 每外折在 inner-val 池上扫描, 匹配目标 = inner-val 池
rec 级联实测输出 token(验证态口径); 记录 target/val_achieved/test_achieved/τ。

scorer 成本(审计问题 3): hybrid/rankef_* 每生成一个候选 1 次 CodeT5+ forward;
逐任务累计 scorer_in_tokens(实测)与 scorer_forwards(=n_generated)。

输出:
  predictions/rankef_adaptive_decisions.csv
  results/rankef_adaptive_thresholds.json
  results/rankef_adaptive_metrics.json
"""
import csv, json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, LEGACY7, OUT

BS_MAIN = 4
TAU_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
TAU_Q1_GRID = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]


def rec_cascade(p1, p2, B):
    kept = [0]
    for i in range(1, B):
        last = kept[-1]
        g = p2[last]
        cont = p1[last] < 0.5 and (np.isnan(g) or g > 0.5)
        if not cont or len(kept) >= B:
            break
        kept.append(i)
    return kept


def thr_cascade(score, B, tau):
    """continue while score[last] < tau"""
    kept = [0]
    for i in range(1, B):
        last = kept[-1]
        if score[last] < tau and len(kept) < B:
            kept.append(i)
        else:
            break
    return kept


def main():
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    tasks = {t['task_id']: t for t in json.load(open('./experiments/step2/tasks_step2.json'))}
    t2g = {t: tasks[t]['design_group'] for t in tasks}

    pools = {}
    for m in MODELS:
        z = np.load(f'{OUT}/predictions/q2v2/{m}_pool.npz')
        for key in z.files:
            base, arr = key.rsplit('|', 1)
            if base.startswith(('IID|', 'TRAIN|')):
                pools.setdefault((m, base), {})[arr] = z[key]

    rk_test, rk_val, rk_in_test, rk_in_val = {}, {}, {}, {}
    for k in range(5):
        fp = f'{OUT}/predictions/rankek_scores_fold{k}.csv'
        if os.path.exists(fp):
            for r in csv.DictReader(open(fp)):
                rk_test[(r['model'], k, r['task'], int(r['gid']))] = float(r['score'])
                rk_in_test[(r['model'], k, r['task'], int(r['gid']))] = float(r.get('in_tokens', 0))
        fv = f'{OUT}/predictions/rankek_scores_val_fold{k}.csv'
        if os.path.exists(fv):
            for r in csv.DictReader(open(fv)):
                rk_val[(r['model'], k, r['task'], int(r['gid']))] = float(r['score'])
                rk_in_val[(r['model'], k, r['task'], int(r['gid']))] = float(r.get('in_tokens', 0))

    decisions = []
    thresholds = {}
    for m in MODELS:
        for k in range(5):
            tr = {key.split('|')[2]: d for (mm, key), d in pools.items()
                  if mm == m and key.startswith(f'TRAIN|{k}|')}
            te = {key.split('|')[2]: d for (mm, key), d in pools.items()
                  if mm == m and key.startswith(f'IID|{k}|')}
            if not tr or not te:
                continue

            def val_tokens(gate_fn):
                ts = []
                for t, d in tr.items():
                    n = min(BS_MAIN, len(d['y']))
                    kept = [0]
                    for i in range(1, n):
                        last = kept[-1]
                        if not gate_fn(d, i, last, t) or len(kept) >= BS_MAIN:
                            break
                        kept.append(i)
                    ts.append(float(np.sum(d['len'][kept])))
                return float(np.mean(ts))

            # 审计修复: τ 扫描仅用 inner-val 任务(有 RankEF 分数的训练子集)
            val_tasks_set = {t for t in tr if (m, k, t, 0) in rk_val}
            tr = {t: d for t, d in tr.items() if t in val_tasks_set}
            tgt_ts = []
            for d in tr.values():
                n = min(BS_MAIN, len(d['y']))
                kept = rec_cascade(d['P1'][:n], d['P2'][:n], n)
                tgt_ts.append(float(np.sum(d['len'][kept])))
            target = float(np.mean(tgt_ts))

            val_sc = sorted(rk_val.get((m, k, t, i), 0.0)
                            for t, d in tr.items()
                            for i in range(min(BS_MAIN, len(d['y']))))
            best = None
            for frac in TAU_FRACS:
                tau = float(np.quantile(np.array(val_sc), frac))
                mu = val_tokens(lambda d, i, last, t, tau=tau:
                                rk_val.get((m, k, t, last), 0.0) < tau)
                err = abs(mu - target)
                if best is None or err < best[0]:
                    best = (err, tau, mu)
            tau_R, val_ach_R = best[1], best[2]

            best_q = None
            for tq in TAU_Q1_GRID:
                mu = val_tokens(lambda d, i, last, t, tq=tq: d['P1'][last] < tq)
                err = abs(mu - target)
                if best_q is None or err < best_q[0]:
                    best_q = (err, tq, mu)
            tau_q1, val_ach_q = best_q[1], best_q[2]

            if m == 'CodeV-QW-7B' and k == 0:
                print(f'DEBUG QW fold0: n_val_tasks={len(tr)} target={target:.1f} '
                      f'tau_R={tau_R:.4f} val_ach_R={val_ach_R:.1f} '
                      f'tau_q1={tau_q1} val_ach_q={val_ach_q:.1f}', flush=True)
            thresholds[f'{m}#fold{k}'] = {
                'target_tokens_val': round(target, 1),
                'tau_R': round(tau_R, 4),
                'val_achieved_rankef_tokens': round(val_ach_R, 1),
                'tau_q1_tok': tau_q1,
                'val_achieved_q1only_tokens': round(val_ach_q, 1),
                'n_val_tasks': len(tr)}

            for t, d in te.items():
                n = min(BS_MAIN, len(d['y']))
                p1, p2, y, lens = d['P1'][:n], d['P2'][:n], d['y'][:n], d['len'][:n]
                rk = np.array([rk_test.get((m, k, t, g), 0.0) for g in range(n)])
                rin = np.array([rk_in_test.get((m, k, t, g), 0.0) for g in range(n)])
                n_gen_base = n

                arms = []

                def add(name, kept, sel, s_in, threshold):
                    arms.append(dict(method=name, kept=kept, sel=sel,
                                     s_in=float(s_in), threshold=str(threshold)))

                kept_r = rec_cascade(p1, p2, BS_MAIN)
                a_r = int(np.argmax(p1[kept_r]))
                add('recprobe', kept_r, a_r, 0.0, 'gate=Q2v1;select=Q1')

                kept_h = kept_r
                a_h = kept_h[int(np.argmax(rk[kept_h]))]
                add('hybrid', kept_h, a_h, float(np.sum(rin[kept_h])),
                    'gate=Q2v1;select=RankEF')

                kept_a = thr_cascade(rk, BS_MAIN, tau_R)
                a_a = kept_a[int(np.argmax(rk[kept_a]))]
                add('rankef_adaptive', kept_a, a_a, float(np.sum(rin[kept_a])),
                    threshold=round(tau_R, 4))

                a_f = int(np.argmax(rk))
                add('rankef_fixed', list(range(n)), a_f, float(np.sum(rin)),
                    threshold='fixed_B4')

                kept_q = thr_cascade(p1, BS_MAIN, tau_q1)
                a_q = int(np.argmax(p1[kept_q]))
                add('q1only_tok', kept_q, a_q, 0.0, threshold=round(tau_q1, 4))

                add('alone@1', [0], 0, 0.0, threshold='')
                a_o = int(np.argmax(y))
                add('oracle@4', list(range(n)), a_o, 0.0, threshold='oracle')

                for row in arms:
                    kept, sel = row['kept'], row['sel']
                    if row['method'] == 'oracle@4':
                        sp = int(y[:n].any())
                        sel_gid = int(np.argmax(y[:n]))
                    else:
                        sp = int(y[sel])
                        sel_gid = sel          # 池内位置 = gid(前 4 候选按 gid 升序)
                    is_scorer = row['method'] in ('hybrid', 'rankef_adaptive', 'rankef_fixed')
                    decisions.append(dict(
                        model=m, domain='IID', fold=f'fold{k}', task=t,
                        group=t2g[t], method=row['method'], B=str(len(kept)),
                        n_generated=len(kept), used=len(kept),
                        selected_gid=sel_gid,
                        submitted_pass=sp, gen_tokens=int(np.sum(lens[kept])),
                        scorer_in_tokens=float(row['s_in']) if is_scorer else 0.0,
                        scorer_forwards=len(kept) if is_scorer else 0,
                        threshold=str(row['threshold'])))
        print(m, 'done', flush=True)

    cols = ['model', 'domain', 'fold', 'task', 'group', 'method', 'B',
            'n_generated', 'used', 'selected_gid', 'submitted_pass',
            'gen_tokens', 'scorer_in_tokens', 'scorer_forwards', 'threshold']
    with open(f'{OUT}/predictions/rankef_adaptive_decisions.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(decisions)
    # 预算迁移审计字段(审计问题 1): val 匹配精度 + test 迁移误差, 逐模型-折
    import collections as _c
    _acc = _c.defaultdict(lambda: [0.0, 0])
    for d in decisions:
        if d['method'] == 'rankef_adaptive':
            key = f"{d['model']}#fold{d['fold'][-1]}"
            _acc[key][0] += float(d['gen_tokens'])
            _acc[key][1] += 1
    for key, (tok_sum, cnt) in _acc.items():
        if key in thresholds and cnt:
            tgt = thresholds[key]['target_tokens_val']
            thresholds[key]['test_achieved_rankef_tokens'] = round(tok_sum / cnt, 1)
            thresholds[key]['test_budget_migration_pct'] = round(
                100 * (tok_sum / cnt - tgt) / tgt, 1)
    json.dump(thresholds, open(f'{OUT}/results/rankef_adaptive_thresholds.json', 'w'),
              indent=1)
    print('decisions:', len(decisions), flush=True)

    # ===== 聚合 + 设计组配对 bootstrap =====
    def boot(domain, scope_models, ref='recprobe'):
        recs = collections.defaultdict(list)
        for d in decisions:
            if d['domain'] == domain and d['model'] in scope_models:
                recs[(d['model'], d['method'])].append(
                    (d['group'], int(d['submitted_pass']), float(d['gen_tokens']),
                     float(d['scorer_in_tokens']), int(d['scorer_forwards'])))
        methods = sorted({mm for (_, mm) in recs})
        mods = list(scope_models)
        groups = sorted({g for (mo, mm) in recs for (g, *_rest) in recs[(mo, mm)]})
        gidx = {g: i for i, g in enumerate(groups)}
        M = {mm: {mo: {'S': np.zeros(len(groups)), 'C': np.zeros(len(groups)),
                       'T': np.zeros(len(groups)), 'IN': np.zeros(len(groups)),
                       'F': np.zeros(len(groups))} for mo in mods} for mm in methods}
        for (mo, mm), lst in recs.items():
            for g, sp, tk, s_in, fw in lst:
                gi = gidx[g]
                M[mm][mo]['C'][gi] += 1
                M[mm][mo]['S'][gi] += sp
                M[mm][mo]['T'][gi] += tk
                M[mm][mo]['IN'][gi] += s_in
                M[mm][mo]['F'][gi] += fw
        rng = np.random.RandomState(42)
        idxs = rng.randint(0, len(groups), (10000, len(groups)))

        def series(mm, field):
            return np.mean([M[mm][mo][field][idxs].sum(1) /
                            np.maximum(M[mm][mo]['C'][idxs].sum(1), 1)
                            for mo in mods], axis=0)

        def pt(mm, field):
            return float(np.mean([np.sum(M[mm][mo][field]) /
                                  max(np.sum(M[mm][mo]['C']), 1) for mo in mods]))
        def tok_pcts(mm):
            all_t = np.concatenate([np.repeat(
                np.zeros(0) if False else
                np.array([0.0]), 0) for mo in mods]) if False else None
            # 从 decisions 重新聚合 P50/P95(逐任务 token)
            return None
        res = {}
        per_task_tok = {}
        for d in decisions:
            if d['domain'] == 'IID' and d['model'] in scope_models:
                per_task_tok.setdefault((d['model'], d['method']), []).append(
                    float(d['gen_tokens']))
        for mm in methods:
            res[mm] = {'pass': round(pt(mm, 'S'), 4),
                       'gen_tokens': round(pt(mm, 'T'), 1),
                       'scorer_in_tokens': round(pt(mm, 'IN'), 1),
                       'scorer_forwards': round(pt(mm, 'F'), 3)}
            toks = []
            for (mo, me), lst in per_task_tok.items():
                if me == mm:
                    toks += lst
            if toks:
                res[mm]['p50_tokens'] = round(float(np.percentile(toks, 50)), 1)
                res[mm]['p95_tokens'] = round(float(np.percentile(toks, 95)), 1)
            if mm != ref:
                d_pass = series(mm, 'S') - series(ref, 'S')
                d_tok = series(mm, 'T') - series(ref, 'T')
                res[mm]['delta_pass_vs_%s_pp' % ref] = {
                    'mean': round(100*float(d_pass.mean()), 2),
                    'lo': round(100*float(np.percentile(d_pass, 2.5)), 2),
                    'hi': round(100*float(np.percentile(d_pass, 97.5)), 2)}
                res[mm]['delta_gen_tokens_vs_%s' % ref] = {
                    'mean': round(float(d_tok.mean()), 1),
                    'lo': round(float(np.percentile(d_tok, 2.5)), 1),
                    'hi': round(float(np.percentile(d_tok, 97.5)), 1)}
        return res

    out = {'per_model_thresholds': thresholds}
    for domain, scope, tag in [('IID', MODELS, 'iid12'), ('IID', LEGACY7, 'iid7')]:
        out[tag] = boot(domain, scope, ref='recprobe')
    json.dump(out, open(f'{OUT}/results/rankef_adaptive_metrics.json', 'w'), indent=1)
    for tag in ['iid12', 'iid7']:
        print(f'--- {tag} ---')
        for mm in ['recprobe', 'hybrid', 'q1only_tok', 'rankef_adaptive',
                   'rankef_fixed', 'alone@1', 'oracle@4']:
            v = out[tag].get(mm)
            if v:
                dp = v.get('delta_pass_vs_recprobe_pp', {})
                dt = v.get('delta_gen_tokens_vs_recprobe', {})
                print(f"  {mm:16s} pass={v['pass']:.4f} tok={v['gen_tokens']:7.1f} "
                      f"scorer_in={v['scorer_in_tokens']:7.1f} fwd={v['scorer_forwards']:.2f} "
                      f"Δpass={dp.get('mean','—')} [{dp.get('lo','—')},{dp.get('hi','—')}]")


if __name__ == '__main__':
    main()
