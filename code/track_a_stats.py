#!/usr/bin/env python3
"""Track A 阶段4:聚合、设计级配对 bootstrap、逐折表、fixed-split 对照。

统计单位 = design_group(VE: ve+cc 成对;RTLLM: 单任务)。bootstrap 10k,seed 42。
IID 与 OOD 分开;OOD 主结论 = 5 个折模型的跨折平均(重采样内配对)。
"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from track_a_common import MODELS, OUT

BS = ['2', '4', '8']
METHODS = [('alone', '-')] + [(m, b) for m in ['retry', 'sc', 'recprobe', 'q1only',
                                               'recprobe_s', 'q1only_s', 'judge',
                                               'oracle'] for b in BS]
REP, SEED = 10000, 42


def load_decisions():
    d = list(csv.DictReader(open(f'{OUT}/predictions/decisions_trackA.csv')))
    p_extra = f'{OUT}/predictions/decisions_extra.csv'
    if os.path.exists(p_extra):
        d += list(csv.DictReader(open(p_extra)))
    for r in d:
        r['submitted_pass'] = int(r['submitted_pass']); r['used'] = float(r['used'])
        r['gen_tokens'] = float(r['gen_tokens']); r['B'] = str(r['B'])
    return d


def mk(r):
    return (r['method'], r['B'])


def macro_table(d, domain, fold=None):
    """返回 macro[(method,B)] = (pass, used, tokens),模型平均;及 per_model。"""
    per = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    for r in d:
        if r['domain'] != domain or (fold is not None and r['fold'] != fold):
            continue
        c = per[(r['model'], *mk(r))]
        c[0] += 1; c[1] += r['submitted_pass']; c[2] += r['used']; c[3] += r['gen_tokens']
    per = {k: (v[0], v[1] / v[0], v[2] / v[0], v[3] / v[0]) for k, v in per.items()}
    macro, pm = {}, {}
    for mkey in METHODS:
        vals = [per[(m, *mkey)] for m in MODELS if (m, *mkey) in per]
        if not vals:
            continue
        macro[mkey] = (float(np.mean([v[1] for v in vals])),
                       float(np.mean([v[2] for v in vals])),
                       float(np.mean([v[3] for v in vals])))
        pm[mkey] = {m: (per[(m, *mkey)][1], per[(m, *mkey)][2]) for m in MODELS
                    if (m, *mkey) in per}
    return macro, pm


def design_sums(d, domain, fold=None):
    """T[group][(method,B)][model] = [n, pass_sum]。用于配对 bootstrap。"""
    T = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0.0])))
    for r in d:
        if r['domain'] != domain or (fold is not None and r['fold'] != fold):
            continue
        c = T[r['group']][mk(r)][r['model']]
        c[0] += 1; c[1] += r['submitted_pass']
    return T


def boot(T, groups, rep=REP, seed=SEED, series=False):
    """配对设计级 bootstrap。返回 {method|B: point,lo,hi,delta,delta_lo,delta_hi};
    series=True 时返回 (lv,dl) 原始序列(供跨折配对平均)。
    macro = 7 模型平均;每次重采样同批设计用于所有方法(含 alone,配对)。"""
    groups = sorted(groups)
    gidx = {g: i for i, g in enumerate(groups)}
    mks = sorted({k for g in groups for k in T[g]})
    S = {t: {m: np.zeros(len(groups)) for m in MODELS} for t in mks}
    C = {t: {m: np.zeros(len(groups)) for m in MODELS} for t in mks}
    for g in groups:
        gi = gidx[g]
        for t, per_model in T[g].items():
            for m, (c, s) in per_model.items():
                C[t][m][gi] = c; S[t][m][gi] = s
    rng = np.random.RandomState(seed)
    A = ('alone', '-')
    assert A in mks
    idxs = rng.randint(0, len(groups), (rep, len(groups)))
    lv = {t: np.empty(rep) for t in mks}
    dl = {t: np.empty(rep) for t in mks}
    for r in range(rep):
        idx = idxs[r]
        row = {}
        for t in mks:
            pm = [S[t][m][idx].sum() / C[t][m][idx].sum() if C[t][m][idx].sum() else np.nan
                  for m in MODELS]
            row[t] = float(np.nanmean(pm))
        for t in mks:
            lv[t][r] = row[t]; dl[t][r] = row[t] - row[A]
    if series:
        return lv, dl
    out = {}
    for t in mks:
        out[t] = {'point': round(float(lv[t].mean()), 4),
                  'lo': round(float(np.percentile(lv[t], 2.5)), 4),
                  'hi': round(float(np.percentile(lv[t], 97.5)), 4),
                  'delta': round(float(dl[t].mean()), 4),
                  'delta_lo': round(float(np.percentile(dl[t], 2.5)), 4),
                  'delta_hi': round(float(np.percentile(dl[t], 97.5)), 4)}
    return out


def compute_subset(d, models, tag):
    """对给定模型子集计算 IID/OOD 全套指标(不写文件)。"""
    sub = [r for r in d if r['model'] in models]
    out = {'models': list(models)}
    iid = [r for r in sub if r['domain'] == 'IID']
    ood = [r for r in sub if r['domain'] == 'OOD']
    seen = {(r['model'], r['task']) for r in iid}
    assert len(seen) == len(models) * 312, f'[{tag}] IID 覆盖 {len(seen)} != {len(models)*312}'
    seen_ood = {(r['model'], r['fold'], r['task']) for r in ood}
    assert len(seen_ood) == len(models) * 5 * 50, f'[{tag}] OOD 覆盖 {len(seen_ood)} != {len(models)*5*50}'
    out['integrity'] = {'iid_model_tasks': len(seen), 'ood_model_fold_tasks': len(seen_ood),
                        'iid_rows': len(iid), 'ood_rows': len(ood)}

    macro_iid, _ = macro_table(sub, 'IID')
    T_iid = design_sums(sub, 'IID')
    bt_iid = boot(T_iid, list(T_iid))
    out['iid'] = {
        'macro': {f'{a}|{b}': {'pass': round(v[0], 4), 'used': round(v[1], 3),
                               'tokens': round(v[2], 1)} for (a, b), v in macro_iid.items()},
        'bootstrap': {f'{a}|{b}': v for (a, b), v in bt_iid.items()}}
    out['iid_per_fold'] = {}
    for k in range(5):
        mf, _ = macro_table(sub, 'IID', fold=f'fold{k}')
        out['iid_per_fold'][f'fold{k}'] = {f'{a}|{b}': round(v[0], 4)
                                           for (a, b), v in mf.items()}

    out['ood_per_fold_model'] = {}
    lv_folds = []
    for k in range(5):
        Tk = design_sums(sub, 'OOD', fold=f'fold{k}')
        lv_k, dl_k = boot(Tk, list(Tk), series=True)
        lv_folds.append(lv_k)
        def summ(lv, dl):
            return {t: {'point': round(float(lv[t].mean()), 4),
                        'lo': round(float(np.percentile(lv[t], 2.5)), 4),
                        'hi': round(float(np.percentile(lv[t], 97.5)), 4),
                        'delta': round(float(dl[t].mean()), 4),
                        'delta_lo': round(float(np.percentile(dl[t], 2.5)), 4),
                        'delta_hi': round(float(np.percentile(dl[t], 97.5)), 4)}
                    for t in lv}
        out['ood_per_fold_model'][f'fold{k}'] = {
            f'{t[0]}|{t[1]}': v for t, v in summ(lv_k, dl_k).items()}
    mks = sorted(lv_folds[0].keys())
    cf_lv = {t: np.mean([lf[t] for lf in lv_folds], axis=0) for t in mks}
    A = ('alone', '-')
    out['ood_crossfold'] = {f'{t[0]}|{t[1]}': {
        'point': round(float(cf_lv[t].mean()), 4),
        'lo': round(float(np.percentile(cf_lv[t], 2.5)), 4),
        'hi': round(float(np.percentile(cf_lv[t], 97.5)), 4),
        'delta': round(float((cf_lv[t] - cf_lv[A]).mean()), 4),
        'delta_lo': round(float(np.percentile(cf_lv[t] - cf_lv[A], 2.5)), 4),
        'delta_hi': round(float(np.percentile(cf_lv[t] - cf_lv[A], 97.5)), 4)}
        for t in mks}
    macro_ens, _ = macro_table(sub, 'OOD_ENS')
    out['ood_ens_aux'] = {f'{a}|{b}': {'pass': round(v[0], 4), 'used': round(v[1], 3)}
                          for (a, b), v in macro_ens.items()}
    tables = {'bt_iid': bt_iid, 'macro_iid': macro_iid, 'mks': mks, 'A': A}
    return out, tables


def write_per_model_csv(d, path, domain):
    """逐模型宏平均表(group, model, method, B, n_tasks, pass, mean_used, mean_tokens)。
    IID 用 OOF 行;OOD 先按折内宏平均再对 5 个折模型求均值(与 ood_crossfold 同语义)。"""
    import csv as _csv
    from collections import defaultdict
    new5 = set(MODELS) - set(['CodeV-QW-7B', 'CodeV-DS-6.7B', 'VeriCoder_Qwen14B',
                              'CodeV-R1-Distill-Qwen-7B', 'CodeV-R1-RL-Qwen-7B',
                              'RTLCoder-Deepseek-v1.1', 'DeepSeek-Coder-6.7B'])
    rows_out = []
    mks = sorted({mk(r) for r in d if r['domain'] == domain})
    for m in MODELS:
        sub = [r for r in d if r['model'] == m and r['domain'] == domain]
        if not sub:
            continue
        grp = 'new5' if m in new5 else 'legacy7'
        for key in mks:
            sel = [r for r in sub if mk(r) == key]
            if not sel:
                continue
            if domain == 'IID':
                n_tasks = len({r['task'] for r in sel})
                p = sum(r['submitted_pass'] for r in sel) / n_tasks
                u = sum(r['used'] for r in sel) / n_tasks
                tok = sum(r['gen_tokens'] for r in sel) / n_tasks
                rows_out.append({'group': grp, 'model': m, 'domain': domain,
                                 'method': key[0], 'B': key[1], 'n_tasks': n_tasks,
                                 'pass': round(p, 4), 'mean_used': round(u, 3),
                                 'mean_tokens': round(tok, 1)})
            else:
                folds = sorted({r['fold'] for r in sel})
                ps, us = [], []
                for fd in folds:
                    s2 = [r for r in sel if r['fold'] == fd]
                    n2 = len({r['task'] for r in s2})
                    ps.append(sum(r['submitted_pass'] for r in s2) / n2)
                    us.append(sum(r['used'] for r in s2) / n2)
                rows_out.append({'group': grp, 'model': m, 'domain': domain,
                                 'method': key[0], 'B': key[1], 'n_tasks': 50,
                                 'pass': round(float(np.mean(ps)), 4),
                                 'mean_used': round(float(np.mean(us)), 3),
                                 'mean_tokens': ''})
    with open(path, 'w', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=['group', 'model', 'domain', 'method', 'B',
                                           'n_tasks', 'pass', 'mean_used', 'mean_tokens'])
        w.writeheader()
        w.writerows(rows_out)
    print('wrote', path, len(rows_out), 'rows')


def main():
    d = load_decisions()
    res = {'n_decision_rows': len(d)}
    # 全模型口径(含新增)
    all_res, tables_all = compute_subset(d, MODELS, 'ALL')
    for k in ['models', 'integrity', 'iid', 'iid_per_fold', 'ood_per_fold_model',
              'ood_crossfold', 'ood_ens_aux']:
        res[k] = all_res[k]
    # 原始 7 模型口径(与已交付 v1 结果同子集, 供连续性对照)
    LEGACY7 = ['CodeV-QW-7B', 'CodeV-DS-6.7B', 'VeriCoder_Qwen14B',
               'CodeV-R1-Distill-Qwen-7B', 'CodeV-R1-RL-Qwen-7B',
               'RTLCoder-Deepseek-v1.1', 'DeepSeek-Coder-6.7B']
    legacy7_res = None
    if set(LEGACY7) <= set(MODELS):
        legacy7_res, tables_l7 = compute_subset(d, LEGACY7, 'LEGACY7')
        res['legacy7'] = legacy7_res

    # ---- fixed-split 对照(历史值,非独立复现)
    res['fixed_split_vs_trackA'] = {
        'note': 'fixed-split = sealed iid 旧产物(基线行含 vintage 偏差,handoff README §7.1);'
                'trackA = Group5Fold OOF,独立计算;两者不可当作两次独立复现相加',
        'fixed_split_sealed_iid': {'alone@4': 0.3195, 'recprobe@4': 0.4246, 'used': 1.62},
        'trackA_iid': {'alone': res['iid']['macro']['alone|-'],
                       'recprobe@4': res['iid']['macro']['recprobe|4'],
                       'oracle@4': res['iid']['macro']['oracle|4']}}

    json.dump(res, open(f'{OUT}/results/track_a_results.json', 'w'), indent=1, ensure_ascii=False)
    write_per_model_csv(d, f'{OUT}/results/iid_per_model.csv', 'IID')
    write_per_model_csv(d, f'{OUT}/results/ood_per_model.csv', 'OOD')
    # CSV: 全模型 + legacy7 两套
    for suffix, sub, tables in [('', all_res, tables_all),
                                ('_legacy7', legacy7_res, legacy7_res and tables_l7)]:
        if sub is None:
            continue
        bt, mac = tables['bt_iid'], tables['macro_iid']
        with open(f'{OUT}/results/iid_summary{suffix}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'B', 'macro_pass', 'ci_lo', 'ci_hi', 'delta_vs_alone_pt',
                        'delta_lo', 'delta_hi', 'mean_used', 'mean_tokens'])
            for (a, b) in METHODS:
                if ('alone', '-') == (a, b):
                    continue
                t = (a, b); v = bt[t]; mac_ = mac[t]
                w.writerow([a, b, v['point'], v['lo'], v['hi'], 100 * v['delta'],
                            100 * v['delta_lo'], 100 * v['delta_hi'],
                            round(mac_[1], 3), round(mac_[2], 1)])
        A = tables['A']
        with open(f'{OUT}/results/ood_summary{suffix}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['method', 'B', 'crossfold_macro_pass', 'ci_lo', 'ci_hi',
                        'delta_vs_alone_pt', 'delta_lo', 'delta_hi'])
            for t in tables['mks']:
                if t == A:
                    continue
                v = sub['ood_crossfold'][f'{t[0]}|{t[1]}']
                w.writerow([t[0], t[1], v['point'], v['lo'], v['hi'], 100 * v['delta'],
                            100 * v['delta_lo'], 100 * v['delta_hi']])
    print('IID macro (all):', json.dumps(res['iid']['macro'], indent=1)[:800])
    if 'legacy7' in res:
        print('IID macro (legacy7):', json.dumps(res['legacy7']['iid']['macro'], indent=1)[:400])
    print('OOD crossfold (all):', json.dumps(res['ood_crossfold'], indent=1)[:800])
    print('wrote results/')

if __name__ == '__main__':
    main()
