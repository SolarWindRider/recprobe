#!/usr/bin/env python3
"""冻结轮 A: 统一 AUROC/AUPRC 口径 + 池级逐任务 CSV + 标签正负统计。

输出:
  results/freeze_pool_metrics.csv      — 双域×4 scorer 逐池指标(一切排名数字的复算源)
  results/freeze_label_stats.csv       — 每模型×折 池数/正负样本统计(scorer 无关)
  results/freeze_metrics_unified.json  — 统一口径聚合(micro+macro+逐折)
"""
import json, csv, glob, collections
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']
BS = 4
SC_ORDER = ['cpp', 'slm_judge', 'rankef', 'lat_l27']

def stable_top(y, s):
    ss = np.array(s[:BS]); yy = np.array(y[:BS])
    o = np.argsort(-ss, kind='stable')
    return int(yy[o[0]]), int(yy[o[:2]].max())

def main():
    fa = json.load(open(f'{OUT}/fold_assignments.json'))
    RT = set(fa['ood_rtllm_tasks'])
    t2k = {}
    for k in range(5):
        for t in fa['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k

    # ---------- 标签 ----------
    Y = {'OOD': {}, 'IID': {}}
    for m in MODELS:
        gd = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            t, g = r['task'], int(r['generation_id'])
            if g < BS:
                gd[t][g] = int(bool(r['simulation_pass']))
        for t, d in gd.items():
            if len(d) == BS:
                if t in RT:
                    Y['OOD'][(m, t)] = [d[g] for g in range(BS)]
                if t in t2k:
                    Y['IID'][(m, t2k[t], t)] = [d[g] for g in range(BS)]

    # ---------- 标签统计(逐模型×折) ----------
    lab_rows = []
    for dom, yy in Y.items():
        for m in MODELS:
            for k in range(5):
                pools = [(t, v) for (mm, kk, t), v in (yy.items() if dom == 'IID'
                         else [(kk2, v) for (mm2, kk2, t2), v in [] ])] if False else None
            # 逐折统计
        pass
    # 简化: 直接遍历
    for dom, keys in (('OOD', lambda m, k: [(m, t) for (mm, t) in Y['OOD'] if mm == m]),
                      ('IID', lambda m, k: [(m, k, t) for (mm, kk, t) in Y['IID'] if mm == m and kk == k])):
        for m in MODELS:
            for k in range(5):
                ks = keys(m, k)
                vals = [Y[dom][kk] for kk in ks]
                if not vals:
                    continue
                flat = [x for v in vals for x in v]
                lab_rows.append(dict(
                    domain=dom, fold=f'fold{k}', model=m,
                    n_pools_complete=len(vals),
                    n_pools_both_classes=sum(1 for v in vals if 0 < sum(v) < BS),
                    n_candidates=len(flat), n_pos=sum(flat), n_neg=len(flat)-sum(flat),
                    pos_rate=round(sum(flat)/len(flat), 4)))
    with open(f'{OUT}/results/freeze_label_stats.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['domain','fold','model','n_pools_complete',
                                           'n_pools_both_classes','n_candidates','n_pos','n_neg','pos_rate'])
        w.writeheader(); w.writerows(lab_rows)

    # ---------- 各 scorer 分数 ----------
    S = {dom: collections.defaultdict(dict) for dom in ('OOD', 'IID')}
    for m in MODELS:
        z = np.load(f'{OUT}/predictions/q2v2/{m}_pool.npz')
        for key in z.files:
            b, a = key.rsplit('|', 1)
            if a != 'P1':
                continue
            if b.startswith('OOD|'):
                _, k, t = b.split('|', 2)
                S['OOD']['cpp'][(m, int(k), t)] = z[key][:BS].tolist()
            elif b.startswith('IID|'):
                _, k, t = b.split('|', 2)
                S['IID']['cpp'][(m, int(k), t)] = z[key][:BS].tolist()
    z = np.load(f'{OUT}/results/rtllm_lat_scores.npz')
    for key in z.files:
        m, k, t = key.split('#', 2)
        S['OOD']['lat_l27'][(m, int(k), t)] = z[key].tolist()
    z = np.load(f'{OUT}/results/rtllm_rankek_scores_5fold.npz')
    for key in z.files:
        m, k, t = key.split('#', 2)
        S['OOD']['rankef'][(m, int(k), t)] = z[key].tolist()
    # IID lat
    lat = collections.defaultdict(dict)
    for r in csv.DictReader(open(f'{OUT}/predictions/lat_l27_scores.csv')):
        k = int(r['fold'][-1])
        if r['task'] in t2k and t2k[r['task']] == k:
            lat[(r['model'], k, r['task'])][int(r['gid'])] = float(r['score'])
    for key, d in lat.items():
        if len(d) >= BS:
            S['IID']['lat_l27'][key] = [d[g] for g in range(BS)]
    # IID rankef
    rk = collections.defaultdict(dict)
    for k in range(5):
        fp = f'{OUT}/predictions/rankek_scores_fold{k}.csv'
        for r in csv.DictReader(open(fp)):
            if r['task'] in t2k and t2k[r['task']] == k:
                rk[(r['model'], k, r['task'])][int(r['gid'])] = float(r['score'])
    for key, d in rk.items():
        if len(d) >= BS:
            S['IID']['rankef'][key] = [d[g] for g in range(BS)]
    # SLM(零样本: OOD 复制 5 折; IID 直接)
    sm = collections.defaultdict(dict)
    for fp in glob.glob(f'{OUT}/predictions/slm_judge_scores_*.csv'):
        for r in csv.DictReader(open(fp)):
            sm[(r['model'], r['task'])][int(r['gid'])] = float(r['score'])
    for (m, t), d in sm.items():
        if len(d) < BS:
            continue
        v = [d[g] for g in range(BS)]
        if t in RT:
            for k in range(5):
                S['OOD']['slm_judge'][(m, k, t)] = v
        if t in t2k:
            S['IID']['slm_judge'][(m, t2k[t], t)] = v

    # ---------- 池级 CSV + 统一聚合 ----------
    pool_rows = []
    uni = {}
    for dom in ('OOD', 'IID'):
        for sc in SC_ORDER:
            d = S[dom].get(sc, {})
            au_l, ap_l, p1_l, p2_l = [], [], [], []
            per_fold = collections.defaultdict(lambda: {'au': [], 'ap': [], 'n': 0, 'nb': 0})
            per_gen = collections.defaultdict(lambda: {'au': [], 'ap': [], 'p1': []})
            for key, s in d.items():
                if dom == 'OOD':
                    m, k, t = key; lk = (m, t)
                else:
                    m, k, t = key; lk = key
                if lk not in Y[dom]:
                    continue
                y = Y[dom][lk]
                pr1, pr2 = stable_top(y, s)
                yarr = np.array(y)
                both = 0 < yarr.sum() < BS
                au = ap = ''
                if both:
                    au = round(float(roc_auc_score(yarr, np.array(s[:BS]))), 4)
                    ap = round(float(average_precision_score(yarr, np.array(s[:BS]))), 4)
                    au_l.append(float(au)); ap_l.append(float(ap))
                    per_fold[k]['au'].append(float(au)); per_fold[k]['ap'].append(float(ap))
                    per_fold[k]['nb'] += 1
                    per_gen[m]['au'].append(float(au)); per_gen[m]['ap'].append(float(ap))
                per_fold[k]['n'] += 1
                p1_l.append(pr1); p2_l.append(pr2)
                per_gen[m]['p1'].append(pr1)
                pool_rows.append(dict(
                    domain=dom, scorer=sc, fold=f'fold{k}', model=m, task=t,
                    n_candidates=BS, n_pos=int(yarr.sum()), n_neg=BS-int(yarr.sum()),
                    both_classes=int(both), auroc=au, auprc=ap,
                    pass_rank1=pr1, pass_rank2=pr2,
                    labels=';'.join(map(str, y)),
                    scores=';'.join(f'{x:.6f}' for x in s[:BS])))
            uni[f'{dom}|{sc}'] = {
                'n_pool_obs': len(p1_l),
                'n_both_class_pools': len(au_l),
                'auroc_micro': round(float(np.mean(au_l)), 4) if au_l else None,
                'auprc_micro': round(float(np.mean(ap_l)), 4) if ap_l else None,
                'auroc_macro_per_generator': round(float(np.mean(
                    [np.mean(v['au']) for v in per_gen.values() if v['au']])), 4) if au_l else None,
                'auprc_macro_per_generator': round(float(np.mean(
                    [np.mean(v['ap']) for v in per_gen.values() if v['ap']])), 4) if ap_l else None,
                'pass_rank1_micro_pct': round(float(np.mean(p1_l))*100, 2),
                'pass_rank1_macro_pct': round(float(np.mean(
                    [np.mean(v['p1']) for v in per_gen.values()]))*100, 2),
                'pass_rank2_micro_pct': round(float(np.mean(p2_l))*100, 2),
                'per_fold': {f'fold{k}': {'n_pools': v['n'],
                                          'n_both': v['nb'],
                                          'auroc': round(float(np.mean(v['au'])), 4) if v['au'] else None,
                                          'auprc': round(float(np.mean(v['ap'])), 4) if v['ap'] else None}
                             for k, v in sorted(per_fold.items())}}
    json.dump({'convention': {
        'pool_unit': '(domain, generator, fold, task) x B=4 candidates',
        'auroc_auprc': 'per-pool sklearn roc_auc_score / average_precision_score, '
                       'averaged over pools containing both classes; micro and '
                       'macro(per-generator) both reported',
        'tie_break': 'np.argsort(-score, kind="stable") -> earliest gid',
        'macro_pass': 'per-generator mean then unweighted mean over 12 generators'},
        'metrics': uni},
        open(f'{OUT}/results/freeze_metrics_unified.json', 'w'), indent=1)
    with open(f'{OUT}/results/freeze_pool_metrics.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['domain','scorer','fold','model','task',
                                           'n_candidates','n_pos','n_neg','both_classes',
                                           'auroc','auprc','pass_rank1','pass_rank2',
                                           'labels','scores'])
        w.writeheader(); w.writerows(pool_rows)
    print(f'pool rows: {len(pool_rows)}; label rows: {len(lab_rows)}')
    for key, v in uni.items():
        print(f"{key}: n={v['n_pool_obs']} both={v['n_both_class_pools']} "
              f"AUROC={v['auroc_micro']} AUPRC={v['auprc_micro']} "
              f"p@1micro={v['pass_rank1_micro_pct']} p@1macro={v['pass_rank1_macro_pct']}")

if __name__ == '__main__':
    main()
