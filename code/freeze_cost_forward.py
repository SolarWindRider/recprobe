#!/usr/bin/env python3
"""冻结轮 B: forward 口径修正 + Hybrid/RankEF-Adaptive 逐任务 scorer 成本 CSV。

输出:
  results/freeze_scorer_cost.json        — 修正后的完整成本(候选评分数/forward calls/tokens)
  results/freeze_arm_scorer_cost.csv     — 两臂逐任务成本+决策(可复算宏平均与 CI)
"""
import json, csv, glob, os, collections
import numpy as np

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']
BS = 4

def thr_cascade(score, B, tau):
    kept = [0]
    for i in range(1, B):
        if score[kept[-1]] < tau and len(kept) < B:
            kept.append(i)
        else:
            break
    return kept

def rp_cascade(p1, p2, B):
    kept = [0]
    for i in range(1, B):
        g = p2[kept[-1]]
        if p1[kept[-1]] < 0.5 and (np.isnan(g) or g > 0.5) and len(kept) < B:
            kept.append(i)
        else:
            break
    return kept

def cluster_boot(A, B_, n_boot=10000, seed=42):
    tasks = sorted(set(t for (_, t) in A) & set(t for (_, t) in B_))
    av, bv = collections.defaultdict(list), collections.defaultdict(list)
    for (m, t), v in A.items(): av[t].append(v)
    for (m, t), v in B_.items(): bv[t].append(v)
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(tasks), len(tasks))
        a = [x for i in idx for x in av[tasks[i]]]
        b = [x for i in idx for x in bv[tasks[i]]]
        diffs.append(np.mean(a) - np.mean(b))
    return np.percentile(diffs, [2.5, 97.5])

def macro(d):
    bym = collections.defaultdict(list)
    for (m, t), v in d.items(): bym[m].append(v)
    return float(np.mean([np.mean(v) for v in bym.values()]))

def main():
    fa = json.load(open(f'{OUT}/fold_assignments.json'))
    RT = set(fa['ood_rtllm_tasks'])

    # ===== 1) forward 口径修正 =====
    # OOD RankEF: 每 (model, task) 一次批 forward(batch=4), 5 折
    rkz = np.load(f'{OUT}/results/rtllm_rankek_scores_5fold.npz')
    sizes = collections.Counter(len(rkz[k]) for k in rkz.files)
    ood_batch_calls = len(set((k.split('#')[0], k.split('#', 2)[2], k.split('#')[1])
                              for k in rkz.files))
    ood_cand = sum(len(rkz[k]) for k in rkz.files)
    # IID RankEF: 逐 CSV 统计
    iid_rows = 0; iid_batches = 0
    for k in range(5):
        fp = f'{OUT}/predictions/rankek_scores_fold{k}.csv'
        groups = set()
        for r in csv.DictReader(open(fp)):
            iid_rows += 1
            groups.add((r['model'], r['task']))
        iid_batches += len(groups)
    # SLM
    slm_ood = 0; slm_iid = 0
    for fp in glob.glob(f'{OUT}/predictions/slm_judge_scores_*.csv'):
        for r in csv.DictReader(open(fp)):
            if r['task'] in RT: slm_ood += 1
            else: slm_iid += 1

    # ===== 2) 两臂逐任务成本+决策 =====
    thr = json.load(open(f'{OUT}/results/rankef_adaptive_thresholds.json'))
    Y, GL = {}, {}
    for m in MODELS:
        gd = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if r['task'] in RT and int(r['generation_id']) < BS:
                gd[r['task']][int(r['generation_id'])] = int(bool(r['simulation_pass']))
        for t, d in gd.items():
            if len(d) == BS: Y[(m, t)] = [d[g] for g in range(BS)]
        gl = {}
        for line in open(f'{STEP2}/{m}/labels.jsonl'):
            r = json.loads(line)
            if r['task'] in RT and int(r['generation_id']) < BS:
                gl[(r['task'], int(r['generation_id']))] = int(r['gen_len'])
        GL[m] = gl
    PP = collections.defaultdict(dict)
    for m in MODELS:
        z = np.load(f'{OUT}/predictions/q2v2/{m}_pool.npz')
        for key in z.files:
            b, a = key.rsplit('|', 1)
            if b.startswith('OOD|') and a in ('P1', 'P2'):
                _, k, t = b.split('|', 2)
                PP[(m, int(k), t)][a] = z[key][:BS]
    # 逐候选 scorer input tokens
    it = {}
    for r in csv.DictReader(open(f'{OUT}/predictions/rtllm_rankek_in_tokens.csv')):
        it[(r['model'], int(r['fold'][4]), r['task'], int(r['gid']))] = (
            int(r['in_tokens_raw']), int(r['in_tokens_capped512']))

    arm_rows = []
    arm_tp = {'rankef_adaptive@4': {}, 'hybrid@4': {}}
    arm_tok = {'rankef_adaptive@4': {}, 'hybrid@4': {}}
    for kk in rkz.files:
        m, k, t = kk.split('#', 2); k = int(k)
        if (m, t) not in Y:
            continue
        sc = rkz[kk]; y = Y[(m, t)]
        tau = thr.get(f'{m}#fold{k}', {}).get('tau_R')
        common = dict(model=m, fold=f'fold{k}', task=t)
        # rankef_adaptive
        kept = thr_cascade(list(sc), BS, tau if tau is not None else 0.5)
        sub = int(np.argmax([sc[i] for i in kept]))
        ser_raw = sum(it[(m, k, t, i)][0] for i in kept)
        ser_cap = sum(it[(m, k, t, i)][1] for i in kept)
        full_raw = sum(it[(m, k, t, i)][0] for i in range(BS))
        arm_rows.append({**common, 'arm': 'rankef_adaptive@4',
                         'kept_gids': ';'.join(map(str, kept)),
                         'n_candidates_scored_serial': len(kept),
                         'forward_calls_serial_batch1': len(kept),
                         'forward_calls_offline_batched_b4': 1,
                         'scorer_in_tokens_serial_raw': ser_raw,
                         'scorer_in_tokens_serial_capped512': ser_cap,
                         'scorer_in_tokens_full4_raw': full_raw,
                         'submitted_gid': sub, 'submitted_pass': int(y[sub]),
                         'used_candidates': len(kept),
                         'gen_output_tokens': sum(GL[m][(t, i)] for i in kept),
                         'tau_R_used': tau if tau is not None else ''})
        arm_tp['rankef_adaptive@4'].setdefault((m, t), []).append(int(y[sub]))
        arm_tok['rankef_adaptive@4'].setdefault((m, t), []).append(sum(GL[m][(t, i)] for i in kept))
        # hybrid
        pool = PP.get((m, k, t), {})
        if 'P1' in pool and 'P2' in pool:
            p1 = np.array(pool['P1'], dtype=np.float64)
            p2 = np.array(pool['P2'], dtype=np.float64)
            kh = rp_cascade(p1, p2, BS)
            sh = int(np.argmax([sc[i] for i in kh]))
            ser_raw_h = sum(it[(m, k, t, i)][0] for i in kh)
            ser_cap_h = sum(it[(m, k, t, i)][1] for i in kh)
            arm_rows.append({**common, 'arm': 'hybrid@4',
                             'kept_gids': ';'.join(map(str, kh)),
                             'n_candidates_scored_serial': len(kh),
                             'forward_calls_serial_batch1': len(kh),
                             'forward_calls_offline_batched_b4': 1,
                             'scorer_in_tokens_serial_raw': ser_raw_h,
                             'scorer_in_tokens_serial_capped512': ser_cap_h,
                             'scorer_in_tokens_full4_raw': full_raw,
                             'submitted_gid': sh, 'submitted_pass': int(y[sh]),
                             'used_candidates': len(kh),
                             'gen_output_tokens': sum(GL[m][(t, i)] for i in kh),
                             'tau_R_used': ''})
            arm_tp['hybrid@4'].setdefault((m, t), []).append(int(y[sh]))
            arm_tok['hybrid@4'].setdefault((m, t), []).append(sum(GL[m][(t, i)] for i in kh))
    with open(f'{OUT}/results/freeze_arm_scorer_cost.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['model','fold','task','arm','kept_gids',
                                           'n_candidates_scored_serial',
                                           'forward_calls_serial_batch1',
                                           'forward_calls_offline_batched_b4',
                                           'scorer_in_tokens_serial_raw',
                                           'scorer_in_tokens_serial_capped512',
                                           'scorer_in_tokens_full4_raw',
                                           'submitted_gid','submitted_pass',
                                           'used_candidates','gen_output_tokens','tau_R_used'])
        w.writeheader(); w.writerows(arm_rows)

    # ===== 3) 从逐任务 CSV 复算宏平均 + cluster CI, 对账 replay JSON =====
    rep = json.load(open(f'{OUT}/results/rtllm_policy_replay_v2.json'))
    alone_f, rec_f = collections.defaultdict(list), collections.defaultdict(list)
    for r in csv.DictReader(open(f'{OUT}/predictions/decisions_q2v2.csv')):
        if r['domain'] != 'OOD':
            continue
        if r['method'] == 'alone' and r['B'] == '4':
            alone_f[(r['model'], r['task'])].append(int(r['submitted_pass']))
        elif r['method'] == 'recprobe' and r['B'] == '4':
            rec_f[(r['model'], r['task'])].append(int(r['submitted_pass']))
    alone = {kk: float(np.mean(v)) for kk, v in alone_f.items()}
    rec = {kk: float(np.mean(v)) for kk, v in rec_f.items()}
    verify = {}
    for arm in ('rankef_adaptive@4', 'hybrid@4'):
        tp = {kk: float(np.mean(v)) for kk, v in arm_tp[arm].items()}
        tokm = {kk: float(np.mean(v)) for kk, v in arm_tok[arm].items()}
        mp = macro(tp)*100
        mt = macro(tokm)
        lo, hi = cluster_boot(tp, alone)
        lo2, hi2 = cluster_boot(tp, rec)
        got = rep['arms'][arm]
        verify[arm] = {
            'csv_macro_pass': round(mp, 2), 'json_pass': got['pass']*100,
            'csv_macro_tokens': round(mt, 0), 'json_tokens': got['mean_tokens'],
            'csv_vs_alone_CI': [round(100*float(lo), 2), round(100*float(hi), 2)],
            'json_vs_alone_CI': [got.get('delta', {}).get('lo') if isinstance(got.get('delta'), dict) else
                                 rep['deltas'][arm]['vs_alone_pp']['lo'],
                                 rep['deltas'][arm]['vs_alone_pp']['hi']],
            'csv_vs_recprobe_CI': [round(100*float(lo2), 2), round(100*float(hi2), 2)],
            'json_vs_recprobe_CI': [rep['deltas'][arm]['vs_recprobe_pp']['lo'],
                                    rep['deltas'][arm]['vs_recprobe_pp']['hi']]}
    # ===== 4) 更新成本 JSON =====
    cpath = f'{OUT}/results/scorer_cost_ood.json'
    cost = json.load(open(cpath))
    cost['forward_accounting_correction'] = {
        'note': '冻结轮修正: 早期 scorer_forwards 字段实为候选评分数; 现区分 '
                'candidate_scorings(逐候选) 与 forward_calls(批 forward)',
        'rankef_ood': {'candidate_scorings': int(ood_cand),
                       'forward_calls_batched': int(ood_batch_calls),
                       'batch_size_uniform': dict(sizes),
                       'per_fold_checkpoint': {'candidate_scorings': int(ood_cand/5),
                                               'forward_calls_batched': int(ood_batch_calls/5)},
                       'deployment_single_fold': {'candidate_scorings': int(ood_cand/5),
                                                  'forward_calls_batched': int(ood_batch_calls/5)}},
        'rankef_iid': {'candidate_scorings': iid_rows,
                       'forward_calls_batched': iid_batches,
                       'batch_size': 'variable (1-4, per task candidate count)'},
        'slm_judge': {'ood_request_level_forwards': slm_ood,
                      'iid_request_level_forwards': slm_iid,
                      'note': 'sglang 请求级计法; 每请求 1 输出 token'},
        'cpp_lat': {'candidate_scorings': 0, 'forward_calls': 0}}
    # 两臂部署成本(serial 口径)
    for arm, key in (('rankef_adaptive@4', 'rankef'), ('hybrid@4', 'rankef')):
        rs = [r for r in arm_rows if r['arm'] == arm]
        cost.setdefault('arm_deployment_cost', {})[arm] = {
            'n_tasks': len(rs),
            'mean_candidates_scored_serial': round(float(np.mean(
                [r['n_candidates_scored_serial'] for r in rs])), 3),
            'mean_forward_calls_serial': round(float(np.mean(
                [r['forward_calls_serial_batch1'] for r in rs])), 3),
            'mean_scorer_in_tokens_serial_raw': round(float(np.mean(
                [r['scorer_in_tokens_serial_raw'] for r in rs])), 1),
            'mean_scorer_in_tokens_serial_capped512': round(float(np.mean(
                [r['scorer_in_tokens_serial_capped512'] for r in rs])), 1),
            'note': 'serial=在线逐候选评分(停止即不再评分); offline replay 已预计算全 4 候选'}
    json.dump(cost, open(cpath, 'w'), indent=1)
    print(json.dumps({'forward': cost['forward_accounting_correction'],
                      'verify': verify,
                      'arm_deployment_cost': cost['arm_deployment_cost']}, indent=1))
    print(f'arm cost rows: {len(arm_rows)}')

if __name__ == '__main__':
    main()
