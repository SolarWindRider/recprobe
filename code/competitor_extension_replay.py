#!/usr/bin/env python3
"""竞品排序器预算扩展: lat_fixed / slm_fixed @ B∈{2,4,8} × {IID, OOD}。

与 freeze_budget_replay.py 同 emit 约定(Fixed: 前 B 候选 argmax, 并列最早 gid;
serial scorer cost)。输出追加到 results/freeze_competitor_extension.csv,
并把两条线的宏平均并入 figure_data_budget.json。
"""
import json, os, sys, csv, glob, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, OUT

STEP2 = 'experiments/step2'
BUDGETS = [2, 4, 8]

def main():
    fa = json.load(open(f'{OUT}/fold_assignments.json'))
    t2k = {}
    for k in range(5):
        for t in fa['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    OODT = sorted(set(fa['ood_rtllm_tasks']))

    # 标签/token gid0..7
    Y, GL = {}, {}
    for m in MODELS:
        gd = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if int(r['generation_id']) < 8:
                gd[r['task']][int(r['generation_id'])] = int(bool(r['simulation_pass']))
        gl = {}
        for line in open(f'{STEP2}/{m}/labels.jsonl'):
            r = json.loads(line)
            if int(r['generation_id']) < 8:
                gl[(r['task'], int(r['generation_id']))] = int(r['gen_len'])
        Y[m] = gd; GL[m] = gl

    # 分数: LAT
    LAT = collections.defaultdict(dict)
    z = np.load(f'{OUT}/results/rtllm_lat_scores.npz')
    for key in z.files:
        m, k, t = key.split('#', 2)
        for g, s in enumerate(z[key].tolist()[:4]):
            LAT[(m, int(k), t, 'OOD')][g] = float(s)
    for r in csv.DictReader(open(f'{OUT}/predictions/lat_l27_scores.csv')):
        k = int(r['fold'][-1])
        if r['task'] in t2k and t2k[r['task']] == k:
            LAT[(r['model'], k, r['task'], 'IID')][int(r['gid'])] = float(r['score'])
    for dom, fp in (('IID', f'{OUT}/predictions/lat_scores_gids47_iid.csv'),
                    ('OOD', f'{OUT}/predictions/lat_scores_gids47_ood.csv')):
        for r in csv.DictReader(open(fp)):
            k = int(r['fold'][4])
            LAT[(r['model'], k, r['task'], dom)][int(r['gid'])] = float(r['score'])
    # 分数: SLM(per-model 文件含 fold 列; gid0-3) + gids47 文件
    SL = collections.defaultdict(dict)
    for fp in glob.glob(f'{OUT}/predictions/slm_judge_scores_*.csv'):
        if 'gids47' in fp:
            continue
        for r in csv.DictReader(open(fp)):
            if r['score'] == '':
                continue
            k = t2k.get(r['task'])
            dom = 'OOD' if r['fold'] == 'OOD' else 'IID'
            if dom == 'IID' and (k is None or r['fold'] != f'fold{k}'):
                continue
            key = (r['model'], k, r['task'], dom) if dom == 'IID' else \
                  (r['model'], None, r['task'], 'OOD')
            SL[key][int(r['gid'])] = float(r['score'])
    _p47 = f'{OUT}/predictions/slm_judge_scores_gids47.csv'
    for r in (csv.DictReader(open(_p47)) if os.path.exists(_p47) else []):
        if r['score'] == '':
            continue
        k = t2k.get(r['task'])
        dom = 'OOD' if r['fold'] == 'OOD' else 'IID'
        if dom == 'IID' and (k is None or r['fold'] != f'fold{k}'):
            continue
        key = (r['model'], k, r['task'], dom) if dom == 'IID' else \
              (r['model'], None, r['task'], 'OOD')
        SL[key][int(r['gid'])] = float(r['score'])

    out_rows = []
    for m in MODELS:
        for k in range(5):
            for dom, tasks in (('IID', fa['folds'][f'fold{k}']['test_tasks']),
                               ('OOD', OODT)):
                for t in tasks:
                    ym = Y[m].get(t, {})
                    if len(ym) < 8:
                        continue
                    y = [ym[g] for g in range(8)]
                    for sc_store, pol, s_tok_mode in (
                            (LAT, 'lat_fixed', 'zero'),
                            (SL, 'slm_fixed', 'prompt')):
                        key = ((m, None, t, 'OOD') if (dom == 'OOD' and pol == 'slm_fixed')
                               else (m, k, t, dom))
                        sm = sc_store.get(key, {})
                        if pol == 'slm_fixed' and len(sm) < 4:
                            continue
                        if pol != 'slm_fixed' and len(sm) < 8:
                            continue
                        for B in BUDGETS:
                            avail = [g for g in range(B)
                                     if (sm.get(g) is not None) or pol != 'slm_fixed']
                            if not avail:
                                continue
                            sub = int(np.argmax([sm[g] for g in avail]))
                            tok_ = sum(GL[m][(t, i)] for i in range(B))
                            s_forw = B
                            if s_tok_mode == 'zero':
                                s_tok = 0.0
                            else:
                                s_tok = sum(SLM_TOK.get((m, t, g), 0.0)
                                            for g in range(B))
                            out_rows.append(dict(
                                domain=dom, model=m, fold=f'fold{k}', task=t,
                                policy=pol, max_B=B, actual_n_generated=B,
                                selected_candidate_gid=sub, pass_=y[sub],
                                gen_output_tokens=tok_, extra_scorer_forwards=s_forw,
                                scorer_input_tokens=round(s_tok, 1)))

    exists = os.path.exists(f'{OUT}/results/freeze_competitor_extension.csv')
    with open(f'{OUT}/results/freeze_competitor_extension.csv', 'a', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['domain','model','fold','task','policy','max_B',
                                           'actual_n_generated','selected_candidate_gid',
                                           'pass_','gen_output_tokens',
                                           'extra_scorer_forwards','scorer_input_tokens'])
        if not exists:
            w.writeheader()
        have = set()
        if exists:
            for r in csv.DictReader(open(f'{OUT}/results/freeze_competitor_extension.csv')):
                have.add((r['domain'], r['model'], r['fold'], r['task'],
                          r['policy'], int(r['max_B'])))
        for r in out_rows:
            kk = (r['domain'], r['model'], r['fold'], r['task'], r['policy'], r['max_B'])
            if kk not in have:
                w.writerow(r)
    # 宏平均 → figure_data_budget.json
    fig = json.load(open(f'{OUT}/results/figure_data_budget.json'))
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [])))
    for r in out_rows:
        key = (r['domain'], r['policy'], r['max_B'])
        per[key][r['model']][0].append(r['pass_'])
        per[key][r['model']][1].append(r['gen_output_tokens'])
    for key, bym in per.items():
        fig['macro']['|'.join(map(str, key))] = {
            'pass_pct': round(100*float(np.mean([np.mean(v[0]) for v in bym.values()])), 2),
            'tokens': round(float(np.mean([np.mean(v[1]) for v in bym.values()])), 1),
            'n': sum(len(v[0]) for v in bym.values())}
    json.dump(fig, open(f'{OUT}/results/figure_data_budget.json', 'w'), indent=1)
    for pol in ('lat_fixed', 'slm_fixed'):
        for dom in ('IID', 'OOD'):
            line = [f"{dom}|{pol}|{B}: " +
                    json.dumps(fig['macro'].get(f'{dom}|{pol}|{B}', {})) for B in BUDGETS]
            print(pol, dom, ' | '.join(line), flush=True)
    print(f'rows written: {len(out_rows)}')

SLM_TOK = {}

if __name__ == '__main__':
    # SLM gid0-3 prompt tokens 现算(旧 CSV 无 token 列)
    import hashlib
    if len(sys.argv) > 2 and sys.argv[2] == 'tokens':
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained('models/Qwen2.5-Coder-3B-Instruct',
                                            trust_remote_code=True)
        TM = ('You will be provided with the design specification and a candidate RTL implementation.\n'
              'Your task is to determine whether the candidate correctly implements the specification\n'
              'under the supplied functional behavior.\n\n'
              'Output:\n0 = Incorrect implementation\n1 = Correct implementation\n\n'
              'Specification:\n{spec}\n\nCandidate RTL:\n{rtl}\n\nAnswer:')
        tm = {t['task_id']: t.get('prompt', '') for t in
              json.load(open(f'{STEP2}/tasks_step2.json'))}
        seen = set()
        for fp in glob.glob(f'{OUT}/predictions/slm_judge_scores_*.csv'):
            if 'gids47' in fp:
                continue
            for r in csv.DictReader(open(fp)):
                seen.add((r['model'], r['task'], int(r['gid'])))
        for m in MODELS:
            ev = collections.defaultdict(dict)
            for line in open(f'{STEP2}/{m}/eval.jsonl'):
                r = json.loads(line)
                if (m, r['task'], int(r['generation_id'])) in seen:
                    ev[r['task']][int(r['generation_id'])] = \
                        (r.get('rtl') or r.get('raw_output', ''))[:6000]
            cnt = 0
            for t, gd in ev.items():
                for g, code in gd.items():
                    msg = TM.format(spec=tm.get(t, '')[:6000], rtl=code)
                    try:
                        p = tok.apply_chat_template([{'role': 'user', 'content': msg}],
                                                    tokenize=False,
                                                    add_generation_prompt=True)
                    except Exception:
                        p = msg + "\nAnswer (0/1):"
                    SLM_TOK[(m, t, g)] = len(tok(p)['input_ids'])
                    cnt += 1
            print(f'  {m}: {cnt} gid0-3 prompts', flush=True)
    main()
