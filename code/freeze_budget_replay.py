#!/usr/bin/env python3
"""Fig3/4 冻结重放: 六策略 × B∈{2,4,8} × {IID, OOD} 逐任务表。

策略: recprobe(CPP rank+CCG stop) / sc(lexical self-consistency) /
      judge(same-generator self-judge, 保留概率) / fixed(RankEF-Fixed) /
      adaptive(RankEF-Adaptive, 冻结 B=4 校准 τ_R 跨 B 共享) / hybrid(CCG stop + RankEF select)
锚点校验: 宏平均必须复现 audit 表与既有 v2/metrics JSON。
输出 results/freeze_budget_replay.csv + results/figure_data_budget.json
"""
import csv, json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, OUT, canonical_rows, load_texts

STEP2 = 'experiments/step2'
BUDGETS = [2, 4, 8]

def jaccard(a, b):
    A, B_ = set(a.split()), set(b.split())
    if not A and not B_:
        return 1.0
    return len(A & B_) / max(len(A | B_), 1)

def sc_pick(cands, texts):
    best_i, best_s = 0, -1.0
    for i in range(len(cands)):
        others = [j for j in range(len(cands)) if j != i]
        s = sum(jaccard(texts[cands[i]], texts[cands[j]]) for j in others) / max(len(others), 1)
        if s > best_s:
            best_s, best_i = s, i
    return best_i

def rp_cascade(p1, p2, B):
    kept = [0]
    for i in range(1, B):
        g = p2[kept[-1]]
        if p1[kept[-1]] < 0.5 and (np.isnan(g) or g > 0.5) and len(kept) < B:
            kept.append(i)
        else:
            break
    return kept

def thr_cascade(score, B, tau):
    kept = [0]
    for i in range(1, B):
        if score[kept[-1]] < tau and len(kept) < B:
            kept.append(i)
        else:
            break
    return kept

def judge_cascade(js, B):
    kept = [0]
    for i in range(1, min(B, len(js))):
        if js[kept[-1]] < 0.5:
            kept.append(i)
        else:
            break
    sub = max(kept, key=lambda i: js[i])
    return kept, sub

def main():
    fa = json.load(open(f'{OUT}/fold_assignments.json'))
    t2k = {}
    for k in range(5):
        for t in fa['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    OODT = sorted(set(fa['ood_rtllm_tasks']))
    thr = json.load(open(f'{OUT}/results/rankef_adaptive_thresholds.json'))
    rows = canonical_rows()
    texts_all = load_texts(MODELS[0], rows) if False else None  # texts per model loaded below

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

    # 池 P1/P2 (len 32)
    P = {}
    for m in MODELS:
        z = np.load(f'{OUT}/predictions/q2v2/{m}_pool.npz')
        d = {}
        for key in z.files:
            b, a = key.rsplit('|', 1)
            if a in ('P1', 'P2') and b.startswith(('IID|', 'OOD|')):
                _, k, t = b.split('|', 2)
                d[(a, int(k), t)] = z[key]
        P[m] = d

    # RankEF 分数 gid0..7: IID(fold CSVs + gids47) / OOD(npz + gids47)
    RK = collections.defaultdict(dict)  # (m, k, t, dom) -> {gid: score}
    IT = collections.defaultdict(dict)  # (m, k, t, dom) -> {gid: in_tokens}
    for k in range(5):
        fp = f'{OUT}/predictions/rankek_scores_fold{k}.csv'
        for r in csv.DictReader(open(fp)):
            key = (r['model'], k, r['task'], 'IID')
            RK[key][int(r['gid'])] = float(r['score'])
            IT[key][int(r['gid'])] = float(r.get('in_tokens') or 0)
    z = np.load(f'{OUT}/results/rtllm_rankek_scores_5fold.npz')
    for key in z.files:
        m, k, t = key.split('#', 2)
        for g, s in enumerate(z[key][:4]):
            RK[(m, int(k), t, 'OOD')][g] = float(s)
    for dom, fp in (('IID', f'{OUT}/predictions/rankek_scores_gids47_iid.csv'),
                    ('OOD', f'{OUT}/predictions/rankek_scores_gids47_ood.csv')):
        for r in csv.DictReader(open(fp)):
            k = int(r['fold'][4])
            key = (r['model'], k, r['task'], dom)
            RK[key][int(r['gid'])] = float(r['score'])
            IT[key][int(r['gid'])] = float(r['in_tokens'])
    for fp in (f'{OUT}/predictions/rtllm_rankek_in_tokens.csv',):
        for r in csv.DictReader(open(fp)):
            k = int(r['fold'][4])
            IT[(r['model'], k, r['task'], 'OOD')][int(r['gid'])] = float(r['in_tokens_raw'])

    # judge 概率 + prompt tokens
    JP = {}
    for m in MODELS:
        jp = f'{OUT}/judge_scores/{m}.json'
        JP[m] = json.load(open(jp)) if os.path.exists(jp) else {}

    out_rows = []
    fig = collections.defaultdict(lambda: [[], []])  # (dom, policy, B) -> [passes, tokens]
    anchors_fail = []

    def emit(dom, policy, m, k, t, B, kept, sub, s_forw, s_tok):
        y = Y[m][t][sub]
        tok_ = sum(GL[m][(t, i)] for i in range(max(kept) + 1)) if policy in (
            'recprobe', 'adaptive', 'hybrid', 'judge') else sum(
            GL[m][(t, i)] for i in range(B))
        n_gen = (max(kept) + 1) if policy in ('recprobe', 'adaptive', 'hybrid', 'judge') else B
        out_rows.append(dict(domain=dom, model=m, fold=f'fold{k}', task=t, policy=policy,
                             max_B=B, actual_n_generated=n_gen,
                             selected_candidate_gid=sub, pass_=y,
                             gen_output_tokens=tok_,
                             extra_scorer_forwards=s_forw, scorer_input_tokens=round(s_tok, 1),
                             kept_gids=';'.join(map(str, kept))))
        if policy == 'judge':
            for g in kept:
                JUDGE_KEPT[m].add((t, g))
        fig[(dom, policy, B)][0].append(y)
        fig[(dom, policy, B)][1].append(tok_)

    for m in MODELS:
        texts = load_texts(m, rows)
        for k in range(5):
            for dom in ('IID', 'OOD'):
                tasks = (fa['folds'][f'fold{k}']['test_tasks'] if dom == 'IID' else OODT)
                for t in tasks:
                    ym = Y[m].get(t, {})
                    if len(ym) < 8:
                        continue
                    y = [ym[g] for g in range(8)]
                    rk = RK.get((m, k, t, dom), {})
                    it = IT.get((m, k, t, dom), {})
                    if len(rk) < 8:
                        continue
                    p1 = P[m].get(('P1', k, t)); p2 = P[m].get(('P2', k, t))
                    js_raw = JP[m].get(t, {})
                    js = [float(js_raw.get(str(g), 0.5)) for g in range(8)]
                    tl = [texts.get((t, g), '') for g in range(8)]
                    for B in BUDGETS:
                        # recprobe
                        kept = rp_cascade(p1[:8].astype(np.float64), p2[:8].astype(np.float64), B)
                        sub = int(np.argmax(p1[:8][kept]))
                        emit(dom, 'recprobe', m, k, t, B, kept, sub, 0, 0.0)
                        # sc
                        cand = list(range(B))
                        sub = sc_pick(cand, tl)
                        emit(dom, 'sc', m, k, t, B, cand, sub, 0, 0.0)
                        # judge
                        kept, sub = judge_cascade(js, B)
                        emit(dom, 'judge', m, k, t, B, kept, sub, len(kept),
                             sum(it.get(g, 0.0) for g in kept) if False else 0.0)
                        # fixed
                        sub = int(np.argmax([rk[g] for g in range(B)]))
                        emit(dom, 'fixed', m, k, t, B, list(range(B)), sub, B,
                             sum(it.get(g, 0.0) for g in range(B)))
                        # adaptive
                        tau = thr.get(f'{m}#fold{k}', {}).get('tau_R', 0.5)
                        kept = thr_cascade([rk[g] for g in range(B)], B, tau)
                        sub = int(np.argmax([rk[g] for i, g in enumerate(kept)]))
                        emit(dom, 'adaptive', m, k, t, B, kept, sub, len(kept),
                             sum(it.get(g, 0.0) for g in kept))
                        # hybrid
                        kept = rp_cascade(p1[:8].astype(np.float64), p2[:8].astype(np.float64), B)
                        sub = int(np.argmax([rk[g] for g in kept]))
                        emit(dom, 'hybrid', m, k, t, B, kept, sub, len(kept),
                             sum(it.get(g, 0.0) for g in kept))

    # judge scorer input tokens: 仅 judged(kept) 候选, 每模型自身 tokenizer
    print('computing judge scorer input tokens...', flush=True)
    from transformers import AutoTokenizer
    tm = {tt['task_id']: tt.get('prompt', '') for tt in
          json.load(open(f'{STEP2}/tasks_step2.json'))}
    for m in MODELS:
        need = JUDGE_KEPT.get(m, set())
        if not need:
            continue
        tok = AutoTokenizer.from_pretrained(f'models/{m}', trust_remote_code=True)
        ev = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if r['task'] in {t for (t, g) in need}:
                ev[r['task']][int(r['generation_id'])] = (r.get('rtl') or r.get('raw_output', ''))[:6000]
        cnt = 0
        for (t, g) in sorted(need):
            code = ev.get(t, {}).get(g, '')
            q = (f"{tm.get(t, '')}\n\nYou wrote the following Verilog code:\n\n"
                 f"```verilog\n{code}\n```\n\n"
                 "Question: Will this exact code pass the provided testbench "
                 "without any modification? Answer with only one word, Yes or No.")
            try:
                p = tok.apply_chat_template([{'role': 'user', 'content': q}],
                                            tokenize=False, add_generation_prompt=True)
            except Exception:
                p = q + "\nAnswer (Yes/No):"
            JUDGE_TOK[(m, t, g)] = len(tok(p)['input_ids'])
            cnt += 1
        print(f'  {m}: {cnt} prompts', flush=True)
    for r in out_rows:
        if r['policy'] == 'judge':
            kept = [int(x) for x in r['kept_gids'].split(';')]
            r['scorer_input_tokens'] = round(float(sum(
                JUDGE_TOK.get((r['model'], r['task'], g), 0.0) for g in kept)), 1)
    # ---- 输出 ----
    with open(f'{OUT}/results/freeze_budget_replay.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['domain','model','fold','task','policy','max_B',
                                           'actual_n_generated','selected_candidate_gid',
                                           'pass_','gen_output_tokens',
                                           'extra_scorer_forwards','scorer_input_tokens'])
        w.writeheader()
        for r in out_rows:
            w.writerow({k: v for k, v in r.items() if k != 'kept_gids'})
    # 宏平均
    macro = {}
    per = collections.defaultdict(lambda: collections.defaultdict(lambda: ([], [])))
    for r in out_rows:
        key = (r['domain'], r['policy'], r['max_B'])
        per[key][r['model']][0].append(r['pass_'])
        per[key][r['model']][1].append(r['gen_output_tokens'])
    for key, bym in per.items():
        macro['|'.join(map(str, key))] = {
            'pass_pct': round(100*float(np.mean([np.mean(v[0]) for v in bym.values()])), 2),
            'tokens': round(float(np.mean([np.mean(v[1]) for v in bym.values()])), 1),
            'n': sum(len(v[0]) for v in bym.values())}
    json.dump({'macro': macro,
               'anchors': AUDIT},
              open(f'{OUT}/results/figure_data_budget.json', 'w'), indent=1)
    print(json.dumps(macro, indent=1)[:3000])
    print(f'rows: {len(out_rows)}')

JUDGE_KEPT = collections.defaultdict(set)
JUDGE_TOK = {}
AUDIT = {
    'iid': {'recprobe|2': [38.49, 2505.3], 'recprobe|4': [39.93, 2905.2], 'recprobe|8': [40.25, 3070.3],
            'sc|2': [35.58, 3706.6], 'sc|4': [39.42, 7390.5], 'sc|8': [40.92, 14776.0],
            'judge|2': [34.40, 2298.9], 'judge|4': [33.23, 2778.9], 'judge|8': [32.35, 3341.5],
            'fixed|4': [44.58, 7390.5], 'adaptive|4': [41.37, 3696.0], 'hybrid|4': [40.46, 2905.2]},
    'ood': {'recprobe|2': [33.23, 2779.8], 'recprobe|4': [33.90, 3323.3], 'recprobe|8': [34.13, 3669.6],
            'sc|2': [33.00, 3960.2], 'sc|4': [36.00, 7914.3], 'sc|8': [38.83, 15807.7],
            'judge|2': [30.75, None], 'judge|4': [29.25, None], 'judge|8': [27.75, None],
            'fixed|4': [39.67, 7914.3], 'adaptive|4': [37.20, 4860.0], 'hybrid|4': [34.83, 3323.3]}}

if __name__ == '__main__':
    main()
