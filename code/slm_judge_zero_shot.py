#!/usr/bin/env python3
"""P2: SLM-as-a-Judge zero-shot(Qwen2.5-Coder-3B-Instruct)逐候选打分。

方案 §4.2 模板;score = P('1')/(P('1')+P('0'))(首 token top-20 logprobs)。
无 silent fallback:既无 '0' 也无 '1' 的样本 fmt_ok=False 且 score=None,计数披露。
每 (model, task, gid) 只打一次分(judge 与折无关);覆盖 VE test + RTLLM 前 4 候选。
输出: predictions/slm_judge_scores.csv(+ fmt 失败计数于末行日志)
用法: venv_sg/bin/python3 slm_judge_zero_shot.py <model> <card>   # model=被评生成器
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = f'{HERE}/..'
STEP2 = './experiments/step2'
BS_MAIN = 4


def main(gen, card):
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = card
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    tasks = {t['task_id']: t.get('prompt', '')
             for t in json.load(open(f'{STEP2}/tasks_step2.json'))}
    t2k = {}
    for k in range(5):
        for t in folds['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    # 候选: 每个 (task, gid<=3) 的提取 RTL
    cands = {}
    for r in map(json.loads, open(f'{STEP2}/{gen}/gen.jsonl')):
        gid = int(r['generation_id'])
        if gid < BS_MAIN:
            cands[(r['task'], gid)] = (r.get('rtl') or r.get('raw_output', ''))
    need = sorted({(t, g) for (t, g) in cands if t in tasks})
    print(f'[{gen}|judge3b] {len(need)} 候选', flush=True)

    TEMPLATE = (
        "You will be provided with the design specification and a candidate RTL implementation.\n"
        "Your task is to determine whether the candidate correctly implements the specification\n"
        "under the supplied functional behavior.\n\n"
        "Output:\n0 = Incorrect implementation\n1 = Correct implementation\n\n"
        "Specification:\n{spec}\n\nCandidate RTL:\n{rtl}\n\nAnswer:")

    sys.path.insert(0, './scripts')
    import _sglang_patch  # noqa: F401
    from sglang.srt.entrypoints.engine import Engine
    from transformers import AutoTokenizer
    jm = 'models/Qwen2.5-Coder-3B-Instruct'
    tok = AutoTokenizer.from_pretrained(jm, trust_remote_code=True)
    engine = Engine(model_path=jm, dtype='bfloat16', trust_remote_code=True,
                    log_level='error', skip_tokenizer_init=True,
                    disable_cuda_graph=True)
    reqs, keys = [], []
    for (t, g) in need:
        msg = TEMPLATE.format(spec=tasks[t][:6000], rtl=cands[(t, g)][:6000])
        try:
            p = tok.apply_chat_template([{'role': 'user', 'content': msg}],
                                        tokenize=False, add_generation_prompt=True)
        except Exception:
            p = msg + "\nAnswer (0/1):"
        reqs.append(tok(p)['input_ids'])
        keys.append((t, g))
    outs = engine.generate(input_ids=reqs,
                           sampling_params={'temperature': 0, 'max_new_tokens': 1},
                           return_logprob=True, logprob_start_len=-1,
                           top_logprobs_num=20)
    engine.shutdown()

    rows, fmt_fail = [], 0
    for (t, g), o in zip(keys, outs):
        top = (o.get('meta_info', {}).get('output_top_logprobs') or [[None]])
        p0 = p1 = 0.0
        for lp, tid, *_ in (top[0] or []):
            s = (tok.decode([tid]) or '').strip()
            if s == '0':
                p0 = float(np.exp(float(lp)))
            elif s == '1':
                p1 = float(np.exp(float(lp)))
        if p0 + p1 > 0:
            score = p1 / (p0 + p1)
            ok = True
        else:
            score, ok = None, False
            fmt_fail += 1
        k = t2k.get(t)
        rows.append(dict(model=gen, fold=f'fold{k}' if k is not None else 'OOD',
                         task=t, gid=g, score='' if score is None else round(score, 6),
                         fmt_ok=ok))
    outp = f'{OUT}/predictions/slm_judge_scores_{gen}.csv'
    import csv
    with open(outp, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['model', 'fold', 'task', 'gid', 'score', 'fmt_ok'])
        w.writeheader(); w.writerows(rows)
    print(f'[{gen}|judge3b] DONE rows={len(rows)} fmt_fail={fmt_fail} '
          f'({100*(1-fmt_fail/len(rows)):.1f}% 可解析) -> {outp}', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
