#!/usr/bin/env python3
"""LLM-as-judge(self-eval)基线打分: 模型自评其密封集样本能否通过 testbench。

对每个密封任务的 gid 0..7, 构造判分 prompt, 贪心 1 token,
取 Yes/No 的归一化概率为分数。输出 experiments/step2/judge_scores/{m}.json
用法: step2_judge_scores.py <model> <card>
"""
import json
import os
import sys

ROOT = 'experiments/step2'


def main():
    m, card = sys.argv[1], sys.argv[2]
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = card
    split = json.load(open(f'{ROOT}/sealed_split.json'))
    seals = set(split['sealed_iid']) | set(split['sealed_shift'])
    all_tasks = json.load(open(f'{ROOT}/tasks_step2.json'))
    seal_ids = {t['task_id'] for t in all_tasks if t['design_group'] in seals}
    tasks = {t['task_id']: t for t in all_tasks if t['task_id'] in seal_ids}
    ev = {}
    for r in map(json.loads, open(f'{ROOT}/{m}/eval.jsonl')):
        if r['task'] in tasks:
            ev[(r['task'], r['generation_id'])] = (r.get('rtl') or
                                                   r.get('raw_output', ''))
    # 组 prompt(用模型自己的 chat template)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(f'models/{m}', trust_remote_code=True)

    def to_prompt(desc, code):
        q = (f"{desc}\n\nYou wrote the following Verilog code:\n\n"
             f"```verilog\n{code}\n```\n\n"
             "Question: Will this exact code pass the provided testbench "
             "without any modification? Answer with only one word, "
             "Yes or No.")
        try:
            return tok.apply_chat_template([{'role': 'user', 'content': q}],
                                           tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            return q + "\nAnswer (Yes/No):"

    reqs, keys = [], []
    for tid, t in sorted(tasks.items()):
        for gid in range(8):
            key = (tid, gid)
            if key not in ev or not ev[key].strip():
                continue
            reqs.append(to_prompt(t['prompt'], ev[key][:6000]))
            keys.append(key)
    print(f'[{m}|judge] {len(reqs)} 判分请求', flush=True)

    from sglang.srt.entrypoints.engine import Engine
    engine = Engine(model_path=f'models/{m}', dtype='bfloat16',
                    trust_remote_code=True, log_level='error',
                    skip_tokenizer_init=True)
    enc = tok(reqs)["input_ids"]
    outs = engine.generate(input_ids=enc,
                           sampling_params={"temperature": 0,
                                            "max_new_tokens": 1},
                           return_logprob=True, logprob_start_len=-1,
                           top_logprobs_num=15)
    engine.shutdown()

    scores = {}
    n_hit = 0
    for key, o in zip(keys, outs):
        top = (o.get('meta_info', {}).get('output_top_logprobs') or [[None]])
        first = top[0] or []
        p_yes = p_no = 0.0
        for lp, tid, *_ in first:
            tokstr = (tok.decode([tid]) or '').strip().lower()
            if tokstr == 'yes':
                p_yes = float(np.exp(float(lp)))
            elif tokstr == 'no':
                p_no = float(np.exp(float(lp)))
        if p_yes + p_no > 0:
            score = p_yes / (p_yes + p_no)
            n_hit += 1
        else:
            score = 0.5
        scores.setdefault(key[0], {})[str(key[1])] = round(score, 4)
    os.makedirs(f'{ROOT}/judge_scores', exist_ok=True)
    json.dump(scores, open(f'{ROOT}/judge_scores/{m}.json', 'w'), indent=1)
    print(f'[{m}|judge] DONE: {n_hit}/{len(keys)} 命中 Yes/No -> '
          f'{ROOT}/judge_scores/{m}.json', flush=True)


import numpy as np  # noqa: E402

if __name__ == '__main__':
    main()
