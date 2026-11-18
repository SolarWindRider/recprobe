#!/usr/bin/env python3
"""P0-3(辅助指标): 每模型 prompt 输入 token 数(总 API token = 输入+输出)。

仅用 tokenizer 计数(CPU), 不做推理。VE 312 + RTLLM 50 任务去重 prompt。
输出: results/prompt_tokens.json
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, OUT

STEP2 = './experiments/step2'


def main(models):
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    tasks = {}
    for t in json.load(open(f'{STEP2}/tasks_step2.json')):
        tasks[t['task_id']] = t.get('prompt', '')
    ve = sorted(set(t for k in range(5) for t in folds['folds'][f'fold{k}']['test_tasks']))
    rtllm = sorted(set(folds['ood_rtllm_tasks']))
    out = {'note': '输入 token = 每任务去重 prompt 的 tokenizer 计数(不含 chat 模板/多候选重复); '
                   '多候选采样共享同一 prompt 时, 实际输入成本应按采样次数放大',
           'models': {}}
    for m in models:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(f'models/{m}', trust_remote_code=True)
        ve_n = [len(tok(tasks[t])['input_ids']) for t in ve if t in tasks]
        rt_n = [len(tok(tasks[t])['input_ids']) for t in rtllm if t in tasks]
        out['models'][m] = {
            've_mean': round(float(np.mean(ve_n)), 1),
            've_p50': round(float(np.percentile(ve_n, 50)), 1),
            've_p95': round(float(np.percentile(ve_n, 95)), 1),
            'rtllm_mean': round(float(np.mean(rt_n)), 1)}
        print(m, out['models'][m], flush=True)
    json.dump(out, open(f'{OUT}/results/prompt_tokens.json', 'w'), indent=1, ensure_ascii=False)
    print('-> results/prompt_tokens.json')


if __name__ == '__main__':
    ms = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    main(ms)
