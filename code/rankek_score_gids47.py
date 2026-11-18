#!/usr/bin/env python3
"""Fig3/4 冻结: RankEF gid=4..7 补分(IID 每折测试任务 + OOD, 5 折 checkpoint)。

协议与 rtllm_rankek_score_5fold.py / rankek_score.py 一致:
  文本 = f"Task: {spec[:3000]}\nCode:\n{rtl[:4000]}", max_length=512, batch=每任务 4 候选
输出 predictions/rankek_scores_gids47_{iid,ood}.csv
  列: model, fold, task, gid, score, in_tokens(截断后实际长度)
"""
import json, os, sys, csv, collections
import numpy as np
import torch, torch_npu
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']
GIDS = [4, 5, 6, 7]

fa = json.load(open(f'{OUT}/fold_assignments.json'))
t2k = {}
for k in range(5):
    for t in fa['folds'][f'fold{k}']['test_tasks']:
        t2k[t] = k
OOD = sorted(set(fa['ood_rtllm_tasks']))
tasks_meta = {t['task_id']: t.get('prompt', '') for t in json.load(open(f'{STEP2}/tasks_step2.json'))}

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '4'
dev = 'npu'
tok = AutoTokenizer.from_pretrained('models/codet5p-770m', trust_remote_code=True)

rows_iid, rows_ood = [], []
for k in range(5):
    d = f'{OUT}/checkpoints/rankek_fold{k}'
    model = AutoModelForSeq2SeqLM.from_pretrained(d, trust_remote_code=True, torch_dtype=torch.float32)
    ck = torch.load(f'{d}/cls_best.pt', map_location='cpu')
    head = nn.Linear(model.config.d_model, 2)
    head.load_state_dict(ck['cls_head'])
    model = model.npu().eval(); head = head.to(dev).eval()
    print(f'fold{k} loaded (step={ck.get("step")})', flush=True)

    def score(texts):
        enc = tok(texts, truncation=True, max_length=512, padding='max_length',
                  return_tensors='pt')
        ntok = enc['attention_mask'].sum(1).tolist()
        enc = {kk: v.to(dev) for kk, v in enc.items()}
        dec = torch.full((len(texts), 1), 0, device=dev)
        with torch.no_grad():
            o = model(input_ids=enc['input_ids'], attention_mask=enc['attention_mask'],
                      decoder_input_ids=dec, output_hidden_states=True)
            h = o.encoder_last_hidden_state.mean(1)
            logits = head(h).cpu()
        return torch.softmax(logits, 1)[:, 1].tolist(), [int(x) for x in ntok]

    # 候选文本缓存(每模型一次读 gen.jsonl)
    cache = {}
    for m in MODELS:
        d4 = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/gen.jsonl'):
            r = json.loads(line)
            g = int(r['generation_id'])
            if g in GIDS:
                d4[r['task']][g] = (r.get('rtl') or r.get('raw_output') or '')[:4000]
        cache[m] = d4

    # IID: 本折测试任务
    for m in MODELS:
        for t, gd in cache[m].items():
            if t2k.get(t) != k or len(gd) < 4:
                continue
            texts = [f"Task: {tasks_meta.get(t, '')[:3000]}\nCode:\n{gd.get(g, '')}" for g in GIDS]
            probs, ntok = score(texts)
            for j, g in enumerate(GIDS):
                rows_iid.append((m, f'fold{k}', t, g, round(probs[j], 6), ntok[j]))
    # OOD
    for m in MODELS:
        for t in OOD:
            gd = cache[m].get(t, {})
            if len(gd) < 4:
                continue
            texts = [f"Task: {tasks_meta.get(t, '')[:3000]}\nCode:\n{gd.get(g, '')}" for g in GIDS]
            probs, ntok = score(texts)
            for j, g in enumerate(GIDS):
                rows_ood.append((m, f'fold{k}', t, g, round(probs[j], 6), ntok[j]))
    del model, head
    torch.npu.empty_cache()
    print(f'fold{k} done: iid={len(rows_iid)} ood={len(rows_ood)}', flush=True)

with open(f'{OUT}/predictions/rankek_scores_gids47_iid.csv', 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(['model','fold','task','gid','score','in_tokens']); w.writerows(rows_iid)
with open(f'{OUT}/predictions/rankek_scores_gids47_ood.csv', 'w', newline='') as fo:
    w = csv.writer(fo); w.writerow(['model','fold','task','gid','score','in_tokens']); w.writerows(rows_ood)
print(f'TOTAL iid={len(rows_iid)} ood={len(rows_ood)}')
