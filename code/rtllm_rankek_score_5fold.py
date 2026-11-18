#!/usr/bin/env python3
"""RankEF RTLLM OOD 打分(修正版): 微调 encoder + cls_head, 五折。

修正(审计 P0): encoder 必须从 checkpoints/rankek_fold{k}/ 加载(与 IID
rankek_score.py 完全一致), 而非 models/codet5p-770m 预训练 base。
输出 results/rtllm_rankek_scores_5fold.npz + checkpoint_provenance.json
"""
import json, os, sys, hashlib, collections
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
BS = 4

folds = json.load(open(f'{OUT}/fold_assignments.json'))
rtllm = sorted(set(folds['ood_rtllm_tasks']))
tasks_meta = {t['task_id']: t.get('prompt', '') for t in json.load(open(f'{STEP2}/tasks_step2.json'))}

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = sys.argv[1] if len(sys.argv) > 1 else '4'
dev = 'npu'
tok = AutoTokenizer.from_pretrained('models/codet5p-770m', trust_remote_code=True)
from transformers import AutoModelForSeq2SeqLM

provenance = {}
results = {}
for k in range(5):
    ckpt_dir = f'{OUT}/checkpoints/rankek_fold{k}'
    # encoder 权重指纹(与 IID 同一文件)
    binp = f'{ckpt_dir}/pytorch_model.bin'
    provenance[f'fold{k}_encoder_sha256_1MB'] = hashlib.sha256(
        open(binp, 'rb').read(1 << 20)).hexdigest()
    provenance[f'fold{k}_encoder_path'] = os.path.abspath(binp)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir, trust_remote_code=True, torch_dtype=torch.float32)
    ckpt = torch.load(f'{ckpt_dir}/cls_best.pt', map_location='cpu')
    cls_head = nn.Linear(model.config.d_model, 2)
    cls_head.load_state_dict(ckpt['cls_head'])
    model = model.npu().eval(); cls_head = cls_head.to(dev).eval()
    provenance[f'fold{k}_cls_step'] = ckpt.get('step')
    provenance[f'fold{k}_cls_val_f1'] = ckpt.get('val_f1')
    print(f'fold{k}: encoder={ckpt_dir} step={ckpt.get("step")} val_f1={ckpt.get("val_f1"):.4f}', flush=True)

    for m in MODELS:
        rtl_map = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/gen.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm and int(r['generation_id']) < BS:
                rtl_map[r['task']][int(r['generation_id'])] = \
                    (r.get('rtl') or r.get('raw_output') or '')[:4000]
        for t in rtl_map:
            spec = tasks_meta.get(t, '')[:3000]
            texts = [f"Task: {spec}\nCode:\n{rtl_map[t].get(g, '')}" for g in range(BS)]
            enc = tok(texts, truncation=True, max_length=512,
                      padding='max_length', return_tensors='pt')
            enc = {kk: v.to(dev) for kk, v in enc.items()}
            dec = torch.full((BS, 1), 0, device=dev)
            with torch.no_grad():
                o = model(input_ids=enc['input_ids'],
                          attention_mask=enc['attention_mask'],
                          decoder_input_ids=dec, output_hidden_states=True)
                h = o.encoder_last_hidden_state.mean(1)
                logits = cls_head(h).cpu()
            probs = torch.softmax(logits, 1)[:, 1]
            results[(m, k, t)] = probs.tolist()
    del model, cls_head
    torch.npu.empty_cache()
    print(f'fold{k} done: {sum(1 for key in results if key[1]==k)} pools', flush=True)

out = {}
for (m, k, t), scores in results.items():
    out[f'{m}#{k}#{t}'] = scores
np.savez(f'{OUT}/results/rtllm_rankek_scores_5fold.npz', **out)
json.dump(provenance, open(f'{OUT}/results/rankek_ood_checkpoint_provenance.json', 'w'), indent=1)
print(f'Total: {len(results)} pools → rtllm_rankek_scores_5fold.npz', flush=True)
