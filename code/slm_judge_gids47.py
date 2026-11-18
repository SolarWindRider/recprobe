#!/usr/bin/env python3
"""SLM-Judge gid=4..7 补分(transformers + torch_npu 版, 资源受限环境)。

与 slm_judge_zero_shot.py 同模板/同截断/同 top-20 归一化打分。
推理: bf16 权重 + 末位 hidden state 手动 lm_head(避免全词表 logits 显存爆炸)。
分块 batch, 逐模型增量写 predictions/slm_judge_scores_gids47.csv。
"""
import json, os, sys, csv


def main(card):
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = card
    import torch
    import torch_npu  # noqa: F401
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM

    OUT = 'experiments/recprobe_group5fold'
    STEP2 = 'experiments/step2'
    MODELS = ['CodeV-QW-7B', 'CodeV-DS-6.7B', 'VeriCoder_Qwen14B',
              'CodeV-R1-Distill-Qwen-7B', 'CodeV-R1-RL-Qwen-7B',
              'RTLCoder-Deepseek-v1.1', 'DeepSeek-Coder-6.7B',
              'VeriSeek', 'SiliconMind-V1.2-Qwen3-8B',
              'SiliconMind-V1.2-Qwen2.5-C-7B-I', 'SiliconMind-V1.2-Qwen3-4B-T-2507',
              'OriGen']
    GIDS = [4, 5, 6, 7]
    dev = 'npu'

    fa = json.load(open(f'{OUT}/fold_assignments.json'))
    t2k = {}
    for k in range(5):
        for t in fa['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    tasks_all = {t['task_id']: t.get('prompt', '') for t in
                 json.load(open(f'{STEP2}/tasks_step2.json'))}

    TEMPLATE = (
        "You will be provided with the design specification and a candidate RTL implementation.\n"
        "Your task is to determine whether the candidate correctly implements the specification\n"
        "under the supplied functional behavior.\n\n"
        "Output:\n0 = Incorrect implementation\n1 = Correct implementation\n\n"
        "Specification:\n{spec}\n\nCandidate RTL:\n{rtl}\n\nAnswer:")

    jm = 'models/Qwen2.5-Coder-3B-Instruct'
    tok = AutoTokenizer.from_pretrained(jm, trust_remote_code=True)
    tok.padding_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        jm, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(dev).eval()
    lm_head = model.get_output_embeddings()

    rows = []
    done = set()
    _p47 = f'{OUT}/predictions/slm_judge_scores_gids47.csv'
    if os.path.exists(_p47):
        for r in csv.DictReader(open(_p47)):
            done.add(r['model'])
        print(f'resuming: completed models skipped: {sorted(done)}', flush=True)
    for gen in MODELS:
        if gen in done:
            print(f'[skip] {gen} already complete', flush=True)
            continue
        cands = {}
        for line in open(f'{STEP2}/{gen}/gen.jsonl'):
            r = json.loads(line)
            g = int(r['generation_id'])
            if g in GIDS:
                cands[(r['task'], g)] = (r.get('rtl') or r.get('raw_output', ''))
        need = sorted({(t, g) for (t, g) in cands if t in tasks_all})
        prompts, keys = [], []
        for (t, g) in need:
            msg = TEMPLATE.format(spec=tasks_all[t][:6000], rtl=cands[(t, g)][:6000])
            try:
                p = tok.apply_chat_template([{'role': 'user', 'content': msg}],
                                            tokenize=False, add_generation_prompt=True)
            except Exception:
                p = msg + "\nAnswer (0/1):"
            prompts.append(p)
            keys.append((t, g))
        BATCH = 8
        fmt_fail = 0
        with torch.no_grad():
            for i in range(0, len(prompts), BATCH):
                chunk = prompts[i:i + BATCH]
                enc = tok(chunk, return_tensors='pt', padding=True,
                          truncation=True, max_length=3072)
                ntok = enc['attention_mask'].sum(1).tolist()
                inp = {k: v.to(dev) for k, v in enc.items()}
                hs = model.model(input_ids=inp['input_ids'],
                                 attention_mask=inp['attention_mask']
                                 ).last_hidden_state[:, -1, :]
                logits = lm_head(hs).float()
                logp = torch.log_softmax(logits, -1)[0] if False else \
                    torch.log_softmax(logits, dim=-1)
                topv, topi = torch.topk(logp, 20, dim=-1)
                topv = topv.cpu().numpy(); topi = topi.cpu().numpy()
                for j in range(len(chunk)):
                    p0 = p1 = 0.0
                    for lp, tid in zip(topv[j], topi[j]):
                        s = (tok.decode([int(tid)]) or '').strip()
                        if s == '0':
                            p0 = float(np.exp(lp))
                        elif s == '1':
                            p1 = float(np.exp(lp))
                    if p0 + p1 > 0:
                        score, ok = p1 / (p0 + p1), True
                    else:
                        score, ok = None, False
                        fmt_fail += 1
                    t, g = keys[i + j]
                    k = t2k.get(t)
                    rows.append(dict(model=gen,
                                     fold=f'fold{k}' if k is not None else 'OOD',
                                     task=t, gid=g,
                                     score='' if score is None else round(score, 6),
                                     fmt_ok=ok, in_tokens=int(ntok[j])))
                if (i // BATCH) % 5 == 0:
                    print(f'  [{gen}] {i + len(chunk)}/{len(prompts)}', flush=True)
        # 逐模型增量落盘
        new = [r for r in rows if r['model'] == gen]
        with open(_p47, 'a', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=['model', 'fold', 'task', 'gid', 'score',
                                               'fmt_ok', 'in_tokens'])
            if fo.tell() == 0:
                w.writeheader()
            w.writerows(new)
        print(f'[{gen}|judge3b gids47] rows={len(need)} fmt_fail={fmt_fail}', flush=True)
        rows = [r for r in rows if r['model'] != gen]

    print(f'TOTAL rows={len(rows)}', flush=True)

if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '0')
