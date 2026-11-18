#!/usr/bin/env python3
"""RankEF hard-parameter-sharing 训练(fold0 smoke / 全量两用)。

忠实核心(ASE 2024): CodeT5+ encoder 双任务
  cls: encoder last-hidden mean-pool → Linear(→2), CE
  gen: decoder 生成 execution feedback 文本, LM CE
  L = 0.1*CE_cls + 0.9*CE_gen (λ=0.9)
推理只用 cls head。inner-val (20% 训练组) F1 选 checkpoint;outer-test 不接触。
用法: venv_rankek/bin/python rankek_train.py <outer_fold> <max_samples_per_epoch> <epochs>
"""
import json, os, sys, collections, time, random
import numpy as np
import torch, torch_npu
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = f'{HERE}/..'
N_INNER = 5
LAMBDA_GEN = 0.9
MAX_IN = 512
MAX_FB = 96


def build_ds(fold, limit):
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    st_path = f'{OUT}/results/rankek_quadruples.jsonl'
    # 训练组: 该折训练任务(VE) + 不含测试折
    t2k = {}
    for k in range(5):
        for t in folds['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    tr, va = [], []
    rng = random.Random(42)
    rows = []
    for line in open(st_path):
        r = json.loads(line)
        if r['outer_fold'] is None or r['outer_fold'] == fold:
            continue
        rows.append(r)
    rng.shuffle(rows)
    groups = sorted({r['design_group'] for r in rows})
    gperm = rng.sample(groups, len(groups))
    n_val_g = max(1, int(0.2 * len(gperm)))
    val_g = set(gperm[:n_val_g])
    for r in rows:
        (va if r['design_group'] in val_g else tr).append(r)
    if limit and len(tr) > limit:
        tr = rng.sample(tr, limit)
    return tr, va


def main(fold, limit, epochs):
    dev = 'npu'
    tok = AutoTokenizer.from_pretrained('models/codet5p-770m', trust_remote_code=True)
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        'models/codet5p-770m', trust_remote_code=True, torch_dtype=torch.float32)
    cls_head = nn.Linear(model.config.d_model, 2)
    model = model.npu(); cls_head = cls_head.to(dev)
    model.train(); cls_head.train()
    tr, va = build_ds(fold, limit)
    print(f'fold{fold}: train={len(tr)} val={len(va)}', flush=True)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(cls_head.parameters()), lr=2e-5)
    bce = nn.CrossEntropyLoss()

    def encode(batch_rows, with_fb):
        inp = tok([f"Task: {r['prompt'][:3000]}\nCode:\n{r['candidate_rtl'][:4000]}"
                   for r in batch_rows], truncation=True, max_length=MAX_IN,
                  padding='max_length', return_tensors='pt')
        dec = tok([r['feedback'][:MAX_FB] for r in batch_rows],
                  truncation=True, max_length=MAX_FB, padding='max_length',
                  return_tensors='pt')
        labels = dec['input_ids'].clone()
        labels[labels == tok.pad_token_id] = -100
        return inp, dec, labels

    def cls_logits(rows):
        inp, _, _ = encode(rows, False)
        b = inp['input_ids'].shape[0]
        dec_start = torch.full((b, 1), 0, device=dev)   # CodeT5+ config 无默认 decoder_start_token_id
        out = model(input_ids=inp['input_ids'].to(dev),
                    attention_mask=inp['attention_mask'].to(dev),
                    decoder_input_ids=dec_start)
        h = out.encoder_last_hidden_state.mean(1)
        return cls_head(h)

    def val_f1():
        model.eval(); cls_head.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for i in range(0, len(va), 8):
                chunk = va[i:i+8]
                lg = cls_logits(chunk).cpu()
                pred = lg.argmax(1)
                gold = torch.tensor([r['label'] for r in chunk])
                tp += int(((pred == 1) & (gold == 1)).sum())
                fp += int(((pred == 1) & (gold == 0)).sum())
                fn += int(((pred == 0) & (gold == 1)).sum())
        model.train(); cls_head.train()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        return f1, (tp, fp, fn)

    f10, _ = val_f1()
    print(f'fold{fold}: init val F1={f10:.4f}', flush=True)
    best_f1, best_step = -1, 0
    step = 0
    os.makedirs(f'{OUT}/checkpoints/rankek_fold{fold}', exist_ok=True)
    for ep in range(epochs):
        rng2 = random.Random(42 + ep)
        order = rng2.sample(range(len(tr)), len(tr))
        for i in range(0, len(order) - 11, 12):
            batch = [tr[j] for j in order[i:i+12]]
            inp, dec, labels = encode(batch, True)
            with torch.autocast(dev, dtype=torch.bfloat16):
                out = model(input_ids=inp['input_ids'].to(dev),
                            attention_mask=inp['attention_mask'].to(dev),
                            decoder_input_ids=dec['input_ids'][:, :-1].to(dev),
                            labels=labels[:, 1:].to(dev))
            lg = cls_logits(batch)
            gold = torch.tensor([r['label'] for r in batch]).to(dev)
            loss_cls = bce(lg, gold)
            loss_gen = out.loss
            loss = (1 - LAMBDA_GEN) * loss_cls + LAMBDA_GEN * loss_gen
            opt.zero_grad(); loss.backward(); opt.step()
            step += 1
            if step % 100 == 0:
                print(f'fold{fold} ep{ep} step{step}: loss={float(loss):.4f} '
                      f'(cls={float(loss_cls):.4f} gen={float(loss_gen):.4f})', flush=True)
            if step % 500 == 0:
                f1v, _ = val_f1()
                print(f'fold{fold} step{step}: val F1={f1v:.4f}', flush=True)
                if f1v > best_f1:
                    best_f1, best_step = f1v, step
                    torch.save({'cls_head': cls_head.state_dict(),
                                'step': step, 'val_f1': f1v},
                               f'{OUT}/checkpoints/rankek_fold{fold}/cls_best.pt')
                    model.save_pretrained(f'{OUT}/checkpoints/rankek_fold{fold}',
                                          safe_serialization=False)
    f1v, cnt = val_f1()
    print(f'fold{fold} FINAL: val F1={f1v:.4f} cnt={cnt} best={best_f1:.4f}@{best_step}',
          flush=True)
    json.dump({'fold': fold, 'best_val_f1': round(best_f1, 4), 'best_step': best_step,
               'final_val_f1': round(f1v, 4), 'n_train': len(tr), 'n_val': len(va),
               'lambda_gen': LAMBDA_GEN},
              open(f'{OUT}/results/rankek_train_fold{fold}.json', 'w'), indent=1)


if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 20000,
         int(sys.argv[3]) if len(sys.argv) > 3 else 2)
