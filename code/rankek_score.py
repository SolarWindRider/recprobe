#!/usr/bin/env python3
"""RankEF 推理打分 v2: outer-test + inner-validation 双覆盖, 记录输入 token 数。

用法: venv_rankek/bin/python rankek_score.py <outer_fold>
输出:
  predictions/rankek_scores_fold{k}.csv       测试任务(列: model,task,gid,score,in_tokens)
  predictions/rankek_scores_val_fold{k}.csv   inner-val 任务(τ_R 选择用)
  results/rankek_in_tokens_fold{k}.json       每 (model,task,gid) 的 scorer 输入 token
"""
import json, os, sys, collections
import numpy as np
import torch, torch_npu
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = f'{HERE}/..'
STEP2 = './experiments/step2'
MAX_IN = 512
BS_MAIN = 4


def main(fold):
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    test_tasks = set(folds['folds'][f'fold{fold}']['test_tasks'])
    from rankek_train import build_ds
    tr, va = build_ds(fold, None)          # 与训练脚本完全同一 inner 划分
    val_tasks = sorted({r['task'] for r in va})
    dev = 'npu'
    tok = AutoTokenizer.from_pretrained('models/codet5p-770m', trust_remote_code=True)
    from transformers import AutoModelForSeq2SeqLM
    model = AutoModelForSeq2SeqLM.from_pretrained(
        f'{OUT}/checkpoints/rankek_fold{fold}', trust_remote_code=True,
        torch_dtype=torch.float32)
    ckpt = torch.load(f'{OUT}/checkpoints/rankek_fold{fold}/cls_best.pt',
                      map_location='cpu')
    cls_head = nn.Linear(model.config.d_model, 2)
    cls_head.load_state_dict(ckpt['cls_head'])
    model = model.npu().eval(); cls_head = cls_head.to(dev).eval()
    print('loaded ckpt step', ckpt.get('step'), 'val_f1', ckpt.get('val_f1'), flush=True)

    from track_a_common import MODELS, tasks_dict
    specs = tasks_dict()

    def run(task_set, tag):
        rows, in_tok = [], {}
        for m in MODELS:
            base = f'{STEP2}/{m}'
            genmap = collections.defaultdict(dict)
            for line in open(f'{base}/gen.jsonl'):
                r = json.loads(line)
                if int(r['generation_id']) < BS_MAIN and r['task'] in task_set:
                    genmap[r['task']][int(r['generation_id'])] = r
            evmap = collections.defaultdict(dict)
            for line in open(f'{base}/eval.jsonl'):
                r = json.loads(line)
                if int(r['generation_id']) < BS_MAIN and r['task'] in task_set:
                    evmap[r['task']][int(r['generation_id'])] = r
            print(m, tag, 'tasks:', len(genmap), flush=True)
            for t in sorted(genmap):
                gids = sorted(g for g in genmap[t] if g < BS_MAIN)
                spec = specs.get(t, {}).get('prompt', '')[:3000]
                texts = [f"Task: {spec}\nCode:\n{(evmap[t].get(g, {}).get('rtl') or genmap[t][g].get('raw_output') or '')[:4000]}"
                         for g in gids]
                enc = tok(texts, truncation=True, max_length=MAX_IN,
                          padding='max_length', return_tensors='pt')
                in_len = enc['attention_mask'].sum(1).tolist()
                enc = {k: v.to(dev) for k, v in enc.items()}
                dec_start = torch.full((len(gids), 1), 0, device=dev)
                with torch.no_grad():
                    out = model(input_ids=enc['input_ids'],
                                attention_mask=enc['attention_mask'],
                                decoder_input_ids=dec_start)
                    h = out.encoder_last_hidden_state.mean(1)
                    logits = cls_head(h).cpu()
                probs = torch.softmax(logits, 1)[:, 1]
                for (g, r), p, il in zip(enumerate(zip(gids, [None]*len(gids))), probs, in_len):
                    pass
                for j, g in enumerate(gids):
                    rows.append(dict(model=m, task=t, gid=g,
                                     score=round(float(probs[j]), 6)))
                    in_tok[(m, t, g)] = int(in_len[j])
        return rows, in_tok

    test_rows, in_test = run(test_tasks, 'test')
    val_rows, in_val = run(val_tasks, 'val')

    import csv
    def dump(rows_, in_map, path):
        with open(path, 'w', newline='') as fo:
            w = csv.DictWriter(fo, fieldnames=['model', 'task', 'gid', 'score', 'in_tokens'])
            w.writeheader()
            for r in rows_:
                r['in_tokens'] = in_map.get((r['model'], r['task'], r['gid']), 0)
                w.writerow(r)
    dump(test_rows, in_test, f'{OUT}/predictions/rankek_scores_fold{fold}.csv')
    dump(val_rows, in_val, f'{OUT}/predictions/rankek_scores_val_fold{fold}.csv')
    mean_in = float(np.mean(list(in_test.values()))) if in_test else 0.0
    json.dump({'mean_input_tokens_per_forward': round(mean_in, 1),
               'n_test': len(test_rows), 'n_val': len(val_rows)},
              open(f'{OUT}/results/rankek_in_tokens_fold{fold}.json', 'w'), indent=1)
    print(f'fold{fold}: test={len(test_rows)} val={len(val_rows)} '
          f'mean_in_tokens={mean_in:.1f}', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
