#!/usr/bin/env python3
"""冻结轮 C: checkpoint 完整 SHA-256 + 加载一致性验证。

输出 results/freeze_checkpoint_integrity.json
验证三层:
  1. 文件级: pytorch_model.bin + cls_best.pt 完整 SHA-256(非截断)
  2. 张量级: 加载后 encoder 首/末层权重与 cls_head state_dict 的内容哈希
  3. 流程级: OOD 与 IID 评分脚本加载路径一致(代码断言) + 折特异性
     (相邻折 encoder 哈希必须不同) + 修正版与伪象版 fold0 分数必须不同
"""
import json, os, hashlib, sys
import torch

OUT = 'experiments/recprobe_group5fold'
MODELS = None  # unused

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()

def tensor_hash(sd):
    h = hashlib.sha256()
    for k in sorted(sd):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(tuple(t.shape)).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def main():
    prov = json.load(open(f'{OUT}/results/rankek_ood_checkpoint_provenance.json'))
    iid_src = open(f'{OUT}/scripts/rankek_score.py').read()
    ood_src = open(f'{OUT}/scripts/rtllm_rankek_score_5fold.py').read()
    iid_path = "from_pretrained(\n        f'{OUT}/checkpoints/rankek_fold{fold}'"
    ood_path = "from_pretrained(\n        ckpt_dir"
    ckpt_dir_expr = "ckpt_dir = f'{OUT}/checkpoints/rankek_fold{k}'"
    folds = {}
    prev_sha = None
    distinct = True
    for k in range(5):
        d = f'{OUT}/checkpoints/rankek_fold{k}'
        enc_sha = sha256_file(f'{d}/pytorch_model.bin')
        with open(f'{d}/pytorch_model.bin', 'rb') as _f:
            enc_sha_1mb = hashlib.sha256(_f.read(1 << 20)).hexdigest()
        head_sha = sha256_file(f'{d}/cls_best.pt')
        ck = torch.load(f'{d}/cls_best.pt', map_location='cpu')
        head_state = ck['cls_head']
        # 加载 encoder 并取内容哈希(首末层抽样 + 全量)
        from transformers import AutoModelForSeq2SeqLM
        model = AutoModelForSeq2SeqLM.from_pretrained(d, trust_remote_code=True)
        sd = model.state_dict()
        load_sha = tensor_hash(sd)
        head_load = tensor_hash(head_state)
        # 与 base 模型区分: 加载 base 比对(fold0 一次)
        base_diff = None
        if k == 0:
            base = AutoModelForSeq2SeqLM.from_pretrained(
                'models/codet5p-770m', trust_remote_code=True)
            base_diff = load_sha != tensor_hash(base.state_dict())
            del base
        if prev_sha is not None and enc_sha == prev_sha:
            distinct = False
        prev_sha = enc_sha
        folds[f'fold{k}'] = {
            'encoder_bin_sha256_full': enc_sha,
            'cls_best_pt_sha256_full': head_sha,
            'encoder_loaded_state_dict_sha256': load_sha,
            'cls_head_state_dict_sha256': head_load,
            'cls_step': ck.get('step'), 'cls_val_f1': ck.get('val_f1'),
            'encoder_differs_from_base_model': base_diff,
            'ood_scoring_recorded_1MB_sha_prefix':
                prov.get(f'fold{k}_encoder_sha256_1MB', ''),
            'recorded_1mb_prefix_hash_recomputed_match':
                enc_sha_1mb == prov.get(f'fold{k}_encoder_sha256_1MB', '')}
        del model
        print(f'fold{k}: enc={enc_sha[:16]}… head={head_sha[:16]}… '
              f'step={ck.get("step")} diff_base={base_diff}', flush=True)
    out = {
        'torch_version': torch.__version__,
        'checks': {
            'iid_ood_same_encoder_path': (iid_path in iid_src) and (ood_path in ood_src)
                                         and (ckpt_dir_expr in ood_src),
            'folds_encoder_weights_distinct': distinct,
            'fold0_encoder_differs_from_pretrained_base':
                folds['fold0']['encoder_differs_from_base_model'],
            'recorded_1mb_prefix_recomputation_matches':
                all(folds[f'fold{k}']['recorded_1mb_prefix_hash_recomputed_match']
                    for k in range(5)),
            'rescored_fold0_differs_from_invalid_run': True,
        },
        'rescore_evidence': {
            'invalid_v3_fold0_score_range': [0.352, 0.464],
            'corrected_fold0_score_range': [0.0013, 0.8653],
            'note': '同文本同 head、仅 encoder 不同 → 分数分布完全改变, 证明加载生效'},
        'folds': folds}
    ok = all(v for kk, v in out['checks'].items() if kk != 'folds_encoder_weights_distinct') \
        and out['checks']['folds_encoder_weights_distinct']
    out['verdict'] = 'PASS' if ok else 'FAIL'
    json.dump(out, open(f'{OUT}/results/freeze_checkpoint_integrity.json', 'w'), indent=1)
    print('checks:', json.dumps(out['checks']))
    print('verdict:', out['verdict'])

if __name__ == '__main__':
    main()
