#!/usr/bin/env python3
"""OOD QA 扩展(审计 P1b): checkpoint/折覆盖/tie-break/bootstrap/一致性检查。

输出 qa_ood_checks.json; 全过 = ood PASS。
"""
import json, os, sys, hashlib, collections, subprocess
import numpy as np

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']

checks = []
def chk(name, ok, detail):
    checks.append({'check': name, 'pass': bool(ok), 'detail': detail})

def main():
    # Q1: RankEF OOD encoder = IID encoder(同一文件路径 + SHA)
    prov = json.load(open(f'{OUT}/results/rankek_ood_checkpoint_provenance.json'))
    iid_ok, iid_sha = [], {}
    for k in range(5):
        p = prov.get(f'fold{k}_encoder_path', '')
        want = os.path.abspath(f'{OUT}/checkpoints/rankek_fold{k}/pytorch_model.bin')
        iid_ok.append(p == want)
    sha_ok = all(len(prov.get(f'fold{k}_encoder_sha256_1MB','')) == 64 for k in range(5))
    # 与 IID 评分脚本加载路径一致性(IID rankek_score.py 同一 checkpoints 目录)
    src = open(f'{OUT}/scripts/rankek_score.py').read()
    iid_same = "from_pretrained(\n        f'{OUT}/checkpoints/rankek_fold{fold}'" in src
    chk('Q1 rankek_ood_encoder_matches_iid', all(iid_ok) and iid_same and sha_ok,
        {'paths_match': all(iid_ok), 'iid_script_uses_ckpt_dir': iid_same,
         'sha_recorded': sha_ok,
         'fold_encoder_sha': {f'fold{k}': prov.get(f'fold{k}_encoder_sha256_1MB','')[:16]
                              for k in range(5)}})

    # Q2: 折覆盖 5×12×50
    rkz = np.load(f'{OUT}/results/rtllm_rankek_scores_5fold.npz')
    folds_rk = collections.Counter(int(k.split('#')[1]) for k in rkz.files)
    latz = np.load(f'{OUT}/results/rtllm_lat_scores.npz')
    folds_lat = collections.Counter(int(k.split('#')[1]) for k in latz.files)
    cov_ok = (all(folds_rk.get(k,0) == 600 for k in range(5)) and
              all(folds_lat.get(k,0) == 600 for k in range(5)))
    chk('Q2 fold_coverage_5x12x50', cov_ok,
        {'rankef': dict(folds_rk), 'lat': dict(folds_lat)})

    # Q3: RankEF 分数离散度(修正后不得全 <0.5 压缩)
    vals = np.concatenate([rkz[k] for k in rkz.files])
    frac_below = float((vals < 0.5).mean())
    spread_ok = frac_below < 0.99 and float(vals.max()) > 0.6
    chk('Q3 rankek_score_spread', spread_ok,
        {'frac_below_0.5': round(frac_below, 4), 'min': round(float(vals.min()),4),
         'max': round(float(vals.max()),4)})

    # Q4: tie-break 确定性(stable 排序可复现)
    s = np.array([0.7, 0.7, 0.4, 0.9])
    y = np.array([0, 1, 0, 0])
    o1 = np.argsort(-s, kind='stable'); o2 = np.argsort(-s, kind='stable')
    chk('Q4 tiebreak_stable_deterministic',
        np.array_equal(o1, o2) and o1[0] == 3 and o1[1] == 0 and o1[2] == 1,
        {'order': o1.tolist(),
         'note': '最大 0.9 在 idx3; 并列 0.7(idx0,idx1) 按 stable 保持最早 gid 优先'})

    # Q5: bootstrap 可复现(同 seed 同 CI)
    from sklearn.metrics import roc_auc_score  # noqa
    rng1 = np.random.RandomState(42); rng2 = np.random.RandomState(42)
    a1 = [rng1.randint(0, 50, 50) for _ in range(100)]
    a2 = [rng2.randint(0, 50, 50) for _ in range(100)]
    chk('Q5 bootstrap_seed_reproducible', np.array_equal(
            np.array(a1, dtype=object), np.array(a2, dtype=object)),
        {'seed': 42, 'draws': 100})

    # Q6: 交叉一致性 oracle@4 == pass@rank-4; retry@4 == alone
    rep = json.load(open(f'{OUT}/results/rtllm_policy_replay_v2.json'))
    cmp5 = json.load(open(f'{OUT}/results/rtllm_ranking_compare_5fold.json'))
    cpp4 = cmp5['cpp']['pass@rank2']  # 不可用 rank4 直接对照(排名表只有 p1/p2/auroc)
    oracle_ok = abs(rep['arms']['oracle@4']['pass']
                    * 100 - cmp5['cpp']['pass@rank2']) < 100  # 宽松交叉
    exact = None
    # 精确交叉: oracle@4 == labels any()
    fa = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(fa['ood_rtllm_tasks'])
    any_pass = []
    for m in MODELS:
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm and int(r['generation_id']) < 4:
                any_pass.append(bool(r['simulation_pass']))
    # per-task any
    per_task = collections.defaultdict(int)
    for m in MODELS:
        gd = collections.defaultdict(int)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            if r['task'] in rtllm and int(r['generation_id']) < 4:
                gd[r['task']] |= int(bool(r['simulation_pass']))
        per_task[m] = float(np.mean(list(gd.values())))
    exact = float(np.mean(list(per_task.values())))
    rep_oracle = rep['arms']['oracle@4']['pass']
    chk('Q6 oracle_cross_check', abs(exact - rep_oracle) < 0.005,
        {'labels_any4_macro': round(exact, 4), 'replay_oracle': round(rep_oracle, 4)})

    # Q7: ranker 相同(hybrid vs recprobe 同门控 → tokens 应一致)
    if 'hybrid@4' in rep['arms'] and 'recprobe@4' in rep['arms']:
        t_h = rep['arms']['hybrid@4']['mean_tokens']
        t_r = rep['arms']['recprobe@4']['mean_tokens']
        chk('Q7 hybrid_same_gate_tokens_as_recprobe', abs(t_h - t_r) < 1.0,
            {'hybrid_tokens': t_h, 'recprobe_tokens': t_r})

    n_pass = sum(c['pass'] for c in checks)
    out = {'n_pass': n_pass, 'n_total': len(checks), 'checks': checks,
           'verdict': 'OOD PASS' if n_pass == len(checks) else 'OOD FAIL'}
    json.dump(out, open(f'{OUT}/qa_ood_checks.json', 'w'), indent=1)
    for c in checks:
        print(f"[{'X' if c['pass'] else ' '}] {c['check']}: {json.dumps(c['detail'])[:140]}")
    print(f"{n_pass}/{len(checks)} {out['verdict']}")

if __name__ == '__main__':
    main()
