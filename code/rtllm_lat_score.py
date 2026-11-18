#!/usr/bin/env python3
"""RTLLM LAT-L27 排序(独立实现,零依赖)。"""
import json, os, sys, hashlib, collections
import numpy as np

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']
BS = 4

def main():
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(folds['ood_rtllm_tasks'])
    tasks_meta = {t['task_id']: t for t in json.load(open(f'{STEP2}/tasks_step2.json'))}
    t2g = {t['task_id']: t['design_group'] for t in json.load(open(f'{STEP2}/tasks_step2.json'))}
    t2k = {}
    for k in range(5):
        for t in folds['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k

    results = {}
    for m in MODELS:
        hs = f'{STEP2}/hidden_stream/{m}'
        # 加载 layer-27 残差 hash 映射
        ev = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            ev[r['task']][int(r['generation_id'])] = bool(r.get('simulation_pass'))
        # 构建残差缓存: 每 (task, gid) → (hash, pass)
        hash_map = {}
        for t in ev:
            for g in ev[t]:
                h = hashlib.md5((m + t + str(g)).encode()).hexdigest()[:16]
                fp_ = f'{hs}/{h}.npz'
                if os.path.exists(fp_):
                    hash_map[(t, g)] = h

        def load_res(t, g):
            h = hash_map.get((t, g))
            if h is None:
                return None
            fp_ = f'{hs}/{h}.npz'
            if not os.path.exists(fp_):
                return None
            zz = np.load(fp_, allow_pickle=False)
            man = json.loads(str(zz['pos_manifest']))
            tk = {tg: k2 for k2, tags in man.items() for tg in tags}
            if 'w100' in tk:
                return zz[f"{tk['w100']}@L27"].astype(np.float64)
            return None

        for k in range(5):
            # PCA 方向: 从 fold k 训练折 pass/fail 对中提取
            train_tasks = [t for t in ev if t2k.get(t) != k and t not in rtllm]
            rng = np.random.RandomState(42)
            pairs = []
            for t in train_tasks:
                gmap = ev[t]
                p_idx = [g for g in sorted(gmap) if gmap[g]]
                f_idx = [g for g in sorted(gmap) if not gmap[g]]
                if p_idx and f_idx:
                    gp = rng.choice(p_idx)
                    gf = rng.choice(f_idx)
                    vp = load_res(t, gp)
                    vf = load_res(t, gf)
                    if vp is not None and vf is not None:
                        pairs.append(vp - vf)
            if len(pairs) < 20:
                continue
            D = np.array(pairs)
            Dc = D - D.mean(0, keepdims=True)
            _, _, Vt = np.linalg.svd(Dc, full_matrices=False)
            v = Vt[0]
            if float((D @ v).mean()) < 0:
                v = -v

            # RTLLM 投影(前 4 候选)
            rtllm_set = set(rtllm)
            for t in sorted(rtllm_set):
                scores = []
                for g in range(BS):
                    r = load_res(t, g)
                    scores.append(float(r @ v) if r is not None else 0.0)
                results[(m, k, t)] = scores
        print(f'{m}: {sum(1 for key in results if key[0]==m)} RTLLM (model,fold) pools', flush=True)

    # 保存
    out = {}
    for (m, k, t), scores in results.items():
        out[f'{m}#{k}#{t}'] = scores
    np.savez_compressed(f'{OUT}/results/rtllm_lat_scores.npz', **out)
    print(f'Total: {len(results)} RTLLM pools → results/rtllm_lat_scores.npz', flush=True)

if __name__ == '__main__':
    main()
