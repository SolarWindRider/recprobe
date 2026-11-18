#!/usr/bin/env python3
"""LAT-L27 gid=4..7 补投影(纯 CPU): 复用 rtllm_lat_score.py 的同款方向拟合
(相同 seed/遍历顺序 → 方向与 gid0-3 逐位一致), 对保存的残差做点积。
输出 predictions/lat_scores_gids47_iid.csv / _ood.csv(列: model,fold,task,gid,score)
"""
import json, os, sys, csv, hashlib, collections
import numpy as np

OUT = 'experiments/recprobe_group5fold'
STEP2 = 'experiments/step2'
MODELS = ['CodeV-QW-7B','CodeV-DS-6.7B','VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B','CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1','DeepSeek-Coder-6.7B',
          'VeriSeek','SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I','SiliconMind-V1.2-Qwen3-4B-T-2507','OriGen']
GIDS = [4, 5, 6, 7]
BS = 4  # 方向拟合沿用原脚本的 gid 池约定

def load_res(hs, m, t, g, hmap_cache):
    h = hmap_cache.get((t, g))
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

def main():
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(folds['ood_rtllm_tasks'])
    t2k = {}
    for k in range(5):
        for t in folds['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    rows_iid, rows_ood = [], []
    for m in MODELS:
        hs = f'{STEP2}/hidden_stream/{m}'
        ev = collections.defaultdict(dict)
        for line in open(f'{STEP2}/{m}/eval.jsonl'):
            r = json.loads(line)
            ev[r['task']][int(r['generation_id'])] = bool(r.get('simulation_pass'))
        hmap = {}
        for t in ev:
            for g in ev[t]:
                hmap[(t, g)] = hashlib.md5((m + t + str(g)).encode()).hexdigest()[:16]
        for k in range(5):
            # 与 rtllm_lat_score.py 完全相同的方向拟合
            train_tasks = [t for t in ev if t2k.get(t) != k and t not in rtllm]
            rng = np.random.RandomState(42)
            pairs = []
            for t in train_tasks:
                gmap = ev[t]
                p_idx = [g for g in sorted(gmap) if gmap[g]]
                f_idx = [g for g in sorted(gmap) if not gmap[g]]
                if p_idx and f_idx:
                    gp = rng.choice(p_idx); gf = rng.choice(f_idx)
                    vp = load_res(hs, m, t, gp, hmap)
                    vf = load_res(hs, m, t, gf, hmap)
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
            # gid 4..7 投影
            for t in sorted(set(ev) & rtllm):
                r = load_res(hs, m, t, 4, hmap)  # warm check
                vals = {}
                for g in GIDS:
                    res = load_res(hs, m, t, g, hmap)
                    vals[g] = float(res @ v) if res is not None else 0.0
                rows_ood.append((m, f'fold{k}', t, vals))
            for t in sorted(t2k):
                if t2k[t] != k or t not in ev:
                    continue
                vals = {}
                for g in GIDS:
                    res = load_res(hs, m, t, g, hmap)
                    vals[g] = float(res @ v) if res is not None else 0.0
                rows_iid.append((m, f'fold{k}', t, vals))
        print(f'{m}: iid={len(rows_iid)} ood={len(rows_ood)}', flush=True)
    for path, rows in ((f'{OUT}/predictions/lat_scores_gids47_iid.csv', rows_iid),
                       (f'{OUT}/predictions/lat_scores_gids47_ood.csv', rows_ood)):
        with open(path, 'w', newline='') as fo:
            w = csv.writer(fo); w.writerow(['model', 'fold', 'task', 'gid', 'score'])
            for m, fk, t, vals in rows:
                for g in GIDS:
                    w.writerow([m, fk, t, g, f'{vals[g]:.6f}'])
    print(f'TOTAL iid_rows={len(rows_iid)} ood_rows={len(rows_ood)}')

if __name__ == '__main__':
    main()
