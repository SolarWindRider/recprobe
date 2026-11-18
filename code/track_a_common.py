#!/usr/bin/env python3
"""Track A 共享常量与加载器(只读原始 step2 数据)。

行序规范:全部 7 个模型共用同一行序 (task, gid) 升序,362*32=11584 行/模型。
"""
import glob, hashlib, json, os
from collections import defaultdict
import numpy as np

STEP2 = './experiments/step2'
OUT = './experiments/recprobe_group5fold'
MODELS = ['CodeV-QW-7B', 'CodeV-DS-6.7B', 'VeriCoder_Qwen14B',
          'CodeV-R1-Distill-Qwen-7B', 'CodeV-R1-RL-Qwen-7B',
          'RTLCoder-Deepseek-v1.1', 'DeepSeek-Coder-6.7B',
          # Track A 扩展 (2026-10-20): 5 个新模型
          'VeriSeek', 'SiliconMind-V1.2-Qwen3-8B',
          'SiliconMind-V1.2-Qwen2.5-C-7B-I', 'SiliconMind-V1.2-Qwen3-4B-T-2507',
          'OriGen']
LEGACY7 = ['CodeV-QW-7B', 'CodeV-DS-6.7B', 'VeriCoder_Qwen14B',
           'CodeV-R1-Distill-Qwen-7B', 'CodeV-R1-RL-Qwen-7B',
           'RTLCoder-Deepseek-v1.1', 'DeepSeek-Coder-6.7B']
LAYER = 27
N_PER_MODEL = 362 * 32

def tasks_dict():
    return {t['task_id']: t for t in json.load(open(f'{STEP2}/tasks_step2.json'))}

def canonical_rows():
    """行序:(task_id, gid) 升序。返回 [(task, gid)],共 11584。"""
    tasks = tasks_dict()
    rows = []
    for t in sorted(tasks):
        for g in range(32):
            rows.append((t, g))
    assert len(rows) == N_PER_MODEL
    return rows

def load_model_matrix(m, rows=None):
    """返回 dict: task,gid,group,passed,rec,gen_len(np数组) + X(11584,4096) float32。

    passed 来自 eval.jsonl simulation_pass(与原实现一致);
    recoverable = 任务级任一样本通过;gen_len 来自 labels.jsonl。
    """
    tasks = tasks_dict()
    if rows is None:
        rows = canonical_rows()
    ev = {}
    for r in map(json.loads, open(f'{STEP2}/{m}/eval.jsonl')):
        ev[(r['task'], r['generation_id'])] = bool(r['simulation_pass'])
    glen = {}
    for r in map(json.loads, open(f'{STEP2}/{m}/labels.jsonl')):
        glen[(r['task'], r['generation_id'])] = int(r['gen_len'])
    task_pass = defaultdict(bool)
    for (t, g), p in ev.items():
        task_pass[t] |= p
    X = None  # 维度按模型首个可用 npz 推断(4096/5120/3584...)
    meta = {'task': [], 'gid': [], 'group': [], 'passed': [], 'rec': [], 'gen_len': []}
    n_missing = 0
    for i, (t, g) in enumerate(rows):
        key = (t, g)
        if key not in ev or key not in glen:
            raise RuntimeError(f'{m}: 缺 eval/labels: {key}')
        h = hashlib.md5((m + t + str(g)).encode()).hexdigest()[:16]
        zf = f'{STEP2}/hidden_stream/{m}/{h}.npz'
        if not os.path.exists(zf):
            n_missing += 1
            if X is None:
                raise RuntimeError(f'{m}: 首行 {key} 缺 npz,无法推断维度')
            meta['task'].append(t); meta['gid'].append(g)
            meta['group'].append(tasks[t]['design_group'])
            meta['passed'].append(bool(ev[key])); meta['rec'].append(bool(task_pass[t]))
            meta['gen_len'].append(glen[key])
            continue
        z = np.load(zf, allow_pickle=False)
        man = json.loads(str(z['pos_manifest']))
        tk = {tg: k for k, tags in man.items() for tg in tags}
        if 'w100' not in tk:
            raise RuntimeError(f'{m}: {key} 无 w100 tag')
        v = z[f"{tk['w100']}@L{LAYER}"].astype(np.float32)
        if X is None:
            X = np.empty((len(rows), v.shape[0]), dtype=np.float32)
        X[i] = v
        meta['task'].append(t); meta['gid'].append(g)
        meta['group'].append(tasks[t]['design_group'])
        meta['passed'].append(bool(ev[key])); meta['rec'].append(bool(task_pass[t]))
        meta['gen_len'].append(glen[key])
    meta = {k: np.array(v) for k, v in meta.items()}
    return X, meta, n_missing

def load_texts(m, rows=None):
    """(task,gid) -> output_text。仅前 8 个 gid 被回放需要,这里全量加载一次。"""
    if rows is None:
        rows = canonical_rows()
    need = {t: set() for t, _ in rows}
    for t, g in rows:
        if g < 8:
            need[t].add(g)
    texts = {}
    for jf in sorted(glob.glob(f'{STEP2}/{m}/samples/*.json')):
        if jf.endswith('.lp.npz'):
            continue
        base = os.path.basename(jf)[:-len('.json')]
        t, _, gs = base.rpartition('__')
        g = int(gs)
        if g < 8:
            texts[(t, g)] = json.load(open(jf)).get('output_text', '')
    return texts

def build_folds(seed=42, k=5):
    """156 个 VE design_group -> k 折。shuffle 后轮转发牌(与原 group_kfold 同风格)。"""
    tasks = tasks_dict()
    ve_groups = sorted({t['design_group'] for t in tasks.values()
                        if t['task_id'].startswith(('ve_', 'cc_'))})
    assert len(ve_groups) == 156
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ve_groups))
    folds = {f'fold{i}': {'groups': sorted(ve_groups[j] for j in order[i::k])}
             for i in range(k)}
    rtllm_groups = sorted({t['design_group'] for t in tasks.values()
                           if not t['task_id'].startswith(('ve_', 'cc_'))})
    assert len(rtllm_groups) == 50
    return folds, rtllm_groups

if __name__ == '__main__':
    folds, rtllm = build_folds()
    sizes = [len(f['groups']) for f in folds.values()]
    print('fold group sizes:', sizes, 'total', sum(sizes))
    print('rtllm groups:', len(rtllm))
