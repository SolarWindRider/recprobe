#!/usr/bin/env python3
"""VerilogEval–RTLLM 跨基准任务重叠审计。

三层检查(协议 §7):
  1. exact duplicate: 归一化 prompt 全等(小写+空白折叠)
  2. lexical near-duplicate: 3-gram Jaccard >= 0.5(报告 >=0.3 的 top 对)
  3. interface overlap: module_name 同名 且 端口签名(port names+方向)一致
输出 results/ood_overlap_audit.json + ood_overlap_pairs.csv
"""
import json, re, collections, csv
from itertools import product

STEP2 = 'experiments/step2'
OUT = 'experiments/recprobe_group5fold'

def norm(s):
    return re.sub(r'\s+', ' ', (s or '').lower()).strip()

def ngrams(s, n=3):
    w = norm(s).split()
    return set(tuple(w[i:i+n]) for i in range(len(w)-n+1)) if len(w) >= n else {tuple(w)} if w else set()

def ports(prompt):
    """从 prompt 抽模块端口声明行: input/output/inout 后的信号名集合"""
    ps = set()
    for m in re.finditer(r'\b(input|output|inout)\s+(?:\[(\d+):(\d+)\]\s*)?([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)', prompt or ''):
        for name in re.split(r'\s*,\s*', m.group(4)):
            ps.add((m.group(1), name))
    return ps

def main():
    ts = json.load(open(f'{STEP2}/tasks_step2.json'))
    fa = json.load(open(f'{OUT}/fold_assignments.json'))
    rtllm = set(fa['ood_rtllm_tasks'])
    ve = [t for t in ts if t['task_id'] not in rtllm]
    rt = [t for t in ts if t['task_id'] in rtllm]
    print(f'VE tasks: {len(ve)}  RTLLM tasks: {len(rt)}')

    exact, pairs = [], []
    ve_ng = [(t['task_id'], ngrams(t['prompt']), norm(t['prompt']),
              ports(t['prompt']), t['module_name']) for t in ve]
    for r in rt:
        rn, rng, rports = norm(r['prompt']), ngrams(r['prompt']), ports(r['prompt'])
        for vid, vng, vn, vp, vmn in ve_ng:
            j = len(rng & vng) / max(len(rng | vng), 1)
            exact_port = (rports == vp and rports)
            if rn == vn:
                exact.append({'rtllm': r['task_id'], 've': vid, 'kind': 'exact_prompt'})
            if j >= 0.3:
                pairs.append({'rtllm': r['task_id'], 've': vid,
                              'jaccard3': round(j, 3),
                              'module_same': int(r['module_name'] == vmn),
                              'port_exact': int(exact_port)})
    pairs.sort(key=lambda x: -x['jaccard3'])
    # interface overlap: module_name 同名
    vmn = collections.defaultdict(list)
    for t in ve:
        vmn[t['module_name']].append(t['task_id'])
    iface = [{'rtllm': r['task_id'], 've': v, 'module': r['module_name']}
             for r in rt for v in vmn.get(r['module_name'], [])]

    summary = {
        'n_ve': len(ve), 'n_rtllm': len(rt),
        'exact_prompt_dupes': len(exact),
        'near_dup_jaccard_ge_05': sum(1 for p in pairs if p['jaccard3'] >= 0.5),
        'near_dup_jaccard_ge_03': sum(1 for p in pairs if p['jaccard3'] >= 0.3),
        'module_name_collisions': len(iface),
        'port_exact_matches_among_collisions': sum(
            1 for p in pairs if p['module_same'] and p['port_exact']),
        'verdict': ('FAIL: near-duplicates present' if
                    (exact or any(p['jaccard3'] >= 0.5 for p in pairs))
                    else 'PASS: no exact dup, no near-dup >= 0.5'),
        'top_pairs': pairs[:15]}
    json.dump({'summary': summary, 'exact': exact, 'interface': iface},
              open(f'{OUT}/results/ood_overlap_audit.json', 'w'), indent=1)
    with open(f'{OUT}/results/ood_overlap_pairs.csv', 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=['rtllm','ve','jaccard3','module_same','port_exact'])
        w.writeheader(); w.writerows(pairs)
    print(json.dumps(summary, indent=1))

if __name__ == '__main__':
    main()
