#!/usr/bin/env python3
"""RankEF 训练四元组构造(方案 §5.3 数据条件;go/no-go 的数据交付)。

四元组 = (task_prompt, candidate_rtl, binary_label, execution_feedback_text)。
feedback 模板(按方案, 如实分档, 禁止统一 "incorrect"):
  Pass                    → "This code is correct."
  extraction failure      → "RTL extraction failed: no Verilog module could be
                             extracted from the model output."
  syntax failure          → iverilog parser message(前 400 字符, 真实诊断)
  compile/elaboration     → 编译器/链接诊断(前 400 字符, 含 .v:<行号> 时保留)
  functional mismatch     → "Functional simulation failed: the candidate did not
                             produce matching outputs against the provided
                             testbench. (no I/O mismatch details recorded)"
  timeout/runtime/unknown → 类别声明 + 日志摘要(若有)

输出: results/rankek_quadruples.jsonl(全 139,008 候选, 含 fold 拆分字段;
训练集 = 每外折训练折任务; 测试行仅标注 outer_fold 不入训练)。
"""
import json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from track_a_common import MODELS, OUT

STEP2 = './experiments/step2'
CLIP = 400


def build_feedback(r, rtl):
    """按方案模板从 eval.jsonl 字段构造 feedback 文本(不虚构)。"""
    ht = str(r.get('hallucination_type'))
    logs = str(r.get('logs') or '')
    seg = {}
    if logs.startswith('{'):
        try:
            seg = eval(logs)
            if not isinstance(seg, dict):
                seg = {}
        except Exception:
            seg = {}
    syntax = str(seg.get('syntax') or '').strip()
    compile_ = str(seg.get('compile') or '').strip()
    sim = str(seg.get('sim') or '').strip()
    synth = str(seg.get('synthesis') or '').strip()
    if bool(r.get('simulation_pass')):
        return 'This code is correct.'
    if ht == 'syntax':
        if syntax:
            return f'Compile failed. Verilog parser error: {syntax[:CLIP]}'
        return ('RTL extraction failed: no Verilog module could be extracted '
                'from the model output.')
    if ht == 'elaboration':
        if compile_:
            return f'Elaboration failed. Compiler diagnostic: {compile_[:CLIP]}'
        return 'Elaboration failed: the design failed iverilog elaboration.'
    if ht == 'functional':
        if sim and len(sim) > 30 and 'Error' not in sim[:30]:
            return (f'Functional simulation failed. Simulator output: '
                    f'{sim[:CLIP]}')
        return ('Functional simulation failed: the candidate did not produce '
                'matching outputs against the provided testbench. '
                '(no I/O mismatch details recorded)')
    if ht == 'synthesis':
        return (f'Post-synthesis check failed. Yosys reported: '
                f'{(synth or sim or compile_)[:CLIP] or "synthesis check failed"}')
    # unknown/none-but-failed
    if compile_:
        return f'Compilation failed. Tool diagnostic: {compile_[:CLIP]}'
    return ('Verification failed for an unclassified reason '
            '(no detailed diagnostics recorded).')


def main(models):
    folds = json.load(open(f'{OUT}/fold_assignments.json'))
    tasks = {t['task_id']: t for t in json.load(open(f'{STEP2}/tasks_step2.json'))}
    t2k = {}
    for k in range(5):
        for t in folds['folds'][f'fold{k}']['test_tasks']:
            t2k[t] = k
    outp = f'{OUT}/results/rankek_quadruples.jsonl'
    fb_dist = collections.Counter()
    n = 0
    with open(outp, 'w') as fo:
        for m in models:
            gen = {json.loads(l)['generation_id']: json.loads(l)
                   for l in []}
            genmap = collections.defaultdict(dict)
            for line in open(f'{STEP2}/{m}/gen.jsonl'):
                r = json.loads(line)
                genmap[r['task']][int(r['generation_id'])] = r
            evmap = collections.defaultdict(dict)
            for line in open(f'{STEP2}/{m}/eval.jsonl'):
                r = json.loads(line)
                evmap[r['task']][int(r['generation_id'])] = r
            for t, gmap in genmap.items():
                meta = tasks.get(t, {})
                spec = meta.get('prompt', '')
                oof = t2k.get(t)
                for gid in sorted(gmap):
                    g = int(gid)
                    ev = evmap[t].get(gid)
                    if ev is None:
                        continue
                    rtl = ev.get('rtl') or ''
                    label = int(bool(ev.get('simulation_pass')))
                    if not rtl.strip():
                        fb = ('RTL extraction failed: no Verilog module could '
                              'be extracted from the model output.')
                    else:
                        fb = build_feedback(ev, rtl)
                    fb_dist[fb.split('.')[0][:60]] += 1
                    fo.write(json.dumps(dict(
                        generator=m, task=t,
                        design_group=meta.get('design_group'),
                        candidate_index=g,
                        outer_fold=oof,       # None=RTLLM(冻结迁移, 不入训练)
                        prompt=spec, candidate_rtl=rtl,
                        label=label, feedback=fb), ensure_ascii=False) + '\n')
                    n += 1
    json.dump(dict(total=n, feedback_head_dist=dict(
        collections.Counter({k: v for k, v in fb_dist.most_common(12)}))),
        open(f'{OUT}/results/rankek_quadruples_stats.json', 'w'),
        indent=1, ensure_ascii=False)
    print('quadruples:', n, '->', outp)
    for k, v in fb_dist.most_common(12):
        print(f'  {v:6d}  {k}')


if __name__ == '__main__':
    ms = sys.argv[1].split(',') if len(sys.argv) > 1 else MODELS
    main(ms)
