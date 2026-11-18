#!/usr/bin/env python3
"""冻结轮 D: 生成 PAPER_EVIDENCE_LEDGER.md + FINAL_FREEZE_REPORT.md。"""
import json, csv, os, hashlib

OUT = 'experiments/recprobe_group5fold'

def main():
    uni = json.load(open(f'{OUT}/results/freeze_metrics_unified.json'))['metrics']
    rep = json.load(open(f'{OUT}/results/rtllm_policy_replay_v2.json'))
    cost = json.load(open(f'{OUT}/results/scorer_cost_ood.json'))
    chk = json.load(open(f'{OUT}/results/freeze_checkpoint_integrity.json'))
    ovl = json.load(open(f'{OUT}/results/ood_overlap_audit.json'))['summary']
    qa = json.load(open(f'{OUT}/qa_ood_checks.json'))

    L = []
    L.append('# PAPER EVIDENCE LEDGER(论文数字依据表, 唯一有效版)')
    L.append('\n> 数字 → 来源文件 → 字段/过滤 → 复算方式。所有主结果可从逐任务文件独立复算。')
    L.append('> 内部代号 → 论文术语映射见文末; 过期数字(作废)单列于 §5。')
    L.append('\n## 1. IID 主结果(VE 测试折, 3,744 池 = 312 任务 × 12 生成器, B=4)\n')
    L.append('| 论文数字 | 值 | 来源 | 复算 |')
    L.append('|---|---|---|---|')
    L.append('| Single-shot pass / tokens | 35.58% / 1,847 | `predictions/decisions_trackA.csv` IID, method=alone,B=- | mean(submitted_pass) / mean(gen_tokens) 宏平均 |')
    L.append('| Self-consistency@4 | 39.42% / 7,390 | 同上 method=sc,B=4 | 同上 |')
    L.append('| RecProbe@4(主方法) | 39.93% / 2,905 | 同上 method=recprobe,B=4 | 同上 |')
    L.append('| CPP-only@4 | 41.45% / 4,790 | 同上 method=q1only,B=4 | 同上 |')
    L.append('| Oracle@4 | 53.74% / 7,390 | 同上 method=oracle,B=4 | 同上 |')
    L.append('| 排名: RankEF / Q1 / SLM / LAT pass@rank-1 | 44.58 / 41.69 / 39.69 / 38.43% | `results/freeze_pool_metrics.csv` domain=IID | 按 scorer 分组 mean(pass_rank1) |')
    L.append('| 排名 AUROC | 0.7833 / 0.6871 / 0.6516 / 0.5696 | 同上 mean(auroc over both-class pools) | 同上 |')
    L.append('| 排名 AUPRC | 0.8424 / 0.7843 / 0.7537 / 0.7221 | 同上 mean(auprc) | 同上 |')
    L.append('\n## 2. RTLLM OOD 排名(5 折, 3,000 池-折观测 × 4 scorer)\n')
    L.append('| 论文数字 | AUROC | AUPRC | Pass@rank-1 | Δ vs CPP [CI] |')
    L.append('|---|---:|---:|---:|---|')
    names = {'rankef': 'RankEF(微调 770M)', 'slm_judge': 'SLM-Judge 3B(zero-shot)',
             'cpp': 'CPP(生成器内部探针)', 'lat_l27': 'LAT-L27'}
    for s in ['rankef', 'slm_judge', 'cpp', 'lat_l27']:
        v = uni[f'OOD|{s}']
        src = 'results/rtllm_ranking_compare_5fold.json' if s != 'cpp' else '—(基线)'
        L.append(f"| {names[s]} | {v['auroc_micro']} | {v['auprc_micro']} | "
                 f"{v['pass_rank1_micro_pct']:.2f}% | "
                 + ('—' if s == 'cpp' else
                    json.dumps(json.load(open(f'{OUT}/results/rtllm_ranking_compare_5fold.json'))[s]['delta_vs_cpp_pp']))
                 + ' |')
    L.append('\n复算: `freeze_pool_metrics.csv` WHERE domain=OOD GROUP BY scorer → '
             'mean(auroc/auprc/pass_rank1)(both-class 池); CI 用 task-cluster bootstrap')
    L.append('(整簇抽 50 任务 ID, 12 生成器同进同出, 10,000 次, seed=42)。\n')
    L.append('## 3. RTLLM OOD 策略回放(12 臂, 规则 IID 冻结零样本迁移)\n')
    L.append('| 臂(论文术语) | Pass | Tokens | Δ vs alone [CI] | Δ vs RecProbe@4 [CI] |')
    L.append('|---|---:|---:|---|---|')
    term = {'alone@4': 'Single-shot', 'retry@4': 'Retry@4', 'sc@4': 'Self-consistency@4',
            'q1only@4': 'CPP-only@4', 'q1only_tok@4': 'CPP-only-Token@4',
            'prompt_only_tok@4': 'Prompt-only@4', 'q2lite@4': 'Q2-Lite@4',
            'q2lite_tok@4': 'Q2-Lite-Token@4', 'recprobe@4': 'RecProbe@4',
            'hybrid@4': 'Hybrid@4', 'rankef_adaptive@4': 'RankEF-Adaptive@4',
            'oracle_stop@4': 'Oracle-Stop@4', 'oracle@4': 'Oracle@4'}
    for arm, s in rep['arms'].items():
        dl = rep['deltas'].get(arm, {})
        va, vr = dl.get('vs_alone_pp'), dl.get('vs_recprobe_pp')
        f = lambda d: f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]" if d else '—'
        L.append(f"| {term.get(arm, arm)} | {100*s['pass']:.2f}% | {s['mean_tokens']:.0f} | "
                 f"{f(va)} | {f(vr)} |")
    L.append('\n复算: 单调臂 ← `predictions/decisions_q2v2.csv` OOD 行; RankEF-Adaptive/Hybrid ← '
             '`results/freeze_arm_scorer_cost.csv`(含 kept_gids/决策/tokens, 逐任务); '
             '宏平均+CI 已从该 CSV 独立复算并与上表逐位一致。\n')
    L.append('## 4. Scorer 成本(冻结修正版)\n')
    L.append('| Scorer | 候选评分数 | forward calls | scorer input tokens |')
    L.append('|---|---|---|---|')
    fc = cost['forward_accounting_correction']
    L.append(f"| CPP / LAT | 0 | 0 | 0(复用生成时残差) |")
    L.append(f"| RankEF IID | {fc['rankef_iid']['candidate_scorings']:,} | {fc['rankef_iid']['forward_calls_batched']:,} 批(变长 1-4) | 均值 {cost['rankef_iid_reference']['mean_in_tokens']} /候选(截断后) |")
    L.append(f"| RankEF OOD | {fc['rankef_ood']['candidate_scorings']:,} | {fc['rankef_ood']['forward_calls_batched']:,} 批(均匀 batch=4) | {cost['rankef']['input_tokens_per_candidate_mean_capped512']} /候选(capped512; 原始 {cost['rankef']['input_tokens_per_candidate_mean_raw']}) |")
    ac = cost['arm_deployment_cost']
    L.append(f"| RankEF-Adaptive@4 部署(serial) | 均 {ac['rankef_adaptive@4']['mean_candidates_scored_serial']} /任务 | 同左(batch=1) | {ac['rankef_adaptive@4']['mean_scorer_in_tokens_serial_raw']} raw / {ac['rankef_adaptive@4']['mean_scorer_in_tokens_serial_capped512']} capped 每任务 |")
    L.append(f"| Hybrid@4 部署(serial) | 均 {ac['hybrid@4']['mean_candidates_scored_serial']} /任务 | 同左 | {ac['hybrid@4']['mean_scorer_in_tokens_serial_raw']} raw / {ac['hybrid@4']['mean_scorer_in_tokens_serial_capped512']} capped 每任务 |")
    L.append(f"| SLM-Judge | OOD {fc['slm_judge']['ood_request_level_forwards']:,} / IID {fc['slm_judge']['iid_request_level_forwards']:,} 请求 | 同左(每请求 1 输出 token) | {cost['slm_judge']['input_tokens_per_candidate_mean']} /请求(p95 {cost['slm_judge']['p95']}) |")
    L.append('\n复算: `predictions/rtllm_rankek_in_tokens.csv`(逐候选) + `results/freeze_arm_scorer_cost.csv`(逐任务两臂)。\n')
    L.append('## 5. 作废数字(禁止进入论文)\n')
    L.append('| 作废值 | 原因 | 生效替代 |')
    L.append('|---|---|---|')
    L.append('| RankEF OOD AUROC 0.5363 / pass@rank-1 33.17%(v3) | encoder 误加载预训练 base | 0.7333 / 39.67%(五折) |')
    L.append('| "OOD 排名完全反转"叙事(v3) | 加载伪象 | RankEF≈SLM 并列最强 |')
    L.append('| v3 `rtllm_ranking_per_model.csv`(2,400 行 fold0) | 折覆盖不完整 | `rtllm_ranking_per_model_5fold.csv`(240 行) |')
    L.append('| scorer_forwards=2,400(早期成本口径) | 实为候选评分数 | candidate_scorings 与 forward_calls 分列 |')
    L.append('| 审计表 Q2-Lite@4 34.60%/Prompt-only 34.23%/Q2-Lite-Tok 34.10% 三行 | 与仓库 CSV 不符(来源不明) | 34.97 / 34.27(prompt_only_tok) / 34.23(q2lite_tok) |')
    L.append('\n## 6. 内部代号 → 论文术语\n')
    L.append('| 内部代号 | 论文术语 |')
    L.append('|---|---|')
    for a, b in [('CPP / q1 / P1', 'Generator-internal correctness probe (RecProbe-Q1)'),
                 ('recprobe@4', 'RecProbe@4 (Q1 rank + CCG stop)'),
                 ('q1only@4', 'CPP-only@4'), ('q2lite / q2lite_tok', 'Q2-Lite / Q2-Lite-Token'),
                 ('prompt_only_tok', 'Prompt-only gate'), ('rankef_adaptive', 'RankEF-Adaptive'),
                 ('hybrid', 'Hybrid (CCG stop + RankEF select)'),
                 ('decisions_q2v2.csv / decisions_trackA.csv', 'Internal replay ledgers(共享臂 0.000pp 一致)'),
                 ('q2v2 pool npz', 'Internal feature store'), ('OOD_ENS', '(不入论文)')]:
        L.append(f'| {a} | {b} |')
    L.append('\n## 7. 重叠审计覆盖范围(如实声明)\n')
    L.append(f"- 已覆盖: exact prompt 重复(小写+空白折叠归一化)={ovl['exact_prompt_dupes']}; "
             f"3-gram Jaccard≥0.5 近重复={ovl['near_dup_jaccard_ge_05']}(≥0.3 亦 {ovl['near_dup_jaccard_ge_03']}); "
             f"module 同名+端口(方向,名称)集合全等冲突={ovl['module_name_collisions']}")
    L.append('- **未覆盖(不得声称)**: 语义/embedding 相似度去重未运行; testbench 语义等价未审计; '
             '自然语言改写检测未做; 端口模糊匹配(仅全等)未做')
    L.append('- 允许的主张: "跨基准未检出 exact 或高词汇重叠任务"; '
             '**不允许**: "已语义去重"。\n')
    open(f'{OUT}/results/PAPER_EVIDENCE_LEDGER.md', 'w').write('\n'.join(L))
    print('PAPER_EVIDENCE_LEDGER.md written')

    # ---------- FINAL_FREEZE_REPORT ----------
    F = []
    F.append('# FINAL FREEZE REPORT(RTLLM OOD 数据冻结, 唯一有效版)\n')
    F.append('## A. 八项任务执行情况\n')
    F.append('| # | 任务 | 状态 | 关键产物 |')
    F.append('|---|---|---|---|')
    F.append('| 1 | RankEF 候选评分数与 forward 修正 | 完成 | '
             f"candidate_scorings 12,000(5×2,400)与 forward_calls 3,000 批(batch=4 均匀)分列; IID {fc['rankef_iid']['candidate_scorings']:,} 候选/{fc['rankef_iid']['forward_calls_batched']:,} 批(变长) |")
    F.append('| 2 | Hybrid/Adaptive 逐任务 scorer 成本 | 完成 | freeze_arm_scorer_cost.csv(6,000 行); serial 部署均 2.545 / 1.727 次评分每任务 |')
    F.append('| 3 | checkpoint 完整 SHA-256 + 一致性 | 完成 | freeze_checkpoint_integrity.json: 文件级+张量级哈希, '
             f"verdict={chk.get('verdict')} |")
    F.append('| 4 | IID/OOD AUROC/AUPRC 统一口径 | 完成 | freeze_pool_metrics.csv(26,976 池行)+ freeze_metrics_unified.json(micro+macro+逐折) |')
    F.append('| 5 | 每模型×折 池数/正负统计 | 完成 | freeze_label_stats.csv(120 行) |')
    F.append('| 6 | 旧 Manifest 归档 + 唯一最终 Manifest | 完成 | handoff-final/manifest_archive/(v2,v3,v4) + handoff-final/MANIFEST.sha256 |')
    F.append('| 7 | 论文数字依据表 | 完成 | PAPER_EVIDENCE_LEDGER.md(§5 作废数字清单) |')
    F.append('| 8 | 重叠审计范围声明 | 完成 | Ledger §7(明确 4 项未覆盖, 禁止语义去重声称) |')
    F.append('\n## B. 修正前后差异\n')
    F.append('| 项 | 前 | 后 |')
    F.append('|---|---|---|')
    F.append('| RankEF OOD forward 口径 | "scorer_forwards=2,400/折"(实为候选数) | candidate_scorings 2,400/折; forward_calls 600 批/折(batch=4) |')
    F.append('| 成本粒度 | 仅聚合均值 | 逐候选 in_tokens CSV + 两臂逐任务 serial/offline 双口径 |')
    F.append('| checkpoint 溯源 | 首 1MB 截断哈希 | 完整 SHA-256(文件)+ state_dict 张量哈希(加载内容) |')
    F.append('| AUROC 口径 | 各脚本各自实现 | 单一 freeze_pool_metrics.csv + 统一函数; micro/macro/逐折并报 |')
    F.append('| 逐模型 CSV | 2,400 行 fold0 旧版 | 240 行五折版(折加权=聚合逐位一致) |')
    F.append('\n## C. QA\n')
    F.append(f"- Track A 冻结 QA: 165/165(继承, 未改动 IID 数据)")
    F.append(f"- OOD QA: {qa['n_pass']}/{qa['n_total']} {qa['verdict']}")
    F.append("- 冻结新增自证: freeze_arm_scorer_cost.csv 复算宏平均与 CI = replay JSON 逐位一致"
             "(adaptive 37.20%/4,860 CI[+2.27,+6.13]; hybrid 34.83%/3,323 CI[+0.57,+3.13])")
    F.append("- 池级 CSV 复算 AUROC/AUPRC/pass@rank-1 = 各聚合 JSON 逐位一致\n")
    F.append('## D. 最终论文数字(唯一有效)\n')
    F.append('### IID(3,744 池)')
    F.append('| 方法 | Pass | Tokens |')
    F.append('|---|---:|---:|')
    for k, v in [('alone@4', None), ('sc@4', None), ('recprobe@4', None),
                 ('q1only@4', None), ('oracle@4', None)]:
        pass
    ta = {}
    for r in csv.DictReader(open(f'{OUT}/predictions/decisions_trackA.csv')):
        if r['domain'] != 'IID':
            continue
        key = (r['method'], r['B'] if r['B'] != '-' else '1')
        ta.setdefault(key, []).append((int(r['submitted_pass']), int(r['gen_tokens'])))
    import collections
    for meth, b in [('alone', '1'), ('sc', '4'), ('recprobe', '4'), ('q1only', '4'), ('oracle', '4')]:
        d = ta.get((meth, b))
        if not d:
            continue
        bym = collections.defaultdict(lambda: ([], []))
        for (pp, tt), mm in zip(d, [None]*len(d)):
            pass
        # 宏平均(按模型)
        rows = list(csv.DictReader(open(f'{OUT}/predictions/decisions_trackA.csv')))
        sub = [r for r in rows if r['domain'] == 'IID' and r['method'] == meth
               and (r['B'] == b or (meth == 'alone' and r['B'] == '-'))]
        bym2 = collections.defaultdict(list); byt2 = collections.defaultdict(list)
        for r in sub:
            bym2[r['model']].append(int(r['submitted_pass']))
            byt2[r['model']].append(int(r['gen_tokens']))
        mp = 100*float(np.mean([np.mean(v) for v in bym2.values()])) if False else None
        import numpy as np
        mp = 100*float(np.mean([np.mean(v) for v in bym2.values()]))
        mt = float(np.mean([np.mean(v) for v in byt2.values()]))
        F.append(f"| {term.get(meth+'@'+b, meth)} | {mp:.2f}% | {mt:.0f} |")
    F.append('')
    F.append('### RTLLM OOD')
    F.append('- 排名与回放全部数字以 §Ledger §2/§3 为准(本报告不重复抄写, 避免双源漂移)')
    F.append(f"- checkpoint verdict: **{chk.get('verdict')}**; QA {qa['n_pass']}/{qa['n_total']}")
    F.append('\n## E. 冻结声明\n')
    F.append('- 本轮未新增模型/Benchmark/方法/候选; 所有数字可由 handoff-final/ 逐任务文件独立复算')
    F.append('- 唯一有效 Manifest: `handoff-final/MANIFEST.sha256`; 历史版本在 manifest_archive/')
    open(f'{OUT}/results/FINAL_FREEZE_REPORT.md', 'w').write('\n'.join(F))
    print('FINAL_FREEZE_REPORT.md written')

if __name__ == '__main__':
    main()
