#!/usr/bin/env python3
"""重生成 OOD_FINAL_REPORT.md(v2: 五折排名 + 修正 RankEF + Adaptive/Hybrid OOD)。"""
import json

OUT = 'experiments/recprobe_group5fold'

def fmt_ci(d):
    return f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]" if d else '—'

def main():
    cmp5 = json.load(open(f'{OUT}/results/rtllm_ranking_compare_5fold.json'))
    rep = json.load(open(f'{OUT}/results/rtllm_policy_replay_v2.json'))
    prov = json.load(open(f'{OUT}/results/rankek_ood_checkpoint_provenance.json'))
    qa = json.load(open(f'{OUT}/qa_ood_checks.json'))
    ovl = json.load(open(f'{OUT}/results/ood_overlap_audit.json'))

    SC = {'cpp': 'CPP (Q1 LR probe)', 'lat_l27': 'LAT-L27 (PCA diff)',
          'slm_judge': 'SLM-Judge 3B (zero-shot)', 'rankef': 'RankEF 770M (fine-tuned ckpt)'}
    L = []
    L.append('# RTLLM OOD 评估报告(v2, 审计修正版)\n')
    L.append('> 修正记录: (1) RankEF OOD encoder 改从微调 checkpoint 加载'
             '(初版误用预训练 base, 初版结果作废); (2) 排名对照升级为五折;'
             '(3) tie-break = stable 排序并列最早 gid; (4) bootstrap = task-cluster'
             '(50 任务整簇, 12 生成器同进同出); (5) 新增 Adaptive/Hybrid OOD 回放'
             '(IID-val 冻结阈值)。\n')

    L.append('## 1. 四 scorer 五折 OOD 排名(固定池 B=4, 12 生成器 × 50 任务)\n')
    L.append('| Scorer | Pass@rank-1(5折均值) | 折间范围 | AUROC | Δ vs CPP [task-cluster 95% CI] |')
    L.append('|---|---:|---|---:|---|')
    for s in ['cpp', 'slm_judge', 'rankef', 'lat_l27']:
        d = cmp5.get(s)
        if not d:
            continue
        pf = d.get('per_fold_pass1', {})
        rng = (f"{min(pf.values()):.1f}–{max(pf.values()):.1f}" if pf else '—')
        L.append(f"| {SC[s]} | {d['pass@rank1']:.2f}% | {rng} | "
                 f"{d['candidate_auroc']:.4f} | {fmt_ci(d.get('delta_vs_cpp_pp'))} |")
    L.append(f"\ntie 并列率(fold0 最高分并列): "
             f"{json.dumps(cmp5.get('_tie_rate_at_max_fold0', {}))}\n")

    L.append('## 2. 策略回放 v2(全部规则 IID 冻结, 零样本迁移)\n')
    L.append('| 臂 | Pass | Tokens | Δ vs alone [CI] | Δ vs recprobe@4 [CI] |')
    L.append('|---|---:|---:|---|---|')
    for arm in ['alone@4', 'sc@4', 'q1only@4', 'q1only_tok@4', 'prompt_only_tok@4',
                'q2lite@4', 'q2lite_tok@4', 'recprobe@4', 'hybrid@4',
                'rankef_adaptive@4', 'oracle_stop@4', 'oracle@4']:
        s = rep['arms'].get(arm)
        if not s:
            continue
        dl = rep['deltas'].get(arm, {})
        L.append(f"| {arm} | {100*s['pass']:.2f}% | {s['mean_tokens']:.0f} | "
                 f"{fmt_ci(dl.get('vs_alone_pp'))} | {fmt_ci(dl.get('vs_recprobe_pp'))} |")

    L.append('\n## 3. checkpoint 溯源\n')
    L.append('| Fold | Encoder SHA-256(首 1MB, 前 16) | cls step | val F1 |')
    L.append('|---|---|---:|---:|')
    for k in range(5):
        L.append(f"| fold{k} | `{prov.get(f'fold{k}_encoder_sha256_1MB','')[:16]}` | "
                 f"{prov.get(f'fold{k}_cls_step')} | "
                 f"{prov.get(f'fold{k}_cls_val_f1', 0):.4f} |")
    L.append('\nIID 与 OOD 评分使用同一 checkpoint 文件'
             '(`checkpoints/rankek_fold{k}/pytorch_model.bin`),路径与 SHA 见'
             ' `rankek_ood_checkpoint_provenance.json`。\n')

    L.append('## 4. 跨基准重叠审计\n')
    s = ovl['summary']
    L.append(f"- VE 任务 {s['n_ve']} / RTLLM 任务 {s['n_rtllm']}; "
             f"exact prompt 重复 **{s['exact_prompt_dupes']}**; "
             f"3-gram Jaccard≥0.5 近重复 **{s['near_dup_jaccard_ge_05']}**"
             f"(≥0.3 亦为 {s['near_dup_jaccard_ge_03']}); "
             f"module 同名冲突 **{s['module_name_collisions']}**")
    L.append(f"- 判定: **{s['verdict']}**\n")

    L.append('## 5. OOD QA\n')
    L.append(f"- QA: **{qa['n_pass']}/{qa['n_total']}** — **{qa['verdict']}**")
    for c in qa['checks']:
        L.append(f"  - [{'X' if c['pass'] else ' '}] {c['check']}")
    L.append('')
    open(f'{OUT}/results/OOD_FINAL_REPORT.md', 'w').write('\n'.join(L))
    print('written OOD_FINAL_REPORT.md (v2)')

if __name__ == '__main__':
    main()
