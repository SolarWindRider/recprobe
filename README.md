# RecProbe — Evaluation Code

Evaluation, replay, and statistics code for **RecProbe: execution-free candidate
control for LLM Verilog generation** (RTL code generation; DAC submission).

This repository contains **only the evaluation/replay code** and the two protocol
split files. It intentionally contains no model weights, no raw generations, and
no third-party benchmark text.

## What RecProbe is

Given a code LLM that samples up to `B` candidates per task, RecProbe trains two
lightweight logistic probes on the model's **generation-time residual stream**
(layer 27, final-window position `w100`):

- **Q1 (CPP)** — candidate correctness probe: which sampled candidate is likely
  to pass the testbench;
- **Q2 (CCG)** — continue/stop gate: whether resampling this task is worth the
  extra tokens.

Everything is **execution-free**: real testbench outcomes are used only to
*score* submitted answers after the fact, never inside any decision loop.

## Repository layout

```
code/                       all evaluation code (flat; scripts import siblings)
├── track_a_common.py       shared loaders: canonical task/row lists, texts, folds
├── track_a_score.py        probe fitting (logistic, standardized)
├── track_a_scalar.py       per-model scalar-probe metadata loader
├── track_a_replay.py       offline replay: RecProbe / Q1-only / SC / retry / oracle (IID+OOD)
├── track_a_judge_replay.py same-generator LLM Self-Judge baseline replay
├── track_a_stats.py        macros, design-group paired bootstrap, per-fold tables
├── q2v2_lite.py            token-parity arms (Q2-Lite*, Prompt-only*, Q1-only-token, ...)
├── rankef_adaptive_replay.py  RankEF-Adaptive (IID): validation-frozen thresholds
├── freeze_budget_replay.py    Fig 3/4 budget sweep: 6 policies x B in {2,4,8} x {IID,OOD}
├── competitor_extension_replay.py  LAT-L27 / SLM-Judge-Fixed budget lines
├── rtllm_policy_replay.py     OOD policy replay (12 arms) + task-cluster bootstrap
├── rtllm_adaptive_replay_ood.py  RankEF-Adaptive/Hybrid on OOD (frozen tau_R)
├── rtllm_compare_5fold.py     OOD five-fold ranker comparison (stable tie-break)
├── freeze_metrics.py          unified AUROC/AUPRC, pool-level metrics (IID+OOD)
├── freeze_cost_forward.py     scorer cost accounting (candidates vs forward calls)
├── freeze_checkpoint_sha.py   checkpoint integrity (full SHA-256, load consistency)
├── ood_overlap_audit.py       cross-benchmark duplicate audit (exact / 3-gram / ports)
├── prompt_tokens.py           scorer prompt-token accounting
├── rankek_quadruples.py       RankEF training-data builder (spec, rtl, label quadruples)
├── rankek_train.py            RankEF training (CodeT5+ 770M, cls head, per fold)
├── rankek_score.py / rankek_score_gids47.py / rtllm_rankek_score_5fold.py  RankEF scoring
├── lat_l27.py / rtllm_lat_score.py / lat_score_gids47.py   LAT-L27 baseline (PCA difference probe)
├── slm_judge_zero_shot.py / slm_judge_gids47.py  external SLM-Judge (Qwen2.5-Coder-3B, zero-shot)
├── step2_judge_scores.py      same-generator Self-Judge probability extraction
├── step2_scalar_probe.py / probe_seeds.py / probe_hidden.py / fh_predict.py
│                             generator-internal probe training (layer sweep, seeds)
├── step2_sealed_eval.py / step2_sealed_ci.py   sealed-set evaluation + bootstrap CI
├── qa_final.py / qa_ood_checks.py              QA suites (165 + 7 checks)
└── gen_ood_report_v2.py / gen_freeze_reports.py  report generation
protocols/
├── sealed_split.json          sealed IID/shift task split (frozen before evaluation)
└── fold_assignments.json      design-grouped 5-fold assignment (frozen)
```

## Protocols (pre-registered)

- **Sealed split.** 122 evaluation tasks (72 IID + 50 RTLLM shift) frozen in
  `protocols/sealed_split.json` before any evaluation; probe weights are fit
  only on training-fold tasks.
- **Design-grouped five-fold cross-validation.** Frozen in
  `protocols/fold_assignments.json`; folds are balanced by design group so that
  task variants never straddle folds.
- **Tie-break.** All rankings use `np.argsort(-score, kind="stable")`: ties are
  resolved toward the earliest generation id.
- **Bootstrap.** Paired confidence intervals use task-cluster resampling
  (10,000 draws, seed 42): one draw resamples the 50 RTLLM task IDs and keeps
  *all* generators of a drawn task together.
- **Threshold freezing.** RankEF-Adaptive thresholds `tau_R` are calibrated on
  IID validation only (per generator, per fold, B=4) and **reused unchanged**
  at B=2/8 and on OOD. No test-time tuning anywhere.
- **External scorer costs** (judge prompts, RankEF input tokens) are reported
  separately from generated output tokens and never counted against the
  generation budget.

## Data layout expected by the code

Scripts read a workspace rooted at the repository root:

```
experiments/step2/<generator>/gen.jsonl      sampled candidates (text + generation id)
experiments/step2/<generator>/eval.jsonl     per-candidate testbench outcome (scoring only)
experiments/step2/<generator>/labels.jsonl   generation lengths + failure class
experiments/step2/hidden_stream/<gen>/<hash>.npz   saved residual-stream buckets
experiments/step2/sealed_split.json          -> protocols/sealed_split.json
experiments/recprobe_group5fold/fold_assignments.json -> protocols/fold_assignments.json
experiments/recprobe_group5fold/checkpoints/rankek_fold<k>/   RankEF checkpoints
```

Residual-stream `.npz` files store, per sampled candidate, a position manifest
(`pos_manifest`) plus arrays named `{position_tag}@L{layer}`; the production
feature is the `w100` window at layer 27 (`w100@L27`). The layer anchor is a
pre-registered protocol constant; a post-hoc depth sweep (7 generators x 18
positions x 4 layers, leave-one-task-out, label permutation control) confirmed
monotonically increasing separability with depth.

Generators: 12 open-weight code LLMs (28-48 layers; Qwen/DeepSeek families).
Tasks: 362 (312 VerilogEval + 50 RTLLM), 32 samples each, temperature 0.8.

## Reproducing the paper numbers

| Claim / table | Entry points |
|---|---|
| IID main table (pass & tokens per policy) | `track_a_replay.py` + `track_a_stats.py` |
| OOD ranking (five scorers, five folds) | `rtllm_compare_5fold.py`, `freeze_metrics.py` |
| OOD policy replay (12 arms, CIs) | `rtllm_policy_replay.py`, `rtllm_adaptive_replay_ood.py` |
| Budget sweep B=2/4/8 (Fig 3/4) | `freeze_budget_replay.py`, `competitor_extension_replay.py` |
| Scorer cost accounting | `freeze_cost_forward.py`, `prompt_tokens.py` |
| Checkpoint integrity | `freeze_checkpoint_sha.py` |
| Duplicate audit | `ood_overlap_audit.py` |
| QA suites | `qa_final.py` (165 checks), `qa_ood_checks.py` (7 checks) |

Every published number is a macro over per-task rows; the replay scripts emit
one row per (domain, generator, fold, task, policy, budget) so all aggregates
can be recomputed independently from those files.

## Requirements

Python >= 3.10; `numpy`, `scikit-learn`, `scipy`, `pandas`; scorer training and
inference additionally use `torch` (Ascend NPU build, `torch_npu`) or any CUDA
build of PyTorch, `transformers`, and (for the judge baselines) `sglang`.
`_sglang_patch.py` is an environment-compatibility shim and can be deleted when
running on CUDA.

## Notes

- `rankek_score.py` and the older fold0-only OOD scripts were superseded by
  their `_5fold` / `_gids47` counterparts; they are kept for provenance.
- The layer-sweep diagnostic (`probe_hidden.py` + `probe_auroc.json` outputs)
  is a post-hoc analysis and does not participate in any model selection: the
  layer anchor was fixed at generation time, before any probe was trained.
