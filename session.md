# Session Notes

## 2026-05-22T02:44:38Z Workflow Start

- Workflow skill used earlier: `kernel-eval-workflow`.
- Scope: process current TODO by priority, starting with measurable evaluation gap analysis for existing profiling before changing kernel models.
- User constraint: `duration_us < 10us` can be treated as launch-overhead dominated.
- Large/occupied gap policy: exclude `duration_us < 10us`; focus on rows with `block_dim`/`mix_block_dim` close to `aic_num` or high `cube_utilization_pct`.
- Commit policy at that point: default local commit after feature/bugfix completion unless user says not to commit.

## 2026-05-22T02:44:38Z Gap Analysis Workflow

- Added reusable large shape / occupied-core report post-processor: `tools/analyze_large_shape_gap.py`.
- Default filters: `duration_us >= 10`, `block_dim` or `mix_block_dim >= 0.8 * aic_num`, or `cube_utilization_pct >= 70`.
- Regenerated base 910B4/910C and `profiling_with_model_code` MatMul/Attention reports with matched platform configs.
- Key result at that time:
  - 910C attention large shape max relative error about `0.116`, p95 about `0.099`.
  - Base 910C MatMul large shape max about `0.542`, mostly `MatMulV2 M=1` path.
  - Base 910B4 attention large shape max about `0.259`.
- High-priority findings:
  - `GroupedMatmul` had many lower-bound violations because `group_list` values/routing are absent.
  - ds3.2 `KvQuantSparseFlashAttention` had max relative error about `1.840` and lower-bound violations.
  - qwen7b `MatMulV2` had lower-bound violation on all large-shape rows under then-current 910B4 assumptions.
- Docs: wrote `docs/current_eval_gap_zh.md` and linked it from `docs/architecture.md`.

## 2026-05-22T03:32:00Z GroupedMatmul Independent Evaluation

- Continued from high-priority TODO: `GroupedMatmul` routing data missing.
- Design: treat `GroupedMatmul` as an independent kernel family. Keep ordinary MatMul theoretical lower bound only as a compatibility/reference field, and add two explicit routing scenarios because `kernel_details.csv` does not expose `group_list`.
- Model:
  - Balanced scenario spreads tokens across `min(expert_count, M)` active experts.
  - Extreme imbalance routes all tokens to one expert.
  - Both scenarios charge active expert weight traffic, activation/output traffic, quant auxiliary bytes, Cube core work-tile occupancy, and dtype/quant compute peak where available.
- Development:
  - Added `tools/matmul_eval/gmm_model.py`.
  - Added `--op-kind grouped_matmul`.
  - Propagated GMM fields through CSV writer.
  - Added large-shape GMM routing-bound summary.
- Validation: `py_compile` passed for updated modules. GroupedMatmul reports generated for base 910B4, longcat/gemma on 910B4, and ds3.2 on 910C.
- Results at this point:
  - Base 910B4 large/occupied GMM bound error max `0.543`, p95 `0.473`, median `0.000`.
  - longcat max `0.542`, p95 `0.513`, median `0.391`.
  - gemma max/p95/median `0.000`.
  - ds3.2 max `0.176`, p95 `0.165`, median `0.090`.
- Review conclusion: this is not a fit. Above-bound rows are low-confidence residuals needing real `group_list` or GMM-specific scheduling/sync/merge evidence.
- Docs updated: `docs/architecture.md`, `docs/matmul_eval_design_zh.md`, `docs/current_eval_gap_zh.md`, and `session.md`.

## 2026-05-22T04:20:00Z GMM Source Review

- Correction: `GroupedMatmul` implementation is in `ops-transformer/gmm/grouped_matmul`, not `ops-nn/matmul`.
- Source review:
  - Host tiling parses `groupType`, `groupListType`, `tuningConfigOptional`, `singleN`, and `usedCoreNum`.
  - Quant path can `SetBlockDim(aicNum)`.
  - Kernel `grouped_matmul.h` loops over `groupList`, skips empty groups, and schedules blocks across groups with `count % coreNum`.
- Design update:
  - Do not infer active experts from `Block Dim`.
  - `Block Dim` indicates launched Cube cores for GMM, not non-empty expert count.
- Model update:
  - Keep balanced/extreme routing scenarios.
  - Add a source-visible group scheduler term for expert counts greater than Cube cores.
  - Use `gmm_routing_bound_error` as the large-shape error metric for GMM reports.
- Validation:
  - Regenerated longcat/gemma/ds3.2 grouped_matmul reports.
  - Longcat bound max improved from about `0.542` to about `0.179`.
  - Gemma remained `0.000`.
  - DS3.2 remained about `0.176`.
- Residual: DS3.2 INT8 GMM above-bound likely needs true `groupList`/`tuningConfigOptional` or exact quant adaptive sliding-window tiling replay.

## 2026-05-22T07:29:49Z Complete Baseline Refresh

- Regenerated a complete commit-scoped evaluation snapshot for HEAD `696f0c4`.
- Snapshot directory: `eval_results/20260522T072949Z_696f0c4`.
- `eval_results/LATEST` now points to this snapshot.
- Summary files tracked by git:
  - `eval_results/20260522T072949Z_696f0c4/eval_summary.csv`
  - `eval_results/20260522T072949Z_696f0c4/metadata.txt`
- Detailed resolved/unresolved CSVs were generated locally in the same snapshot directory, but remain ignored by `eval_results/.gitignore` to avoid committing bulky generated reports.
- Evaluation reports covered 16 combinations:
  - base 910B4/910C MatMul and Attention;
  - base 910B4 GroupedMatmul;
  - ds3.2 910C MatMul/Attention/GroupedMatmul;
  - gemma 910B4 MatMul/Attention/GroupedMatmul;
  - longcat 910B4 MatMul/Attention/GroupedMatmul;
  - qwen3/qwen7b inferred 910B4 MatMul/Attention.

Latest large/occupied precision summary:

| Report | large max | p95 | median | lower-bound notes |
|---|---:|---:|---:|---|
| base 910B4 Attention | 0.259 | 0.208 | 0.089 | no violations |
| base 910C Attention | 0.116 | 0.099 | 0.025 | no violations |
| base 910B4 GroupedMatmul | 0.180 | 0.074 | 0.000 | ordinary ideal lower bound invalid for GMM; use routing-bound |
| base 910B4 MatMul | 0.613 | 0.470 | 0.055 | 209 violations, mostly GMM/reference-bound related plus MatMulV2 tail |
| base 910C MatMul | 0.542 | 0.240 | 0.055 | no violations |
| ds3.2 910C Attention | 1.840 | 1.821 | 1.719 | 80 violations on KvQuantSparseFA |
| ds3.2 910C GroupedMatmul | 0.176 | 0.165 | 0.090 | no ordinary lower-bound violations, but 120/120 above GMM bound |
| ds3.2 910C MatMul | 0.689 | 0.651 | 0.166 | no violations |
| gemma 910B4 Attention | 0.416 | 0.236 | 0.152 | no violations |
| gemma 910B4 GroupedMatmul | 0.000 | 0.000 | 0.000 | within routing bounds; ordinary ideal lower bound not meaningful |
| gemma 910B4 MatMul | 0.516 | 0.188 | 0.052 | 180 GMM/reference-bound violations |
| longcat 910B4 Attention | n/a | n/a | n/a | no large occupied rows |
| longcat 910B4 GroupedMatmul | 0.179 | 0.157 | 0.040 | 17 above-bound and 11 within-bound rows |
| longcat 910B4 MatMul | 0.311 | 0.179 | 0.068 | 28 GMM/reference-bound violations |
| qwen3/qwen7b Attention inferred 910B4 | 0.468 | 0.459 | 0.416 | no violations; FIA decode launch floor low |
| qwen3/qwen7b MatMul inferred 910B4 | 0.811 | 0.584 | 0.390 | 483/483 violations |

New findings from this refresh:

- The previous GMM source-based routing-bound update is visible in current baseline: longcat/base GMM max is now around `18%`, gemma remains fully within bounds, and ds3.2 INT8 GMM remains around `17.6%` above bound.
- `KvQuantSparseFlashAttention` is still the largest attention problem and cannot be treated as regular Flash/FusedInfer attention. It needs a dedicated model from `ops-transformer/attention/kv_quant_sparse_flash_attention`.
- qwen3/qwen7b still maps to 910B4 by BlockNum principle, but MatMulV2 all-large rows violate the physical lower bound. This is not a tuning target; it indicates platform bandwidth, profiling unit, shape/storage interpretation, or kernel-path semantics are inconsistent.
- Base/hyimage 910B4 MatMul now exposes a worse `MatMulV2` small-M tail than the older historical README summary because GMM is no longer dominating the same way after routing-bound separation.
- Ordinary `ideal_lower_bound_us` is not a valid accuracy judge for GMM rows when group routing is unknown. GMM acceptance must use `gmm_routing_bound_error` and `gmm_position`.

Recommended next modeling order:

1. Build a dedicated `KvQuantSparseFlashAttention` evaluator from source-visible sparse/quant tiling and traffic behavior, then refresh ds3.2 attention baseline.
2. Audit qwen3/qwen7b MatMulV2 physical lower-bound violation before modeling changes: verify units, shapes, storage formats, HBM config, and whether exported `Block Num` rows correspond to a different kernel path.
3. Refine `MatMulV2` and `QuantBatchMatmulV3` small-M/long-K,N path from ops-nn source and tiling branches, especially decode-like M=1/2/4 cases.
4. Continue GMM only when real `groupList`, `groupListType`, `tuningConfigOptional`, or source-derived quant/GMM scheduling evidence is available.
5. Keep every feature change paired with a new commit-scoped `eval_results/<UTC>_<commit>/eval_summary.csv` refresh.

## 2026-05-22T08:00:00Z QSFA Attention Model Fix

- Continued TODO priority 1: dedicated `KvQuantSparseFlashAttention` model.
- Source basis:
  - `ops-transformer-master/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp`
  - `ops-transformer-master/attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.h`
  - `ops-transformer-master/attention/kv_quant_sparse_flash_attention/op_kernel/kv_quant_sparse_flash_attention_template_tiling_key.h`
- Key source facts used:
  - QSFA supports query `BSND/TND`, KV `BSND/TND/PA_BSND`.
  - PA mode derives `s2Size = block_table.dim1 * block_size`; key cache first dimension is total block count, not batch.
  - `gSize = n1Size / n2Size`.
  - `QSFAMlaTiling::InitParams()` uses `V_TEMPLATE_MODE`; template key exposes V template mode and split-G flag.
  - `CalcInnerSize()` uses default `sInnerSize=512`; A5/PA workspace formula exposes `S2_BASE_SIZE=128`, `D_SIZE=576`, preload buffers and topK cache bytes.
  - Query/output dtype is FP16/BF16 while key/value dtype is INT8/FP8/HIFLOAT8; generic attention byte accounting overcharged K/V as BF16.
- Code changes:
  - `tools/attention_eval/common.py`: added QSFA-specific shape parser for ds3.2 PA-like cache layout `[block_num,N,block_size,D]` and query/output `[T,N,D]`.
  - `tools/attention_eval/api.py`: added QSFA-specific input byte accounting and source-derived V-template workspace traffic for current-kernel estimate only.
  - `docs/attention_eval_iteration_plan.md`: documented source facts, model semantics and validation result.
- Validation commands:
  - `python3 -m compileall tools`
  - `python3 tools/eval_ops.py --op-kind attention --profiling example_profilings/profiling_with_model_code/ds3.2 --config configs/ascend_910c.json --output /tmp/ds32_attention_qsfa_eval.csv --unresolved-output /tmp/ds32_attention_qsfa_unresolved.csv`
  - `python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py /tmp/ds32_attention_qsfa_eval.csv`
  - Regression generated for base 910B4/910C attention plus gemma/qwen7b attention.
- Accuracy improvement:
  - ds3.2 QSFA before: max `1.840`, p95 `1.821`, median `1.719`, lower-bound violations `80`.
  - ds3.2 QSFA after: max `0.200`, p95 `0.156`, median `0.034`, lower-bound violations `0`.
  - Generic base 910C attention remained max `0.116`, p95 `0.097`, median `0.026`.
  - Generic base 910B4 attention remained max `0.295`, p95 `0.238`, median `0.097`.
- Remaining QSFA residual:
  - Top row line `519`: duration `122.0us`, estimate `97.59us`, relative error `20.0%`.
  - Residual likely requires exact tiling data, actual block table/sparse indices, and real PA/A5 workspace access count. No arbitrary fit knob was added.

## 2026-05-22T08:11:03Z Post-QSFA Baseline Refresh

- Regenerated complete commit-scoped snapshot for commit `0a7ccb1`.
- Snapshot directory: `eval_results/20260522T081103Z_0a7ccb1`.
- `eval_results/LATEST` now points to `20260522T081103Z_0a7ccb1`.
- Summary files tracked:
  - `eval_results/20260522T081103Z_0a7ccb1/eval_summary.csv`
  - `eval_results/20260522T081103Z_0a7ccb1/metadata.txt`
- Full report count remains 16.

Post-QSFA large/occupied precision summary:

| Report | large max | p95 | median | notes |
|---|---:|---:|---:|---|
| base 910B4 Attention | 0.259 | 0.208 | 0.089 | unchanged; FIA decode tail |
| base 910C Attention | 0.116 | 0.099 | 0.025 | unchanged; long FA prefill tail |
| ds3.2 910C Attention | 0.200 | 0.155 | 0.034 | QSFA fixed; no lower-bound violations |
| ds3.2 910C MatMul | 0.689 | 0.657 | 0.318 | QuantBatchMatmulV3 small-M remains |
| ds3.2 910C GroupedMatmul | 0.176 | 0.165 | 0.090 | 120/120 above GMM bound |
| qwen3/qwen7b Attention inferred 910B4 | 0.468 | 0.459 | 0.416 | launch-bound decode floor remains low |
| qwen3/qwen7b MatMul inferred 910B4 | 0.811 | 0.584 | 0.390 | 483/483 lower-bound violations remain |
| base 910B4 MatMul | 0.613 | 0.476 | 0.075 | GMM rows are no longer mixed into MatMul report |
| base 910C MatMul | 0.542 | 0.240 | 0.055 | MatMulV2 M=1 tail |
| gemma 910B4 Attention | 0.416 | 0.236 | 0.152 | FIA decode custom head dim |
| gemma 910B4 GroupedMatmul | 0.000 | 0.000 | 0.000 | within GMM bounds |
| gemma 910B4 MatMul | 0.516 | 0.192 | 0.072 | GMM rows are no longer mixed into MatMul report |
| longcat 910B4 GroupedMatmul | 0.179 | 0.157 | 0.040 | 17 above-bound, 11 within-bound |
| longcat 910B4 MatMul | 0.311 | 0.179 | 0.080 | GMM rows are no longer mixed into MatMul report |

New issues identified:

- MatMul and model-level MatMul rows now exclude `GroupedMatmul` when `--op-kind matmul` is used; this is a semantic cleanup in report boundaries. Historical rows/large counts are not directly comparable unless GMM is added back explicitly.
- QSFA residual is no longer a physical-bound issue. Remaining `20%` tail is a true current-kernel replay gap: exact block table/sparse indices, split-G/A5 path, and workspace reuse/access count are not available in profiling.
- qwen3/qwen7b MatMulV2 remains the most suspicious platform/model consistency issue. It still maps to 910B4 by BlockNum evidence, but all large rows violate `ideal_lower_bound_us`.
- ds3.2 QuantBatchMatmulV3 small-M and TransposeBatchMatMul still dominate ds3.2 MatMul p95.

Recommended next actions:

1. Audit qwen3/qwen7b MatMulV2 lower-bound violation before changing the model: verify units, shape interpretation, dtype/storage, HBM config and whether this profiling export uses a different MatMulV2 path.
2. Build source-derived small-M MatMulV2/QuantBatchMatmulV3 path model from ops-nn tiling/kernel logic.
3. Continue QSFA only if exact tiling or block table/sparse indices can be recovered; otherwise mark remaining `~20%` as source-strategy residual.
4. Continue GMM only with real `groupList`/`tuningConfigOptional` or exact quant GMM tiling evidence.

## Architecture Understanding

- The repository is an Ascend profiling CSV kernel evaluator. Its target is interpretable kernel-aware estimation, not black-box fitting.
- Main CLI entry: `tools/eval_ops.py`, forwarding into `tools/op_eval/cli.py`.
- Library dispatch: `tools/op_eval/api.py`.
- Shared helpers:
  - `tools/op_eval/common.py`: config, dtype, shape, format, numeric helpers.
  - `tools/op_eval/profiling.py`: CSV discovery and report writing.
  - `tools/op_eval/types.py`: report container.
- MatMul package:
  - `tools/matmul_eval/common.py`: MatMul specs and shape/layout inference.
  - `tools/matmul_eval/kernel_model.py`: runtime KB, advanced tiling heuristic, analytic fallback, ideal bounds.
  - `tools/matmul_eval/quant_model.py`: quantized MatMul and ND2NZ helpers.
  - `tools/matmul_eval/gmm_model.py`: GroupedMatmul routing bounds.
  - `tools/matmul_eval/evaluator.py`: profiling row evaluation and report fields.
- Attention package:
  - `tools/attention_eval/common.py`: Attention row detection and Q/K/V shape inference.
  - `tools/attention_eval/tiling_replay.py`: source-strategy replay.
  - `tools/attention_eval/api.py`: current-kernel and ideal-bound cost estimates.
  - `tools/attention_eval/evaluator.py`: profiling evaluation and summaries.

## Tiling Semantics

- Actual kernel tiling should not be modeled as a free search when ops-nn or ops-transformer provides deterministic op/host tiling information.
- Real/op-derived tiling sources are preferred:
  - `runtime_kb_exact`: MatMulV3 runtime knowledge-base preset matched by key.
  - `advanced_tiling_heuristic`: reconstruction of ops-nn advanced tiling logic.
  - `ops_transformer_source_strategy_replay`: Attention source strategy identification, not exact binary tiling data.
- `analytic_search` is only a fallback for missing tiling information or unsupported/older operators, not evidence of actual kernel tiling.
- Reports should keep semantics separate:
  - `actual_tiling`: runtime KB, op_tiling, advanced tiling, or source replay paths.
  - `fallback_tiling`: analytic search used when actual tiling is unknown.
  - `optimal_tiling`: theoretical best-kernel/physical lower-bound comparison.
- `estimated_us` represents current-kernel estimate. `ideal_lower_bound_us` is a physical lower-bound reference and must not be treated as current-kernel time.

## Attention Evaluator State

- `tools/attention_eval` supports Q/K/V profiling-shape parsing, public `estimate_attention(...)`, profiling evaluation, CSV rows, and summaries.
- `tools/op_eval` supports `op_kind="attention"` and CLI `--op-kind attention`.
- With local `ops-transformer-master`, attention reports mark rows as `actual_tiling_source=ops_transformer_source_strategy_replay` and emit strategy/source-file fields.
- Without source, attention falls back to `actual_tiling_source=unavailable_ops_transformer_replay`.
- Source-strategy replay maps profiling Type/shape to ops-transformer strategy families: `FlashAttentionScore`, `FusedInferAttentionScore`, `PromptFlashAttention`, `IncreFlashAttention`, and paged/decode-like paths.
- Current attention estimate keeps `ideal_lower_bound_us` separate. `estimated_us` includes:
  - source-visible tile constants such as `Q_TILE_CEIL=128` and `MAX_KV_STACK_LEN=512`;
  - occupancy efficiency;
  - traffic amplification;
  - workspace score traffic;
  - sync overhead;
  - latency floors;
  - template overhead factors.
- Optional-input traffic capping was added so static metadata/mask/aux shapes do not dominate GM traffic.
- Per-SoC and per-kernel-family latency/template terms are split for FlashAttention vs FusedInferAttention and short/long prefill.
- Latest validation notes in docs:
  - 910B4 attention max relative error around `0.2946`, p95 `0.2380`, median `0.0965`.
  - 910C attention max relative error around `0.1155`, p95 `0.0968`, median `0.0263`.
- Current residuals:
  - 910B4 `FusedInferAttentionScore` decode with `q_seq=1`, `kv_seq=288`, and large/custom `head_dim` remains an unsupported template path until exact host tiling/template replay is added.
  - ds3.2 `KvQuantSparseFlashAttention` is the largest attention gap and needs a dedicated model from `ops-transformer-master/attention/kv_quant_sparse_flash_attention`.

## Profiling Sample Mapping

- `example_profilings/910B4/` contains base 910B4 profiling samples:
  - `kernel_details3.csv`: longcat-like 910B4 sample.
  - `kernel_details_gemma4.csv`: gemma 910B4 sample.
  - `kernel_details_hyimage.csv`: hyimage 910B4 sample.
  - `kernel_details_2.csv`: quant MatMul 910B4 sample.
  - `kernel_details_pangu.csv`: pangu MatMulV3 910B4 sample.
  - `kernel_details.csv`: small base sample with BatchMatMulV2.
- `example_profilings/910C/` contains base 910C samples:
  - `kernel_details4.csv`: 910C FusedInferAttention + MatMul/GMM.
  - `kernel_details5.csv`: 910C FlashAttention + MatMul/GMM.
- `example_profilings/profiling_with_model_code/` contains model-level samples:
  - `ds3.2`: 910C model-level sample.
  - `gemma`: 910B4 model-level sample.
  - `longcat`: 910B4 model-level sample.
  - `qwen7b`: qwen3/qwen7b sample with missing `Block Dim` fields; platform inferred from `Block Num`.
- Added `example_profilings/README.md` documenting this mapping and the caveats.

## Platform Inference Rules

- For Ascend/CANN profiling, platform inference must distinguish Cube and Vector block dimensions.
- Cube-like operators should be compared with `aic_num`: `MatMul`, `BatchMatMul`, `GroupedMatmul`, `QuantBatchMatmul`, and Cube-heavy FA paths.
- Vector-like operators should be compared with `aiv_num`: `Cast`, `Transpose`, `RotaryMul`, `Gather/Scatter`, activation, routing, and many fusion ops.
- 910B4 evidence:
  - Cube max around `20`.
  - Vector max around `40`.
- 910C/A3 evidence:
  - Cube max around `24`.
  - Vector max around `48`.
- Do not use full-file max `Block Dim`/`Block Num` directly, because Vector, communication, or malformed export fields can dominate the global maximum.

Current model-level platform conclusions:

- `ds3.2`: Cube max `24`, Vector max `48`; use 910C/A3 config.
- `gemma`: Cube max `20`, Vector max `40`; use 910B4 config.
- `longcat`: Cube max `20`, Vector max `40`; use 910B4 config.
- `qwen3-7b` / `qwen7b`: original `kernel_details.csv` lacks `Block Dim`/`Mix Block Dim` but has `Block Num`/`Mix Block Num`; `MatMulV2` rows have `Block Num` mostly `20` and some `19`, `FusedInferAttentionScore` rows have `Block Num=16` and `Mix Block Num=32`, and Vector-like rows reach `Block Num=40`. By BlockDim/BlockNum principle this maps to 910B4, not 910C.

## Current Evaluation Findings

- `eval_results/LATEST` previously pointed to a partial GMM-only summary under commit `17ddb4e`.
- Current working tree HEAD observed during review was `4e820be`; therefore old root-level CSV reports without commit-scoped directory were ambiguous.
- Old root-level CSV files under `eval_results/` were cleaned because they lacked stable commit correspondence.
- `eval_results/README.md` now states that future results should live under `<UTC时间>_<commit>/` subdirectories.
- Remaining committed/kept evaluation snapshot:
  - `eval_results/20260522T043350Z_17ddb4e/eval_summary.csv`
  - `eval_results/20260522T043350Z_17ddb4e/metadata.txt`
  - `eval_results/LATEST`
- Historical pre-cleanup precision observations retained as background in `eval_results/README.md`:
  - 910C attention was strongest: max about `11.6%`, p95 about `9.9%`.
  - 910B4 attention main tail was FIA decode.
  - 910C MatMul main tail was `MatMulV2` small-M long-K/N decode-like path, max about `54%`.
  - ds3.2 `KvQuantSparseFlashAttention` had very large error and lower-bound violations.
  - qwen3/qwen7b `MatMulV2` under 910B4 inferred config had all 483 rows with `ideal_lower_bound_us > duration_us`; platform inference still points to 910B4, but model/HBM/storage/path assumptions are not reliable for calibration.

## Current MatMul/GMM/FA Tail Findings

- `GroupedMatmul` logical-shape parsing treats the first dimension of `FRACTAL_NZ` weight as expert count, not a regular batch dimension.
- Logical FLOPs use total routed tokens rather than `expert_count * tokens`.
- Independent `grouped_matmul` entry keeps ordinary MatMul compatibility fields but adds routing scenario fields for balanced experts and extreme load imbalance.
- Without real `group_list`, GMM is an explanatory bound model, not exact replay.
- GMM source location is `ops-transformer/gmm/grouped_matmul`; not `ops-nn/matmul`.
- Source review supports groupList iteration, empty-group skip, and `count % coreNum` scheduling.
- `Block Dim` cannot infer active experts.
- Current GMM validation status:
  - Gemma large/occupied rows are all within routing bounds.
  - Longcat has above-bound rows; latest historical numbers vary depending on report generation version, but this remains a low-confidence residual.
  - DS3.2 INT8 GMM remains above-bound by about `18%` in historical summaries.
- `ds3.2` MatMul remaining tail includes small/tiny shapes where analytic lower bound is far below observed kernel latency, pointing to launch/template/minimum-execution overhead.
- `ds3.2` FA tail is `KvQuantSparseFlashAttention` with specialized sparse/quant behavior; needs dedicated source/tiling modeling.
- `gemma`, `qwen3/qwen7b`, and `longcat` ordinary FA tails are mostly decode/short-prefill fixed overhead and are smaller than the specialized `KvQuantSparseFlashAttention` tail.

## Documentation And Repository State Updates

- Added `example_profilings/README.md` with sample/report mapping, platform caveats, and qwen/qwen3 inference notes.
- Updated `eval_results/README.md` with report retention policy, historical precision background, main conclusions, and unresolved categories.
- Changed `.gitignore` to ignore `example_profilings/*` while allowing `example_profilings/README.md` to be tracked.
- Existing `.gitignore` also contains `/ops-nn`; that change was present during this session and was preserved.
- Cleaned old root-level CSV files from `eval_results/`, keeping only commit-scoped snapshot files and metadata.

## Open Priorities

- Regenerate a complete, commit-scoped evaluation snapshot for current HEAD using matched SoC configs.
- Build a dedicated `KvQuantSparseFlashAttention` evaluator from `ops-transformer-master/attention/kv_quant_sparse_flash_attention`.
- Investigate qwen3/qwen7b `MatMulV2` lower-bound violations under 910B4: platform inference says 910B4, but HBM bandwidth, profiling units, shape/storage interpretation, or MatMulV2 kernel path assumptions are inconsistent.
- Refine `MatMulV2 M=1` long-K/N path from source/tiling, not per-shape fitting.
- Continue GMM convergence only with real `groupList`, `groupListType`, `tuningConfigOptional`, or exact quant/GMM tiling evidence.
