# Session Notes

## Tiling Semantics

- Actual kernel tiling should not be modeled as a free search when ops-nn provides deterministic op_tiling information.
- The evaluator should prioritize real/op-derived tiling sources:
  - `runtime_kb_exact`: MatMulV3 runtime_kb preset tiling matched by key.
  - `advanced_tiling_replay` or current `advanced_tiling_heuristic`: replay/reconstruction of ops-nn advanced_tiling logic.
- `analytic_search` is only a fallback for missing tiling information or unsupported/older operators, not evidence of the actual kernel tiling.
- The report should split tiling semantics explicitly:
  - `actual_tiling`: runtime_kb/op_tiling/advanced_tiling replay paths.
  - `fallback_tiling`: analytic search used when actual tiling is unknown.
  - `optimal_tiling`: theoretical best-kernel bound search, separated from current-kernel estimation.
- Future reports should expose whether the current-kernel estimate is based on actual tiling or fallback tiling so the confidence level is clear.

## TODO
- Refine matmul-like operator tiling. By default, estimation should use tiling options from https://gitcode.com/cann/ops-nn/tree/master/matmul

- Continue refining attention-family operator costs from the `cann/ops-transformer` kernel implementations.
- Source: https://gitcode.com/cann/ops-transformer
- Target operators should include major attention variants implemented there, such as FlashAttention/paged attention/incremental attention/prompt attention if present in the kernel tree.
- Keep the same modeling split used for MatMul: actual op_tiling or kernel replay first, analytic fallback only when real tiling is unavailable, and a separate optimal-kernel bound for comparison.
- Shared parsing/reporting should live under `tools/op_eval`; attention-specific specs, tiling replay, and kernel cost models should live in `tools/attention_eval` rather than under `tools/matmul_eval`.

## Attention Evaluator Progress

- Added `tools/attention_eval` with Q/K/V profiling-shape parsing, public `estimate_attention(...)`, profiling evaluation, CSV report rows, and summaries.
- `tools/op_eval` now supports `op_kind="attention"` and CLI `--op-kind attention`.
- Current attention model estimates QK/PV FLOPs, softmax/vector work, and minimum HBM bytes. With local `ops-transformer-master` present, it marks rows as `actual_tiling_source=ops_transformer_source_strategy_replay` and emits strategy/source-file fields; without source it falls back to `actual_tiling_source=unavailable_ops_transformer_replay`.
- Existing samples contain `FusedInferAttentionScore` rows in both 910B4 and 910C profiling directories, so attention parsing can be validated before ops-transformer source replay is implemented.
- Downloaded local `cann/ops-transformer` source snapshot to `ops-transformer-master`.
- Added source-strategy replay for attention families by mapping profiling Type/shape to ops-transformer source strategies: FlashAttentionScore, FusedInferAttentionScore, PromptFlashAttention, IncreFlashAttention, and paged/decode-like paths.
- Added 910B/910C-aware current-kernel attention estimate. It keeps `ideal_lower_bound_us` as the physical lower bound, while `estimated_us` now includes source-visible tile constants (`Q_TILE_CEIL=128`, `MAX_KV_STACK_LEN=512`), occupancy efficiency, traffic amplification, workspace score traffic, sync overhead, latency floors, and template overhead factors.
- Changed kernel evaluation validation to prioritize maximum relative error, not median error. Added `.agents/skills/kernel-eval-iteration/` with a generic iteration workflow and analyzer script for all operator families.
- Added optional-input traffic capping for attention auxiliary/mask tensors so static metadata shapes do not dominate GM traffic.
- Added per-SoC and per-kernel-family attention latency/template terms: decode floors are split for FlashAttention vs FusedInferAttention, and short/long prefill template factors are split by operator family.
- Current validation: `attention_eval_report_910b4.csv` has max relative error `0.2946`, p95 `0.2380`, median `0.0965`; `attention_eval_report_910c.csv` has max relative error `0.1155`, p95 `0.0968`, median `0.0263`.
- Current accepted residual: 910B4 `FusedInferAttentionScore` decode with `q_seq=1`, `kv_seq=288`, `head_dim=8192` remains under-estimated. This is not an uninspected outlier; it is classified as an unsupported custom-head-dim incremental-attention template path until exact host tiling/template replay is added.
- Next refinement should replay exact host tiling data from the C++ op_host contexts where possible; current attention replay deliberately does not claim exact binary tiling data.

## Profiling With Model Code Platform Check

- For Ascend/CANN profiling, platform inference should distinguish Cube and Vector block dimensions.
- Cube-like operators (`MatMul`, `BatchMatMul`, `GroupedMatmul`, `QuantBatchMatmul`, Cube-heavy FA paths) should be compared with `aic_num`.
- Vector-like operators (`Cast`, `Transpose`, `RotaryMul`, `Gather/Scatter`, activation, routing, fusion) should be compared with `aiv_num`.
- Current新增网络判断：
  - `ds3.2`: Cube max block dim `24`, Vector max block dim `48`, use `910C/A3`.
  - `gemma`: Cube max block dim `20`, Vector max block dim `40`, use `910B4`.
  - `qwen7b`: Cube max block dim `20`, Vector max block dim `40`, use `910B4`; previous 910C inference from residual fitting is not reliable.
  - `longcat`: Cube max block dim `20`, Vector max block dim `40`, use `910B4`.

## Current MatMul/FA Tail Findings

- Added GroupedMatmul-specific logical-shape parsing: the first dimension of `FRACTAL_NZ` weight is expert count, not a regular batch dimension. Logical FLOPs now use total routed tokens rather than `expert_count * tokens`.
- Added independent `grouped_matmul` evaluation entry. The report keeps ordinary MatMul compatibility fields but adds routing scenario fields for balanced experts and extreme load imbalance. Without `group_list`, this is an explanatory bound model, not an exact replay.
- Corrected GMM source location to `ops-transformer/gmm/grouped_matmul`. Source review shows kernel-side groupList iteration, empty-group skip, and `count % coreNum` scheduling; `Block Dim` cannot infer active experts.
- Current GMM validation after source-visible scheduler update: Gemma large/occupied rows are all within routing bounds; Longcat max bound error is about 18%; DS3.2 INT8 GMM is above by about 18%. These above-bound rows remain low confidence until real `groupList`, `groupListType`, `tuningConfigOptional`, sync/merge, or exact quant adaptive sliding-window tiling details are available.
- `ds3.2` matmul largest remaining tail is tiny `MatMul M=4,N=128,K=128`, where the analytic lower bound is below observed kernel latency by two orders of magnitude. This points to small-kernel launch/template/minimum-execution overhead; update only with source-visible template or launch evidence.
- `ds3.2` FA tail is `KvQuantSparseFlashAttention` with `q_seq=128, kv_seq=1, head_dim=576`, currently over-estimated and marked as specialized low-confidence path. Need dedicated `kv_quant_sparse_flash_attention` source/tiling modeling rather than generic FA cost.
- `gemma`, `qwen7b`, and `longcat` ordinary FA tails are mostly decode/short-prefill fixed overhead and are materially smaller than the specialized `KvQuantSparseFlashAttention` tail.
