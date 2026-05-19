# Kernel Evaluation

This repository contains a small evaluator for estimating Ascend operator kernel cost from exported profiling CSV files. MatMul is the first implemented operator family.

## Scope

- Hardware targets: Ascend 910B4 and Ascend 910C.
- Profiling inputs: `example_profilings/910B4` and `example_profilings/910C`.
- Main tool: `tools/eval_ops.py`.
- Design notes: `matmul_eval_design.md` and `matmul_eval_design_zh.md`.

## Information Sources

- Hardware note: `info.md`.
- Profiling samples: `example_profilings`.
- Local ops-nn MatMulV3 source snapshot:
  - `ops-nn-master-matmul-mat_mul_v3`
  - `ops-nn-master-matmul-batch_mat_mul_v3`
- Upstream ops-nn matmul directory:
  - https://gitcode.com/cann/ops-nn/tree/master/matmul
  - Verified git commit during analysis: `f84b7eb83ee9f15df633d9fa9ca676bda83e11a0`.

The local ops-nn kernel code was taken from the upstream GitCode `cann/ops-nn` matmul tree. If the local snapshot and upstream diverge, prefer checking the exact upstream commit and then updating local source references deliberately.

## Tool Layout

- `tools/eval_ops.py`: generic operator profiling CLI entry point.
- `tools/op_eval/common.py`: shared config, dtype, format, shape, and numeric helpers.
- `tools/op_eval/profiling.py`: profiling CSV file discovery, default row filtering, and CSV report writing.
- `tools/op_eval/api.py`: generic `estimate_op(...)` dispatcher for future operator families.
- `tools/op_eval/cli.py`: shared CLI orchestration. It currently dispatches to MatMul evaluation.
- `tools/matmul_eval/api.py`: public MatMul cost API, including `estimate_matmul(...)`.
- `tools/matmul_eval/common.py`: MatMul specs, tile result structs, and MatMul shape inference.
- `tools/matmul_eval/kernel_model.py`: MatMul tiling/runtime_kb/advanced_tiling/Stream-K/full-load model.
- `tools/matmul_eval/quant_model.py`: low-bit MatMul mode, granularity, dequant, and traffic model.
- `tools/matmul_eval/evaluator.py`: MatMul profiling-row evaluation, summaries, and calibration suggestions.

## Useful Upstream Operators

The upstream `matmul` tree contains additional operator implementations that can refine this evaluator beyond plain MatMulV3:

- `mat_mul_v3` and `batch_mat_mul_v3`: primary host tiling, runtime_kb, advanced tiling, Stream-K, full-load, and kernel template references.
- `quant_batch_matmul_v3` and `quant_batch_matmul_v4`: low-bit matmul, per-token/per-channel/per-group/per-block, FP8/INT4/INT8, and dequant-related modeling references.
- `weight_quant_batch_matmul_v2`: weight-only quantization, Weight-NZ handling, split-K, fixpipe, and low-bit weight packing references.
- `fused_mat_mul` and `fused_quant_mat_mul`: fused epilogue overhead, activation/bias fusion, and fused quant matmul references.
- `transpose_batch_mat_mul`: transpose/einsum-style batch matmul tiling and layout handling references.
- `matmul_compress`, `convert_weight_to_int4_pack`, and `rotate_quant`: preprocessing and compressed/packed weight path references.

## Example Commands

Single-kernel API:

```bash
PYTHONPATH=tools python3 -c 'from matmul_eval import estimate_matmul; r = estimate_matmul(1024, 4096, 4096, "DT_BF16", config_path="configs/ascend_910c.json"); print(r.flops_cost_us, r.memory_access_us, r.total_us, r.bound_type)'
```

Generic operator API:

```bash
PYTHONPATH=tools python3 -c 'from op_eval import estimate_op; r = estimate_op("MatMulV3", 1024, 4096, 4096, "DT_BF16", config_path="configs/ascend_910c.json"); print(r.to_dict())'
```

The stable public cost fields are `flops_cost_us`, `memory_access_us`, `total_us`, and `bound_type`. `bound_type` is the end-to-end dominant bound (`compute_bound`, `memory_access_bound`, `launch_bound`, `format_bound`, or `balanced_bound`); `kernel_bound_type` keeps the compute-vs-memory kernel-only classification.

Batch profiling API:

```bash
PYTHONPATH=tools python3 -c 'from op_eval import evaluate_profiling; r = evaluate_profiling("example_profilings/910B4", config_path="configs/ascend_910b4.json"); print(r.resolved_count, r.unresolved_count)'
```

`evaluate_profiling(...)` is the library entry point used by the CLI and intended for upper-layer whole-network evaluators. It returns a `ProfilingEvaluation` object with `rows`, `unresolved`, `resolved_count`, `unresolved_count`, and `to_dict()`.

910B4:

```bash
python3 tools/eval_ops.py \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report_910b4.csv \
  --unresolved-output matmul_eval_unresolved_910b4.csv
```

910C:

```bash
python3 tools/eval_ops.py \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output matmul_eval_report_910c.csv \
  --unresolved-output matmul_eval_unresolved_910c.csv
```
