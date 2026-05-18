# Kernel Evaluation

This repository contains a small evaluator for estimating Ascend matmul kernel cost from exported profiling CSV files.

## Scope

- Hardware targets: Ascend 910B4 and Ascend 910C.
- Profiling inputs: `example_profilings/910B4` and `example_profilings/910C`.
- Main tool: `tools/eval_matmul.py`.
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

## Useful Upstream Operators

The upstream `matmul` tree contains additional operator implementations that can refine this evaluator beyond plain MatMulV3:

- `mat_mul_v3` and `batch_mat_mul_v3`: primary host tiling, runtime_kb, advanced tiling, Stream-K, full-load, and kernel template references.
- `quant_batch_matmul_v3` and `quant_batch_matmul_v4`: low-bit matmul, per-token/per-channel/per-group/per-block, FP8/INT4/INT8, and dequant-related modeling references.
- `weight_quant_batch_matmul_v2`: weight-only quantization, Weight-NZ handling, split-K, fixpipe, and low-bit weight packing references.
- `fused_mat_mul` and `fused_quant_mat_mul`: fused epilogue overhead, activation/bias fusion, and fused quant matmul references.
- `transpose_batch_mat_mul`: transpose/einsum-style batch matmul tiling and layout handling references.
- `matmul_compress`, `convert_weight_to_int4_pack`, and `rotate_quant`: preprocessing and compressed/packed weight path references.

## Example Commands

910B4:

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report_910b4.csv \
  --unresolved-output matmul_eval_unresolved_910b4.csv
```

910C:

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output matmul_eval_report_910c.csv \
  --unresolved-output matmul_eval_unresolved_910c.csv
```
