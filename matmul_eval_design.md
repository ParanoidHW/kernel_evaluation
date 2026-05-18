# MatMul Evaluation Tool Design

## Scope

This tool evaluates Ascend matmul kernels from exported profiling CSV files. The current implementation targets:

- Hardware configs: Ascend 910B4 and Ascend 910C.
- 910B4 HBM bandwidth: 0.8 TB/s.
- 910C HBM bandwidth: 1.6 TB/s.
- 910C BF16/FP16 peak: 400 TFLOPS for the visible device.
- HF32: disabled / not used.
- Included by default: `MatMul`, `MatMulV2`, `MatMulV3`, `BatchMatMulV2`.
- Excluded by default: `GroupedMatmul` and `AllGatherMatmul`.

`GroupedMatmul` is excluded because grouped expert weights are not equivalent to a regular single GEMM or batch GEMM. `AllGatherMatmul` is excluded because communication can dominate the measured duration.

## Model Philosophy

The evaluator does not interpolate over historical profiling samples. It uses a kernel-aware semi-analytic model:

1. Parse profiling rows and infer the logical GEMM problem.
2. Interpret storage formats, including `FRACTAL_NZ`, using the same shape rules seen in the MatMulV3 tiling code.
3. Reconstruct deterministic tiling, alignment, storage-padding, and core-utilization effects from shapes, dtype, cache sizes, and AI Core count.
4. Compute lower bounds from compute throughput and HBM bandwidth.
5. Keep truly unobservable terms as a small set of global calibration parameters.

This keeps the output explainable and avoids fitting shape-specific coefficients.

## Hardware Configuration

Configuration is stored per hardware target:

- `configs/ascend_910b4.json`: 20 AI Cores, 0.8 TB/s HBM, no-HF32 BF16/FP16 peak 240 TFLOPS.
- `configs/ascend_910c.json`: 24 AI Cores inferred from the 910C profiling block dim, 1.6 TB/s HBM, no-HF32 BF16/FP16 peak 400 TFLOPS.

Known or user-supplied fields:

- `aic_num`: AI Core count used for tile/core-efficiency modeling.
- `aiv_num`: Vector core count.
- `hbm_bandwidth_tbps`: HBM bandwidth in TB/s.

Configurable assumptions:

- `l0a_bytes`, `l0b_bytes`, `l0c_bytes`, `l1_bytes`, `l2_bytes`, `ub_bytes`: cache/buffer sizes.
- `peak_tflops`: dtype-specific no-HF32 peak throughput.
- `calibration`: global launch overhead, pipeline efficiency, and optional format-conversion overhead.

These fields should be updated if reliable CANN platform configuration or official target-specific numbers become available.

## Profiling Extraction

For each included matmul row, the evaluator extracts:

- Identity: source file, CSV line, `Name`, `Type`.
- Timing: `Duration(us)`, `aicore_time(us)`, `aic_mac_time(us)`.
- Counters: `Block Dim`, `Mix Block Dim`, `cube_utilization(%)`, AIC MAC/MTE/Fixpipe ratios.
- Spec inputs: `Input Shapes`, `Output Shapes`, `Input Formats`, `Output Formats`, input/output dtypes.
- Inferred GEMM: `M`, `N`, `K`, `batch`, `transA`, `transB`.
- Storage metadata: normalized A/B/output formats and physical storage element counts.

Rows are included by operator `Type`, not by `Name`. This intentionally excludes rows such as `Type=Mul` even if the profiling name contains `Matmul`.

The parser does not assume B is always `[K,N]`. It searches `transA/transB` candidates and scores them against output shape.

## Shape And Format Inference

Formats are normalized as:

```text
FRACTAL_NZ or NZ -> FRACTAL_NZ
all other formats -> ND
```

This follows the MatMulV3/BatchMatMulV3 kernel behavior where compile-time format macros map `FORMAT_FRACTAL_NZ` to `CubeFormat::NZ`; all other formats are treated as `CubeFormat::ND`.

For ND storage:

```text
matrix_dim0 = shape[-2]
matrix_dim1 = shape[-1]
batch_dims = shape[:-2]
```

For `FRACTAL_NZ` storage, the evaluator follows the host tiling `GetInputDims` rule:

```text
matrix_dim0 = storage[-3] * storage[-2]
matrix_dim1 = storage[-4] * storage[-1]
batch_dims = storage[:-4]
```

Candidate inference:

- A candidate is created for each `transA/transB` pair.
- K dimensions must match exactly, or match after a 16-element alignment on a `FRACTAL_NZ` side.
- Output M/N must match the output shape exactly, or match after a 16-element alignment for `FRACTAL_NZ` output.
- The best-scoring candidate is selected, with a small preference for non-transposed layouts.

## FRACTAL_NZ / Weight-NZ Handling

`FRACTAL_NZ` is treated as a storage layout, not as a fitted timing class.

Example from the current profiling set:

```text
x1 shape    = [1, 2816], format ND
x2 shape    = [2048, 176, 16, 16], format FRACTAL_NZ
out shape   = [1, 32768], format ND
x2 logical  = [176 * 16, 2048 * 16] = [2816, 32768]
GEMM spec   = M=1, N=32768, K=2816, batch=1
```

Physical storage elements are computed from storage shapes:

```text
A_storage_elements = product(A_storage_shape)
B_storage_elements = product(B_storage_shape)
C_storage_elements = product(C_storage_shape)
```

This means prepacked Weight-NZ traffic is charged directly as physical bytes. Prepacked `FRACTAL_NZ` does not add `format_overhead_us` by itself. The extra conversion term is reserved for runtime ND-to-NZ paths.

For the three previously unresolved `ND;FRACTAL_NZ` rows, the model now resolves all of them as `M=1, N=32768, K=2816`. Their dominant term is the HBM read of the large prepacked B matrix, not compute.

## Runtime ND2NZ Detection

Runtime ND-to-NZ conversion is detected deterministically. The evaluator currently applies a MatMulV3-style approximation:

- Only ND operands can require runtime ND2NZ.
- If `inner_size * dtype_size` is in `{32, 64, 96, 128, 160, 192, 224, 256, 384}`, the operand is considered supported by GM-to-L0 on-the-way conversion.
- Otherwise, normal ND2NZ can be triggered by non-256B-aligned inner dimensions or `inner_size > 65535`, with FP32 no-HF32 exceptions.
- VNCHW-style ND2NZ can be triggered for large outer dimensions and small unaligned inner dimensions.

The flags are emitted as `nd2nz_a` and `nd2nz_b`. If either is true:

```text
format_overhead_us = count(nd2nz operands) * calibration.format_overhead_us.ND2NZ
```

The default `ND2NZ` overhead is zero until a global value is provided. `calibration.format_overhead_us.FRACTAL_NZ` remains in the config for compatibility, but current code does not charge it for already-prepacked NZ inputs.

## Analytic Cost Components

For a resolved GEMM:

```text
true_flops = 2 * M * N * K * batch
aligned_flops = 2 * aligned_M * aligned_N * aligned_K * batch
compute_us = aligned_flops / (peak_tflops * 1e6 * core_eff)

gm_bytes_min =
  A_storage_elements * input_dtype_size +
  B_storage_elements * input_dtype_size +
  C_storage_elements * output_dtype_size

gm_bytes_tiled_raw =
  tile_N * A_storage_bytes +
  tile_M * B_storage_bytes +
  C_storage_bytes

hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1e6)
lower_bound_us = max(compute_us, hbm_us)
estimated_us = launch_overhead_us + max(compute_us / pipeline_efficiency, hbm_us) + format_overhead_us
```

`storage_padding_ratio` is:

```text
physical_storage_elements / logical_storage_elements
```

It highlights storage layout or padding expansion.

## Tiling Approximation

The tiling search is deterministic:

- Candidate `baseM` and `baseN`: multiples of 16, plus preferred values such as 64, 80, 96, 128, 192, 256, 320, 336.
- Maximum candidate extent is capped at 512.
- L0C single-buffer constraint: `baseM * baseN * 4 <= l0c_bytes`.
- `db_l0c=2` if double-buffered L0C also fits; otherwise `db_l0c=1`.
- `baseK` is derived from L0A/L0B capacity and aligned to 16.
- Candidate score is the physical lower bound `max(compute_us, hbm_us)`.

Core efficiency is estimated from the number of M/N/batch tiles:

```text
mn_tile_count = ceil(M/baseM) * ceil(N/baseN) * batch
rounds = ceil(mn_tile_count / aic_num)
core_eff = mn_tile_count / (rounds * aic_num)
```

Tail efficiency is:

```text
tail_eff = true_flops / aligned_flops
```

HBM bytes are L2-aware:

- If `gm_bytes_min <= l2_bytes`, repeated tile reads are treated as L2 hits and `gm_bytes_min` is used.
- Otherwise, deterministic L2 pressure is applied to a portion of `gm_bytes_tiled_raw - gm_bytes_min`.

## Quant Matmul Handling

Quantized matmul rows use a dedicated path instead of the normal floating-point model. A row is considered quantized if the kernel type contains `Quant` or if A/B input dtypes are low-bit data types such as `INT8`, `INT4`, or `MXFP8`.

The evaluator infers:

- `quant_mode`: `int8`, `int4`, `mxfp8`, etc., from A/B dtypes.
- `quant_compute_path`: `full_quant`, `full_quant_with_dequant`, or `fake_quant_or_mixed`.
- `quant_granularity`: inferred from auxiliary scale tensor shapes.
- `quant_aux_bytes`: scale/offset auxiliary traffic.

Full quantization means A and B are low-bit tensors and the cube path is modeled with low-bit effective TOPS. If low-bit A/B are accompanied by floating scale tensors and the output is FP16/BF16/FP32, the path is marked as `full_quant_with_dequant`: integer accumulation is assumed on the main matmul path, followed by scale/dequant/output conversion. If only part of the inputs are low-bit or the spec is ambiguous, the path is marked as `fake_quant_or_mixed`.

Granularity is inferred from scale shapes:

- Scalar or `[1]`: `per_tensor`.
- One-dimensional scale equal to `N`: `per_channel_n`.
- One-dimensional scale equal to `M`: `per_token_m`.
- If `M == N` and scale is `[M] == [N]`, the result is `per_channel_n_or_per_token_m` because shape alone cannot disambiguate the axis.
- Shapes that divide `M` or `N` are marked as `per_group_or_block`.
- FP8/MXFP8 dtypes are marked through `quant_mode`; block-size details still require more metadata than the profiling CSV currently exposes.

Quant HBM bytes are recomputed with low-bit storage:

```text
quant_A_bytes = A_elements * bitwidth(A) / 8
quant_B_bytes = B_elements * bitwidth(B) / 8
quant_aux_bytes = sum(product(aux_shape) * aux_dtype_size)
quant_output_bytes = C_elements * output_dtype_size
```

Quant compute time uses `configs/ascend_910b4.json::quant_matmul`:

```text
quant_compute_us =
  aligned_flops * operation_factor /
  (peak_tops * 1e6 * core_eff * quant_pipeline_efficiency)
```

For the current `QuantBatchMatmulV3` samples, the inputs are `INT8;INT8;FLOAT;FLOAT`, output is `FLOAT16`, and auxiliary shapes are `[4096]` and `[4096]`. The model infers `int8`, `full_quant_with_dequant`, and `per_channel_n_or_per_token_m` because this shape has `M == N == 4096`.

## Diagnostics

The output includes diagnosis tags:

- `quant_matmul`: quantized matmul path is used.
- `full_quant_dequant`: low-bit matmul with floating output conversion.
- `fake_or_mixed_quant`: ambiguous or mixed quantization path.
- `weight_nz`: B is `FRACTAL_NZ`.
- `fractal_nz`: A or output is `FRACTAL_NZ`.
- `runtime_nd2nz`: runtime ND2NZ path is detected.
- `layout_padding`: physical storage elements exceed logical elements by more than 5%.
- `small_m_overhead`: `M <= 4`.
- `low_tile_count`: M/N/batch tile count is lower than AI Core count.
- `low_cube_utilization`: profiling cube utilization is below 80%.
- `compute_bound`, `memory_bound`, `balanced_bound`: analytic bound classification.
- `large_residual`: measured duration is more than 5x estimate.

Confidence is downgraded for very small M or low tile count because launch, scheduling, vector/fixpipe, and memory latency can dominate.

## Calibration

Only global terms should be calibrated:

- `launch_overhead_us_by_type`: global launch/scheduling overhead per kernel type.
- `pipeline_efficiency_by_dtype`: sustained fraction of dtype peak for large compute-heavy cases.
- `format_overhead_us.ND2NZ`: optional global cost for detected runtime ND2NZ conversion.
- `quant_matmul.peak_tops`: effective low-bit compute throughput.
- `quant_matmul.pipeline_efficiency`: global low-bit pipeline efficiency.
- `quant_matmul.operation_factor`: cost multiplier for full/fake quant paths.
- `quant_matmul.dequant_us_per_output_element`: optional dequant/output conversion term.

The helper suggestion logic is deliberately simple:

- Launch overhead is estimated from low-tile-count residuals with a low percentile.
- Pipeline efficiency is estimated from large, high-cube-utilization rows with a high percentile.

Launch overhead is already part of the estimate:

```text
estimated_us = launch_overhead_us + kernel_bound_us + format_overhead_us
```

Launch overhead fitting is global per kernel type, not shape-specific. The current helper uses low-tile-count rows and computes a low percentile of `duration_us - lower_bound_us`; this estimates an absolute launch/scheduling term and avoids absorbing cache misses or tail effects. Pipeline efficiency uses a high percentile of `achieved_tflops / peak_tflops` from large, high-cube-utilization rows.

The current checked-in calibration values are:

- 910B4 launch overhead: `BatchMatMul=2.35824us`, `MatMul=3.6392us`, `MatMulV2=3.1942us`.
- 910B4 sustained efficiency: `DT_BF16=0.922814`, `FLOAT16=0.976532`, `FLOAT=0.99724`, quant `INT8=0.522674`.
- 910C launch overhead: `MatMulV2=7.61184us`; `MatMulV3` remains `0.0` because the current 910C samples do not provide low-tile-count MatMulV3 calibration rows.
- 910C sustained efficiency: `DT_BF16=0.772807`, using the user-specified 400 TFLOPS peak.

Do not fit per-shape coefficients. If residuals are large, add an explainable term only when it maps to a kernel mechanism.

## Current Validation Snapshot

With the current 910B4 example profiling set:

```text
resolved_matmul_rows = 1255
unresolved_rows = 0
```

With the current 910C example profiling set and `configs/ascend_910c.json`:

```text
resolved_matmul_rows = 672
unresolved_rows = 0
MatMulV2 rows = 416
MatMulV3 rows = 256
MatMulV2 median actual / estimate ~= 1.05
MatMulV3 median actual / estimate ~= 1.03
```

The previous three unresolved rows were `ND;FRACTAL_NZ` weight-NZ rows. They now resolve to:

```text
M=1, N=32768, K=2816, batch=1
B_storage_elements = 2048 * 176 * 16 * 16 = 92274688
estimated_us ~= 234.4
measured_us ~= 246-248
diagnosis = weight_nz|small_m_overhead|memory_bound
```

Regression check: the previously resolved 990 rows keep the same `M/N/K/batch/transA/transB` after format-aware parsing.

The current low-bit profiling file contains 11 `QuantBatchMatmulV3` rows:

```text
Input dtypes = INT8;INT8;FLOAT;FLOAT
Output dtype = FLOAT16
M=4096, N=4096, K=12800
quant_mode = int8
quant_compute_path = full_quant_with_dequant
quant_granularity = per_channel_n_or_per_token_m
median actual / estimate ~= 1.00
median absolute percentage error ~= 1.0%
```

## CLI

Run one SoC directory with the matching config. Directory inputs are scanned recursively, so passing the top-level `example_profilings` directory would mix 910B4 and 910C rows under one hardware config.

910B4 report:

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report_910b4.csv \
  --unresolved-output matmul_eval_unresolved_910b4.csv
```

910C report:

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output matmul_eval_report_910c.csv \
  --unresolved-output matmul_eval_unresolved_910c.csv
```

Calibration suggestions should be generated against the matching SoC directory and config:

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --suggest-calibration \
  --calibration-output matmul_eval_calibration_suggested_910b4.json
```

Optional flags:

- `--include-gmm`: include `GroupedMatmul` rows.
- `--include-allgather`: include `AllGatherMatmul` rows.

## Known Limitations

- The model is not a CANN tiling implementation. It approximates kernel behavior with deterministic constraints.
- GMM is excluded by default and is not modeled as grouped expert GEMM.
- AllGatherMatmul communication is excluded by default.
- Cache sizes and peak TFLOPS are configurable assumptions unless official target-specific values are supplied.
- Runtime ND2NZ detection currently uses MatMulV3-style conditions for all included rows; BatchMatMulV3-specific multi-batch paths may need refinement if those rows become important.
- Quant matmul support is mode-aware but still relies on effective `peak_tops` and pipeline parameters. More quantization modes, especially MXFP8 and per-group variants, need additional profiling examples to validate.
- If profiling exports origin shapes instead of storage shapes for `FRACTAL_NZ`, the current NZ inference would need additional metadata.
