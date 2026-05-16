# MatMul Evaluation Tool Design

## Scope

This tool evaluates Ascend 910B4 matmul kernels from exported profiling CSV files. The first development scope is:

- Target hardware: Ascend 910B4.
- HBM bandwidth: 0.8 TB/s.
- HF32: disabled / not used.
- Included by default: `MatMul`, `MatMulV2`, `MatMulV3`, `BatchMatMulV2`.
- Excluded by default: `GroupedMatmul` and `AllGatherMatmul`.

`GroupedMatmul` is excluded because expert weights introduce grouped semantics that are not equivalent to regular batch matmul. `AllGatherMatmul` is excluded because communication can dominate the measured duration.

## Model Philosophy

The tool should not use interpolation over historical samples. It uses a kernel-aware semi-analytic model:

1. Parse profiling records and infer the actual GEMM problem.
2. Reconstruct deterministic tiling and alignment effects from shapes, dtype, cache sizes, and core count.
3. Compute physical lower bounds from compute peak and HBM bandwidth.
4. Keep truly unobservable terms as a small number of global calibration parameters.

This keeps the output explainable and avoids overfitting to a narrow profiling set.

## Hardware Configuration

Configuration is stored in `configs/ascend_910b4.json`. Some fields are known from the profiling and user input, while cache sizes and peak throughput can be updated when a reliable CANN platform config is available.

Required fields:

- `aic_num`: AI Core count. Current profiling shows `Block Dim=20` for AI_CORE matmul, so the default is 20.
- `aiv_num`: Vector core count. Default is 40.
- `hbm_bandwidth_tbps`: 0.8.
- `l0a_bytes`, `l0b_bytes`, `l0c_bytes`, `l1_bytes`, `l2_bytes`, `ub_bytes`: cache/buffer configuration.
- `peak_tflops`: dtype-specific peak throughput for no-HF32 operation.

The current first-pass cache assumptions are conservative and editable.

## Profiling Extraction

For each matmul row, extract:

- Identity: file, line, `Name`, `Type`.
- Kernel timing: `Duration(us)`, `aicore_time(us)`, `aic_mac_time(us)`.
- Hardware counters: `Block Dim`, `cube_utilization(%)`, AIC MTE/Fixpipe/MAC ratios.
- Spec: input/output shapes, dtype, format.
- Inferred GEMM: `M`, `N`, `K`, batch count, `transA`, `transB`.

The parser must not assume `B` is always `[K,N]`. Many rows store `B` as `[N,K]`; output shape is used to infer `transB=true`.

## Analytic Cost Components

For a resolved GEMM:

```text
true_flops = 2 * M * N * K * batch
aligned_flops = padded tile work from baseM/baseN/baseK
compute_us = aligned_flops / (peak_tflops * 1e6 * core_eff)
gm_bytes_min = bytes(A) + bytes(B) + bytes(C)
gm_bytes_tiled_raw = no-L2 tile repeated GM read estimate
gm_bytes_tiled = L2-aware effective HBM byte estimate
hbm_us = gm_bytes_tiled / (hbm_bandwidth_tbps * 1e6)
lower_bound_us = max(compute_us, hbm_us)
estimated_us = launch_overhead_us + max(compute_us / pipeline_efficiency, hbm_us) + format_overhead_us
```

`core_eff` is derived from tile count versus AI core count:

```text
tile_count = ceil(M/baseM) * ceil(N/baseN) * ceil(K/baseK) * batch
mn_tile_count = ceil(M/baseM) * ceil(N/baseN) * batch
core_eff = mn_tile_count / (ceil(mn_tile_count / aic_num) * aic_num)
```

`tail_eff` is:

```text
tail_eff = true_flops / aligned_flops
```

When `gm_bytes_min <= l2_bytes`, the model treats repeated tile reads as L2 hits and uses `gm_bytes_min` for the HBM bound. When the working set exceeds L2, the model adds a deterministic L2 pressure term instead of fitting a bandwidth multiplier.

## Tiling Approximation

The first implementation uses deterministic candidate search rather than fitted tile sizes:

- Candidate `baseM` and `baseN`: multiples of 16.
- `L0C` constraint: `baseM * baseN * fp32_size * dbL0C <= l0c_bytes`.
- `L0A/L0B` constraint determines maximum feasible `baseK`.
- Candidate score is the physical lower bound `max(compute_us, hbm_us)`.

This approximates the kernel behavior visible in MatMulV3 ASW tiling: base block selection, tail balancing, L0C double buffering, and L1/L0 pressure are handled by deterministic constraints rather than shape interpolation.

## FRACTAL_NZ / Weight-NZ Handling

`FRACTAL_NZ` is treated as a storage layout, not as a fitted timing class. This follows the MatMulV3/BatchMatMulV3 kernel logic:

- Kernel compile-time format macros map `FORMAT_FRACTAL_NZ` to `CubeFormat::NZ`; all other formats are treated as `CubeFormat::ND`.
- Host tiling reconstructs the effective 2D matrix for NZ storage as:

```text
effective_dim0 = storage[-3] * storage[-2]
effective_dim1 = storage[-4] * storage[-1]
```

For example, profiling input `x1=[1,2816]`, `x2=[2048,176,16,16]`, formats `ND;FRACTAL_NZ`, output `[1,32768]` is evaluated as:

```text
x2 effective = [176*16, 2048*16] = [2816, 32768]
M = 1, K = 2816, N = 32768
```

The evaluator uses storage-shape element counts for HBM traffic:

```text
A_bytes = product(A_storage_shape) * dtype_size
B_bytes = product(B_storage_shape) * dtype_size
C_bytes = product(C_storage_shape) * output_dtype_size
```

This captures layout padding and prepacked weight size directly. A prepacked `FRACTAL_NZ` weight does not by itself add conversion overhead. `format_overhead_us.ND2NZ` is reserved for runtime ND-to-NZ conversion paths detected from the kernel's deterministic conditions. For `ND;FRACTAL_NZ` weight-NZ matmul, the expected dominant term is often the HBM read of the large prepacked B matrix, especially for `M=1` or other small-M GEMV-like shapes.

## Calibration Parameters

Only these terms should be calibrated:

- `launch_overhead_us_by_type`: global launch/scheduling overhead per kernel type.
- `pipeline_efficiency_by_dtype`: global sustained fraction of dtype peak for large, compute-bound cases.
- `format_overhead_us`: extra cost for ND2NZ/FRACTAL_NZ/non-contiguous paths.

Do not fit per-shape coefficients. If residuals are large, prefer adding a new explainable term such as tail imbalance, format conversion, or communication-fusion exclusion.

## Output

The CLI outputs one row per resolved matmul with:

- Inferred spec: `M/N/K/batch/transA/transB/dtype`.
- Measured metrics: duration, achieved TFLOP/s, cube utilization.
- Model metrics: selected tile sizes, core efficiency, tail efficiency, compute bound, HBM bound, estimated time.
- Diagnostics: `compute_bound`, `memory_bound`, `small_m_or_low_tile_count`, `low_cube_utilization`, `large_residual`, etc.

## First CLI

Example:

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report.csv
```

The first version defaults to excluding GMM and AllGatherMatmul. Use flags only when explicitly analyzing those cases.

Optional global calibration suggestions:

```bash
python3 tools/eval_matmul.py \
  --profiling example_profilings \
  --config configs/ascend_910b4.json \
  --suggest-calibration \
  --calibration-output matmul_eval_calibration_suggested.json
```

The calibration output is a suggestion only. It estimates low-percentile launch overhead from low-tile-count residuals and high-percentile pipeline efficiency from large high-cube-utilization rows.
