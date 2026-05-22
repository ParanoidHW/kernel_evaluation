# Kernel 评估工具架构

本文是仓库架构入口。算子族设计文档：

- MatMul：`docs/matmul_eval_design_zh.md`
- GroupedMatmul：`docs/gmm_eval_design_zh.md`
- Attention：`docs/attention_kernel_eval_design.md`
- 当前评估差距：`docs/current_eval_gap_zh.md`
- 硬件信息补充：`docs/info.md`

## 目标

本仓库从昇腾 profiling CSV 中估计算子 kernel 耗时，并输出可解释的成本分量、tiling/source 来源、诊断标签和误差字段。设计目标不是对历史样本做黑盒拟合，而是基于以下信息建立 current-kernel 模型：

- profiling CSV 可见 shape、dtype、format、duration、Block Dim 和硬件 counter
- `ops-nn` / `ops-transformer` 中的 kernel、host tiling、模板和调度逻辑
- `configs/` 中的平台核数、HBM 带宽、cache/buffer、峰值吞吐和全局 current-kernel 参数

核心约束：

- `estimated_us` 表示当前 kernel 路径估计。
- `ideal_lower_bound_us` 表示物理下界参考。
- fallback 估计不能伪装成 actual tiling。
- 新增参数必须能解释为 kernel、tiling 或硬件机制，不允许 per-shape 拟合。

## 当前支持的算子族

### matmul

CLI：

```bash
python3 tools/eval_ops.py --op-kind matmul ...
```

覆盖普通 MatMul、BatchMatMul、TransposeBatchMatMul 和量化 MatMul。当前默认排除 `GroupedMatmul` 和 `AllGatherMatmul`，可通过 `--include-gmm` / `--include-allgather` 显式纳入兼容报告。

主要能力：

- ND / FRACTAL_NZ shape 与 storage 解析
- `transA/transB` 候选推断
- runtime KB exact tiling
- advanced tiling heuristic
- analytic fallback tiling
- L2-aware GM 流量估计
- ND2NZ 检测
- INT8/INT4/FP8/MXFP8 等量化路径
- small-M/N MatMulV2 current-kernel overhead
- calibration suggestion

### grouped_matmul

CLI：

```bash
python3 tools/eval_ops.py --op-kind grouped_matmul ...
```

只保留 `Type=GroupedMatmul` 的行。它复用 MatMul evaluator 的通用读取和报告字段，但精度口径由 `tools/matmul_eval/gmm_model.py` 的 routing bounds 决定。

主要能力：

- 把 weight 的 `FRACTAL_NZ` 首维解释为专家数，而不是 batch
- 输出 balanced routing 和 extreme imbalance 两个场景
- 输出 `gmm_bounds_min_us/gmm_bounds_max_us`
- 用 `gmm_duration_position` 判断实测是否落在可解释路由区间内

GMM 缺少真实 `group_list` 时不能用普通 MatMul 单点 lower bound 判断精度。

### attention

CLI：

```bash
python3 tools/eval_ops.py --op-kind attention ...
```

覆盖 FA/FIA/PFA/IFA/Paged/QSFA 等 attention 类 `Type`。当前 replay 是 `ops-transformer` source strategy replay，不是 exact host tiling replay。

主要能力：

- Q/K/V shape 解析
- decode/prefill、MQA/GQA、mask/aux、head_dim 策略标签
- QK/PV FLOPs、softmax/vector、最小 HBM 下界
- occupancy、traffic factor、workspace、sync、latency floor、template factor current-kernel 成本
- QSFA PA cache 专用 parser 和 workspace 模型

## 目录结构

```text
tools/eval_ops.py
  -> CLI 入口

tools/op_eval/
  -> 通用配置、profiling 文件发现、CSV 输出、API 分发

tools/matmul_eval/
  -> MatMul / Quant MatMul / GMM 兼容评估

tools/attention_eval/
  -> Attention parser、source replay、成本模型和报告

configs/
  -> 平台配置和模型参数

docs/
  -> 架构、算子族设计、误差分析、硬件备注

example_profilings/
  -> 实测 profiling 样本

eval_results/
  -> 基线刷新汇总
```

## 共享数据流

profiling 评估流程：

```text
tools/eval_ops.py
  -> op_eval.cli.run_cli()
  -> op_eval.api.evaluate_profiling(...)
  -> op_eval.profiling.iter_input_files(...)
  -> family evaluator 根据 Type 过滤行
  -> 解析 shape / format / dtype / counter
  -> 构造 logical spec
  -> tiling/source replay 或 fallback 模型
  -> compute / vector / HBM / launch / format / sync / template 成本
  -> resolved rows + unresolved rows
  -> CSV 输出和 summary
```

`evaluate_profiling(...)` 返回 `ProfilingEvaluation`：

- `op_kind`
- `rows`
- `unresolved`
- `resolved_count`
- `unresolved_count`
- `to_dict()`

## API 与 CLI 注册

共享入口在 `tools/op_eval/api.py`：

- `estimate_op(op_type, *args, **kwargs)`
- `evaluate_profiling(profiling, op_kind=..., config_path=...)`

当前 `op_kind`：

```text
matmul
grouped_matmul
attention
```

CLI 参数在 `tools/op_eval/cli.py`：

- `--profiling`
- `--config`
- `--op-kind`
- `--output`
- `--unresolved-output`
- `--suggest-calibration`
- `--calibration-output`
- `--include-gmm`
- `--include-allgather`

`--suggest-calibration` 当前只支持 matmul。

## actual / fallback / optimal 语义

所有算子族都必须区分三类语义：

```text
actual_tiling_source
fallback_tiling_source
optimal_tiling_source
```

### MatMul

```text
runtime_kb_exact            -> actual_tiling
advanced_tiling_heuristic   -> actual_tiling，但不是二进制 replay
analytic_search             -> fallback_tiling
physical_lower_bound        -> optimal_tiling_source
```

### GMM

GMM 没有 exact group routing，因为 profiling 缺少真实 `group_list`。当前主语义是 routing bounds：

```text
gmm_model_kind = grouped_matmul_routing_bounds
gmm_duration_position = within/below/above_gmm_bounds
```

普通 MatMul 字段保留用于兼容，但 GMM 精度判断应使用区间误差。

### Attention

```text
ops_transformer_source_strategy_replay -> source_strategy_replay
unavailable_ops_transformer_replay     -> fallback_tiling
physical_lower_bound                   -> optimal_tiling_source
```

Attention 当前没有 exact host tiling replay；source replay 只说明命中哪类源码策略和文件。

## 成本分量

共享概念：

- `compute_us`：Cube FLOPs 成本
- `vector_us`：AIV/vector 工作量，主要用于 Attention
- `hbm_us`：GM/HBM 流量成本
- `launch_overhead_us`：kernel 启动/调度固定开销
- `format_overhead_us`：运行时格式转换，当前主要是 MatMul ND2NZ
- `template_overhead_us`：MatMul 小 M/N current-kernel 模板/流水项
- `template_overhead_factor`：Attention 模板成本倍率
- `sync_overhead_us`：Attention split-fuse / KV tile 同步项
- `latency_floor_us`：decode、小序列或 QSFA 等 current-kernel 时延地板

字段语义：

- `estimated_us`：当前 kernel 总估计。
- `total_us`：与 `estimated_us` 保持一致。
- `ideal_lower_bound_us`：物理下界。
- `current_kernel_bound_us`：当前 tiling/current-kernel body 的理论下界。
- `residual_us = duration_us - estimated_us`
- `duration_over_estimate = duration_us / estimated_us`

## MatMul 内部链路

```text
matmul_eval.common
  -> infer_matmul_spec / infer_transpose_batch_matmul_spec / infer_grouped_matmul_spec

matmul_eval.runtime_kb
  -> load_runtime_kb

matmul_eval.kernel_model
  -> runtime_kb_exact
  -> advanced_tiling_heuristic
  -> analytic_search
  -> ideal_kernel_bounds

matmul_eval.quant_model
  -> infer_quant_spec
  -> infer_nd2nz_operands
  -> estimate_quant_cost

matmul_eval.gmm_model
  -> estimate_grouped_matmul_bounds

matmul_eval.api
  -> estimate_matmul_cost

matmul_eval.evaluator
  -> evaluate_file / print_summary / calibration_suggestions
```

## Attention 内部链路

```text
attention_eval.common
  -> is_attention_row
  -> infer_attention_spec
  -> _infer_kv_quant_sparse_attention_spec

attention_eval.tiling_replay
  -> replay_attention_tiling_strategy

attention_eval.api
  -> estimate_attention_cost
  -> _kernel_aware_components
  -> _kv_quant_sparse_workspace_bytes

attention_eval.evaluator
  -> evaluate_file / classify / print_summary
```

## 平台配置

当前配置文件：

- `configs/ascend_910b4.json`
- `configs/ascend_910b4_1.json`
- `configs/ascend_910c.json`

平台识别原则：

- Cube 类算子主要参考 `aic_num`，例如 MatMul、BatchMatMul、GroupedMatmul、QuantBatchMatmul 和 Cube-heavy attention。
- Vector 类算子主要参考 `aiv_num`，例如 Cast、Transpose、RotaryMul、Gather/Scatter、activation、normalization 和 routing。
- 910B4 常见证据：Cube block dim 约 20，Vector block dim 约 40。
- 910C/A3 常见证据：Cube block dim 约 24，Vector block dim 约 48。
- 不要用全文件最大 `Block Dim` 判断 MatMul/Attention 平台。

`Ascend910B4-1` 是 qwen3-7b/qwen7b 专用配置，使用 20 AIC、40 AIV、1.6 TB/s HBM。

## 报告与诊断

resolved CSV 行至少包含：

- 来源文件和行号
- 原始 `Type/Name`
- 推断后的逻辑规格
- profiling counter
- tiling/source/fallback 来源
- 成本分量
- 下界与 current-kernel 对比
- `estimated_us`
- `residual_us`
- `diagnosis`
- `confidence`

unresolved CSV 行保留：

- `file`
- `line`
- `type`
- `name`
- `input_shapes`
- `output_shapes`
- MatMul 额外保留 input/output formats

诊断标签由各算子族 evaluator 生成，不在共享层硬编码。

## 基线与误差分析

`eval_results/` 保存按时间和 commit 刷新的汇总结果：

```text
eval_results/<UTC>_<commit>/eval_summary.csv
eval_results/<UTC>_<commit>/metadata.txt
eval_results/LATEST
```

误差分析应优先看：

- 最大相对误差
- p95 相对误差
- lower-bound violation
- large/occupied rows
- GMM routing bound error
- diagnosis tail 分布

小于 10us 的 kernel 经常受 launch/调度噪声主导，通常不作为建模主目标。

## 扩展新算子族

新增算子族时应遵循：

1. 新建 `tools/<family>_eval/`。
2. 用 `Type` 过滤 profiling 行，不依赖 `Name` scope。
3. 定义 logical spec 和 parser。
4. 明确源码路径、tiling/source replay 能力和 fallback 边界。
5. 输出成本分量、source 标签、diagnosis 和 confidence。
6. 在 `tools/op_eval/api.py` 和 `tools/op_eval/cli.py` 注册。
7. 写入 `docs/<family>_eval_design_zh.md`。
8. 刷新基线并更新 `session.md`。
