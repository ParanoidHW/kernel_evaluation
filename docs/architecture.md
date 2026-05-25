# Kernel 评估工具架构

本文是仓库架构入口。算子族设计文档：

- MatMul：`docs/matmul_eval_design_zh.md`
- GroupedMatmul：`docs/gmm_eval_design_zh.md`
- Attention：`docs/attention_kernel_eval_design.md`
- 其他算子评估方案：本文“其他算子评估方案”章节
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

### other_ops（规划中）

后续新增 `other_ops` 评估入口，覆盖 profiling 中非 MatMul/GMM/Attention/通信类的常规算子。源码来源以当前新增目录为主：

- `ops-math`：elementwise、conversion、reduction、layout 和部分数学算子。
- `ops-cv`：resize、grid sample、ROI/NMS、图像采样和 CV 数据处理算子。
- 必要时继续引用 `ops-nn` 的 norm、activation、index/scatter 和 fusion 算子，以及 `ops-transformer-master` 的 MoE/routing 算子。

当前 profiling 中其他算子按总耗时和频次看，主要包括：

- layout/memory：`Transpose`、`TransData`、`TensorMove`、`Slice`、`StridedSliceD`、`AsStrided`、`ConcatD/ConcatV2D`、`SplitVD`、`Pack`
- elementwise/vector：`Add`、`Mul`、`Cast`、`Sub`、`Neg`、`RealDiv`、`Pows`、`Swish`、`Gelu`
- reduction/norm/activation：`ReduceSum`、`ReduceMean`、`RmsNorm`、`LayerNormV3`、`AddRmsNorm`、`InplaceAddRmsNorm`、`SwiGlu`
- index/scatter/routing：`GatherV2/V3`、`ScatterUpdate`、`ScatterNdUpdate`、`TopKV2`、`Moe*`
- CV/常规大 kernel：`Conv3DV2`、`GroupNormSilu`、`Resize*`、`GridSample*`、ROI/NMS 类

通信类 `Hcom*`、`hcom_*`、AICPU 通信辅助和 `AllGatherMatmul*` 不纳入 other_ops 首轮计算模型，先单独标记为 `unsupported_communication_or_aicpu`。

## 目录结构

```text
tools/eval_ops.py
  -> CLI 入口

tools/annotate_profiling.py
  -> 在单个 profiling CSV 原始行上追加 kernel_eval_value 列

tools/op_eval/
  -> 通用配置、profiling 文件发现、CSV 输出、API 分发

tools/matmul_eval/
  -> MatMul / Quant MatMul / GMM 兼容评估

tools/attention_eval/
  -> Attention parser、source replay、成本模型和报告

tools/other_ops_eval/      # 规划中
  -> 常规 vector/layout/reduction/CV 算子 parser、源码策略分类和成本模型

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
other_ops   # 规划中
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

另有轻量增列入口 `tools/annotate_profiling.py`：

- 输入单个 profiling CSV 和硬件配置 JSON。
- 输出保留原 CSV 所有列，并追加 `kernel_eval_value`。
- MatMul/Attention 行写入当前 `estimated_us`。
- GroupedMatmul 行默认写入 routing bounds 区间均值；可通过 `--gmm-value bounds` 改为输出 `[low,high]`。
- 非 MatMul/GMM/Attention 行当前留空；`other_ops` 实现后按对应算子族写入 `estimated_us`，通信/AICPU 类仍留空或写入 unsupported。

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

### Other Ops

首轮 `other_ops` 语义按源码可见程度分层：

```text
source_tiling_replay      -> 从 ops-math/ops-cv/ops-nn 源码还原 tiling 分支或关键常量
source_strategy_replay    -> 只能识别源码策略标签，无法还原 exact tiling data
analytic_fallback         -> 只有 shape/dtype/format，可做物理可解释 fallback
physical_lower_bound      -> ideal_lower_bound_us
```

对缺少 attrs 的算子，例如 `Transpose` 缺 permutation、`Slice` 缺 begin/size/stride、`Concat` 缺 axis 时，报告必须标记 `missing_runtime_attrs`，不能声称 exact replay。

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
- `layout_overhead_us`：layout/transpose/slice/concat 等非连续访问、重排或多段搬运开销
- `workspace_us`：workspace 读写或 partial merge 的 HBM 成本

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

## 其他算子评估方案

### 范围和排除规则

`other_ops` 首轮覆盖 profiling `Type` 中的常规 AIC/AIV 算子，排除：

- 已由专用评估器覆盖的 `MatMul*`、`GroupedMatmul`、Attention 类。
- 通信和通信辅助：`Hcom*`、`hcom_*`、`allreduceAicpuKernel`、`broadcastAicpuKernel`、`AllGatherMatmul*`。
- 明显缺少运行时语义且无法从 shape 还原的复杂融合算子，先写 unresolved。

首轮目标不是一次覆盖所有 Type，而是建立统一 parser、报告字段和可解释下界，再按优先级扩展源码 replay。

### 优先级和处理顺序

1. layout/memory 类：`Cast`、`Transpose`、`TransData`、`TensorMove`、`Slice`、`StridedSliceD`、`AsStrided`、`ConcatD/ConcatV2D`、`SplitVD`。
   - 原因：profiling 中总耗时高，主要受 GM/HBM、MTE、非连续访问和重排影响；源码主要在 `ops-math/conversion`，部分在 `ops-nn/index`。
   - 首轮只在 attrs 足够时做 source replay；缺 permutation/axis/stride 时做 analytic fallback 并标记低置信。
2. elementwise/vector 类：`Add`、`Mul`、`Sub`、`Neg`、`RealDiv`、`Pows`、`GreaterEqual`、`Less`、`ZerosLike`、`Fill`。
   - 原因：出现频次最高，模型结构简单，适合作为 `other_ops` parser 和 AIV/HBM 基线。
   - 成本以 input/output bytes、broadcast 后输出元素数和 vector op count 为主。
3. reduction/norm/activation 类：`ReduceSum`、`ReduceMean`、`RmsNorm`、`LayerNormV3`、`AddRmsNorm`、`InplaceAddRmsNorm`、`Swish`、`Gelu`、`SwiGlu`。
   - 原因：模型推理中稳定出现，涉及 reduction、多 pass、workspace 和小 shape latency，需要单独报告字段。
   - `RmsNorm/AddRmsNorm/SwiGlu` 优先对齐 `ops-nn`，普通 reduce 对齐 `ops-math`。
4. index/scatter/routing 类：`GatherV2/V3`、`ScatterUpdate`、`ScatterNdUpdate`、`TopKV2`、`MoeInitRouting*`、`MoeFinalizeRouting*`。
   - 原因：耗时较高但强依赖 indices、topK、routing 分布和 atomic/随机访存；没有运行时输入时只做 bounds 或 unresolved。
5. CV/常规大 kernel：`Conv3DV2`、`GroupNormSilu`、`Resize*`、`GridSample*`、ROI/NMS 类。
   - 原因：当前 profiling 中部分样本耗时很高，但算子族差异大；待前四类公共框架稳定后再从 `ops-cv` 逐类设计。

### profiling 解析规则

通用解析输入：

- `Type` / `Name`
- `Input Shapes`、`Input Data Types`、`Input Formats`
- `Output Shapes`、`Output Data Types`、`Output Formats`
- `Duration(us)`、`Block Dim`、`Mix Block Dim`
- AIC/AIV counter：`aic_*`、`aiv_*`、`cube_utilization(%)`

通用逻辑规格：

```text
op_type
op_family
input_elements[]
output_elements[]
input_bytes[]
output_bytes[]
logical_elements
broadcast_elements
reduction_elements
workspace_bytes
layout_pattern
```

无法从 profiling CSV 还原的运行时 attrs 必须显式记录，例如：

- `Transpose` permutation
- `Slice/StridedSlice` begin/end/stride
- `Concat/Split` axis
- `Gather/Scatter` indices 实际分布
- `TopK/MoE` topK、专家路由和 token 分布

### 源码对齐规则

每个算子 Type 先建立 source map：

```text
Type -> source_repo -> op_host -> tiling -> op_kernel -> template/strategy
```

首轮 source map：

- `ops-math/conversion`：`Cast`、`Transpose`、`TransData`、`AsStrided`、`Slice`、`Concat/Split` 类。
- `ops-math/math`：`Add`、`Mul`、`Sub`、`RealDiv`、`Pows`、比较和填充类。
- `ops-math/math` 或 `ops-nn`：`ReduceSum`、`ReduceMean`、`SoftmaxV2`。
- `ops-nn/norm`、`ops-nn/activation`：norm、activation 和 norm+activation fusion。
- `ops-nn/index`、`ops-transformer-master/moe`：gather/scatter、TopK、MoE routing。
- `ops-cv/image`、`ops-cv/objdetect`：resize、grid sample、ROI/NMS 和图像类 kernel。

若找不到源码或源码路径和 profiling Type 对不上，报告 `source_unavailable`，只输出 fallback 或 unresolved。

### 成本模型

Elementwise/vector：

```text
vector_compute_us = logical_elements * op_factor / vector_throughput
hbm_us = (unique_input_bytes + output_bytes) / hbm_bandwidth
estimated_us = max(vector_compute_us, hbm_us) + launch_overhead_us
```

Broadcast 输入按实际输入 storage 计读流量，按 output elements 计 vector 工作量。

Layout/memory：

```text
hbm_us = (input_bytes + output_bytes + workspace_bytes) / hbm_bandwidth
layout_overhead_us = source_visible_replay_or_stride_penalty
estimated_us = hbm_us + layout_overhead_us + launch_overhead_us
```

`layout_overhead_us` 只能来自源码可见的非连续搬运、tiling 多段循环、workspace merge 或 MTE 重排；缺 attrs 时不引入。

Reduction：

```text
read_us = input_bytes / hbm_bandwidth
write_us = output_bytes / hbm_bandwidth
partial_us = workspace_bytes / hbm_bandwidth
vector_compute_us = reduction_elements * reduce_op_factor / vector_throughput
estimated_us = max(read_us + write_us + partial_us, vector_compute_us) + sync_overhead_us + launch_overhead_us
```

Norm/activation fusion：

- RMS/LayerNorm 按 reduce pass + elementwise pass + optional residual/add/cast 组合。
- Swish/Gelu/SwiGlu 按 elementwise transcendental/activation factor + HBM 读写。
- fusion 算子优先按源码 kernel pass 数，而不是拆成多个独立 op 简单相加。

Index/scatter/routing：

- Gather 按 indices 读 + data 随机读 + output 写估计。
- Scatter/ScatterNd 按 data 读 + indices 读 + random write/atomic 风险估计。
- TopK/MoE routing 在缺少实际 indices/token 分布时只输出 bounds 或 unresolved，不加经验校准。

CV：

- resize/grid sample 按输出元素、插值 tap 数、坐标/weight 读写和边界处理分支估计。
- ROI/NMS 类依赖候选框数量、排序/抑制策略和 workspace，必须先读 `ops-cv` tiling 后单独设计。

### 报告字段

`other_ops` resolved 报告至少包含：

- 来源：`file`、`line`、`name`、`type`
- 分类：`op_family`、`source_repo`、`source_path`、`tiling_source`
- 规模：`logical_elements`、`input_bytes`、`output_bytes`、`workspace_bytes`
- 成本：`vector_compute_us`、`hbm_us`、`layout_overhead_us`、`sync_overhead_us`、`launch_overhead_us`
- 下界：`ideal_lower_bound_us`、`current_kernel_bound_us`
- 结果：`estimated_us`、`residual_us`、`duration_over_estimate`
- 诊断：`diagnosis`、`confidence`

unresolved 至少保留原始 shape/dtype/format、缺失 attrs 和 source lookup 失败原因。

### 实施任务拆分

1. 新增 `tools/other_ops_eval/`，实现通用 shape/dtype/format parser、Type 分类和 source map。
2. 注册 `op_kind=other_ops`，支持 CLI 输出 resolved/unresolved。
3. 先实现 layout/memory 类 analytic fallback 和部分 source strategy replay。
4. 实现 elementwise/vector 类成本模型，作为 AIV/HBM 基线。
5. 增加 reduction/norm/activation 的 pass-based 模型。
6. 对 index/scatter/routing 输出 bounds/unresolved，避免无 indices 拟合。
7. 按 `example_profilings/` 全量刷新 other_ops 基线，分析 top tail 后再决定下一类源码 replay。

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
