# Kernel 评估工具架构

本文是评估工具的架构入口。MatMul 详细建模说明见 `matmul_eval_design_zh.md`；Attention 设计说明见 `attention_kernel_eval_design.md`；当前 profiling 评估差距见 `current_eval_gap_zh.md`；硬件补充信息见 `info.md`。

## 目标

该工具从导出的 profiling CSV 中估计算子 kernel 耗时。设计目标不是对历史样本做黑盒曲线拟合，而是用能对应到 kernel 实现和硬件机制的模型解释耗时。

当前支持的算子族：

- `matmul`：包含 shape/layout 解析、量化路径、runtime knowledge-base 命中、advanced tiling 近似、fallback analytic search 和校准建议。
- `grouped_matmul`：独立过滤 `GroupedMatmul` 行，并报告专家 routing 的均衡与极端不均衡两种成本边界。
- `attention`：包含 Q/K/V shape 解析、QK/PV FLOPs、softmax/vector 工作量、最小 HBM 流量，以及本地 `ops-transformer` 源码存在时的 source strategy replay。`estimated_us` 表示当前 kernel 估计，`ideal_lower_bound_us` 保留为物理下界参考。

## 分层

实现分为共享算子层和算子族专用包：

- `tools/eval_ops.py`：命令行入口。
- `tools/op_eval/cli.py`：CLI 解析、配置加载、summary 分发和 CSV 输出。
- `tools/op_eval/api.py`：库入口，包括 `estimate_op(...)` 和 `evaluate_profiling(...)`。
- `tools/op_eval/common.py`：共享数值、dtype、shape、format 和配置辅助函数。
- `tools/op_eval/profiling.py`：profiling CSV 发现和报告写入。
- `tools/op_eval/types.py`：共享报告容器。
- `tools/matmul_eval/`：MatMul 专用解析、成本 API、kernel/tiling 模型、量化模型、runtime knowledge-base 加载、报告和校准。
- `tools/attention_eval/`：Attention 专用解析、成本 API、tiling/source replay、fallback analytic 模型和报告。

## 数据流

profiling 评估流程：

1. `tools/eval_ops.py` 调用 `op_eval.cli.run_cli()`。
2. `op_eval.cli` 加载硬件配置，并调用 `op_eval.api.evaluate_profiling(...)`。
3. `op_eval.profiling.iter_input_files(...)` 将输入文件或目录展开为 CSV 文件。
4. 选定算子族根据 profiling `Type` 过滤行。
5. 算子族解析 shape、format、dtype、计数器和输出元数据。
6. 算子族估算 compute、memory、launch、format 和 actual/fallback/optimal tiling 相关分量。
7. 返回 resolved 与 unresolved 行，封装为 `ProfilingEvaluation`。
8. CLI 打印算子族 summary，并按需写出 resolved/unresolved CSV 报告。

整体数据流可概括为：

```text
profiling CSV
  -> op_eval 发现文件和分发行
  -> <family>_eval 解析 Type/shape/format/dtype/counter
  -> 逻辑 kernel spec
  -> tiling/source replay 或 fallback 模型
  -> compute/vector/GM/launch/template 成本分量
  -> resolved report + unresolved report
  -> tail/error analyzer
```

其中 `kernel_details.csv` 是输入事实来源，但不是完整 kernel 上下文。它通常缺少 host tiling 中间结果、运行时 group_list 值、cache 命中状态和真实模板选择细节。因此评估器必须在报告中标明估计来源和置信度，不能把 fallback 推断伪装成真实 tiling。

## 建模约定

每个算子族都应显式区分三类概念：

- `actual_tiling`：来自 runtime knowledge-base、host tiling replay 或可直接解析 op_tiling 元数据的真实 kernel tiling 信息。
- `fallback_tiling`：只有在真实 tiling 不可用时才使用的解析估计。
- `optimal_tiling`：用于对比的理想/物理下界，不代表当前 kernel。

MatMul 当前通过 `runtime_kb_exact`、`advanced_tiling_heuristic` 和 `analytic_search` 体现该拆分。Attention 在本地存在 `ops-transformer` 源码时输出 `actual_tiling_source=ops_transformer_source_strategy_replay`，否则退化为 `actual_tiling_source=unavailable_ops_transformer_replay` 和 `fallback_tiling_source=analytic_attention_bound`。

Attention 当前 kernel 估计使用 fused-infer split-fuse 路径中源码可见的常量，例如 `Q_TILE_CEIL=128` 和 `MAX_KV_STACK_LEN=512`，并结合 decode/prefill、mask/aux、MQA/GQA、FlashAttention/FusedInferAttention 等策略标签。在物理下界之上，模型额外加入 occupancy、traffic amplification、workspace score traffic、sync overhead、latency floor 和 template overhead factor。

## 平台识别

对昇腾 profiling，平台识别需要区分 Cube 和 Vector 核数：

- Cube 类算子主要对齐 `aic_num`，例如 `MatMul`、`BatchMatMul`、`GroupedMatmul`、`QuantBatchMatmul` 和 Cube-heavy attention/FA 路径。
- Vector 类算子主要对齐 `aiv_num`，例如 `Cast`、`Transpose`、`RotaryMul`、`Gather/Scatter`、activation、normalization、routing 和部分 fusion。
- 910B4 的典型证据是 Cube 最大 block dim 约 `20`、Vector 最大 block dim 约 `40`。
- 910C/A3 的典型证据是 Cube 最大 block dim 约 `24`、Vector 最大 block dim 约 `48`。

因此不要用全文件最大 `Block Dim` 直接推断 matmul/FA 的 Cube 核数；全局最大值经常来自 Vector 算子。

## Kernel 评估原理

评估器不是“按历史样本回归 duration”，而是把耗时拆成可解释的 kernel 机制：

- `compute_us`：逻辑或对齐 FLOPs 除以目标 SoC 的 dtype 峰值，并考虑 pipeline/core efficiency。
- `vector_us`：attention softmax、elementwise、reduction 等 AIV 工作量。
- `hbm_us`：最小 GM/HBM 流量和 tiling 重读流量除以 HBM 带宽；MatMul 会考虑 L2 对重复流量的折减。
- `launch_overhead_us`：kernel 启动和小 kernel 固定成本。
- `format_overhead_us`：ND/NZ 等运行时格式转换成本。
- `sync/template/latency`：attention split-fuse、decode/prefill、小序列和模板路径带来的额外成本。

`estimated_us` 表示当前 kernel 预测，包含当前模型认为 kernel 实际会承担的成本。`ideal_lower_bound_us` 只表示物理下界，通常不含真实模板开销、同步开销和小 kernel 固定延迟。若 `ideal_lower_bound_us > duration_us`，优先怀疑解析、流量过计或缺失运行时上下文，而不是继续增加经验系数。

## 算子族内部结构

MatMul 评估链路：

```text
matmul_eval.common
  -> 从 shape/format 推断 MatmulSpec
matmul_eval.runtime_kb
  -> 加载 ops-nn runtime knowledge-base
matmul_eval.kernel_model
  -> runtime_kb_exact / advanced_tiling_heuristic / analytic_search
matmul_eval.quant_model
  -> 量化 matmul 的 compute、dequant、aux 和 GM 流量修正
matmul_eval.gmm_model
  -> GroupedMatmul 的专家均衡/极端不均衡 routing 边界
matmul_eval.api
  -> 合并 tiling、compute、memory、launch、format 成 MatmulCostEstimate
matmul_eval.evaluator
  -> 逐行评估 profiling CSV 并生成报告
```

Attention 评估链路：

```text
attention_eval.common
  -> 从 Q/K/V shape 推断 AttentionSpec
attention_eval.tiling_replay
  -> 基于 ops-transformer 源码存在性和 shape/type 输出策略标签
attention_eval.api
  -> 计算 QK/PV、softmax/vector、GM、occupancy、sync、latency/template 成本
attention_eval.evaluator
  -> 逐行评估 profiling CSV 并生成报告
```

当前 attention replay 是 `source_strategy_replay`，不是二进制 tiling replay。它说明命中了哪类源码路径和策略标签，但不声称已拿到 CANN host tiling 的完整输出。

## 配置

硬件和校准假设位于 `configs/`：

- `configs/ascend_910b4.json`：910B4 AI Core 数、HBM 带宽、cache 假设、峰值吞吐、MatMul runtime knowledge-base 路径和校准项。
- `configs/ascend_910b4_1.json`：qwen3-7b/qwen7b 专用 910B4-1 配置，保留 20 AIC/40 AIV 的 910B4 BlockNum 证据，但使用用户确认的 1.6 TB/s HBM。
- `configs/ascend_910c.json`：910C 可见设备假设、峰值吞吐、advanced MatMul tiling 设置和校准项。

拿到更可靠的硬件或 CANN platform 数据后，应优先更新配置值，避免把 per-shape 拟合常量写入模型代码。

## 报告

resolved CSV 行包含来源文件和行号、推断后的逻辑算子规格、profiling counter、估算分量、瓶颈分类、confidence、diagnosis 标签和实测/估计残差。

unresolved CSV 行保留诊断 parser 缺口所需的元数据：文件、行号、算子 type/name、输入 shape、输出 shape，以及可用的 layout 字段。

## 扩展方式

新增算子族时：

1. 在 `tools/<family>_eval/` 下新增包。
2. 根据 profiling `Type` 做行检测，不依赖 scope 风格的 `Name`。
3. 实现公开 estimate API 和 profiling-file evaluator。
4. 报告行需要包含成本分量、来源标签、diagnosis 和 confidence。
5. 在 `tools/op_eval/api.py` 和 `tools/op_eval/cli.py` 注册该算子族。
6. 保持 actual tiling replay、fallback 估算和 optimal bound 语义分离。
