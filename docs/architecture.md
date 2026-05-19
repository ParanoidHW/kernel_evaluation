# Kernel 评估工具架构

本文是评估工具的架构入口。MatMul 详细建模说明见 `matmul_eval_design_zh.md`；Attention 当前迭代计划见 `attention_eval_iteration_plan.md`；硬件补充信息见 `info.md`。

## 目标

该工具从导出的 profiling CSV 中估计算子 kernel 耗时。设计目标不是对历史样本做黑盒曲线拟合，而是用能对应到 kernel 实现和硬件机制的模型解释耗时。

当前支持的算子族：

- `matmul`：包含 shape/layout 解析、量化路径、runtime knowledge-base 命中、advanced tiling 近似、fallback analytic search 和校准建议。
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

## 建模约定

每个算子族都应显式区分三类概念：

- `actual_tiling`：来自 runtime knowledge-base、host tiling replay 或可直接解析 op_tiling 元数据的真实 kernel tiling 信息。
- `fallback_tiling`：只有在真实 tiling 不可用时才使用的解析估计。
- `optimal_tiling`：用于对比的理想/物理下界，不代表当前 kernel。

MatMul 当前通过 `runtime_kb_exact`、`advanced_tiling_heuristic` 和 `analytic_search` 体现该拆分。Attention 在本地存在 `ops-transformer` 源码时输出 `actual_tiling_source=ops_transformer_source_strategy_replay`，否则退化为 `actual_tiling_source=unavailable_ops_transformer_replay` 和 `fallback_tiling_source=analytic_attention_bound`。

Attention 当前 kernel 估计使用 fused-infer split-fuse 路径中源码可见的常量，例如 `Q_TILE_CEIL=128` 和 `MAX_KV_STACK_LEN=512`，并结合 decode/prefill、mask/aux、MQA/GQA、FlashAttention/FusedInferAttention 等策略标签。在物理下界之上，模型额外加入 occupancy、traffic amplification、workspace score traffic、sync overhead、latency floor 和 template overhead factor。

## 配置

硬件和校准假设位于 `configs/`：

- `configs/ascend_910b4.json`：910B4 AI Core 数、HBM 带宽、cache 假设、峰值吞吐、MatMul runtime knowledge-base 路径和校准项。
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
