---
name: kernel-eval-design
description: 用于为某一类 Ascend/CANN kernel 评估器做方案设计；必须从 cann/ops-nn、ops-transformer、ops-math、ops-cv 等源码中的 kernel 实现和 tiling 实现出发，并结合目标硬件架构设计可解释的评估模型。
---

# Kernel 评估方案设计

在为某一类 kernel 启动评估器设计时使用本 skill。适用对象包括但不限于 MatMul、Attention、Elementwise、Reduction、Normalization、CV、Quant、通信融合算子，以及后续从 CANN 开源仓中引入的其他算子族。

本 skill 只负责“方案设计”。进入实现阶段使用 `kernel-eval-development`，进入验证和误差收敛阶段使用 `kernel-eval-iteration`。

## 设计目标

输出一份可落地的评估方案，说明：

- 要评估的 kernel 类型、profiling 行过滤规则和 shape/dtype/format 解析规则。
- 源码中的 kernel 实现路径、host tiling 路径、模板分支和关键常量。
- 目标硬件架构如何影响 compute、vector、GM/HBM、L2/L1/UB/L0、fixpipe、sync、launch 和 occupancy。
- 当前 kernel 估计、fallback 估计和物理下界如何分离。
- 需要生成哪些报告字段、诊断标签、残留分类和验证用例。

不要在方案阶段直接写 per-shape 拟合曲线。经验项必须能对应到 kernel 或硬件机制。

## 源码来源

优先使用本地已下载源码；没有本地源码时再考虑从对应仓库获取。

- `https://gitcode.com/cann/ops-nn`：NN/MatMul/量化/基础深度学习算子。
- `https://gitcode.com/cann/ops-transformer`：Attention、Transformer、paged/decode/prompt 等相关算子。
- `https://gitcode.com/cann/ops-math`：数学、elementwise、reduction 等算子。
- `https://gitcode.com/cann/ops-cv`：CV 图像处理相关算子。

常见本地目录名包括：

- `ops-nn`、`ops-nn-master`
- `ops-transformer`、`ops-transformer-master`
- `ops-math`、`ops-math-master`
- `ops-cv`、`ops-cv-master`

## 方案设计流程

1. 明确算子族和 profiling 目标：算子 `Type`、典型 `Name`、输入/输出数量、dtype、format、shape 语义和目标 SoC。
2. 在对应 CANN 源码仓中定位 op_host、op_kernel、tiling、模板实例化和平台分支。不要只看 kernel 名称，要找到 shape 到 tiling 到执行模板的完整路径。
3. 抽取源码事实：tiling key、block/tile 常量、format 约束、dtype 分支、特殊模板阈值、workspace 规则、atomic/reduction/fixpipe 路径和 fallback 路径。
4. 梳理硬件架构假设：AI Core/AIV 数量、Cube/Vector 能力、L0A/L0B/L0C、L1、L2、UB、GM/HBM 带宽、DMA/MTE、fixpipe、同步和 launch 开销。
5. 定义解析模型：如何从 profiling CSV 推断逻辑规格、物理存储元素、辅助 tensor、mask/scale/bias/workspace，以及无法解析时的 unresolved 输出。
6. 定义成本模型：至少拆分 compute、vector、GM/HBM、cache/tiling 重复搬运、format 转换、workspace、sync、launch、occupancy 和模板开销。
7. 明确三类估计语义：`actual_tiling` 来自源码/tiling replay，`fallback_tiling` 用于真实 tiling 不可得时，`optimal_tiling` 或 `ideal_lower_bound_us` 只作为物理下界。
8. 设计报告字段：必须能解释估计来源、瓶颈、confidence、diagnosis、duration/estimate 残差和最大相对误差 tail。
9. 设计验证计划：按 SoC、dtype、shape regime、kernel 模板和 tail 样本分组，使用最大相对误差作为主要验收指标。
10. 输出方案文档并记录待实现任务；再进入 `kernel-eval-iteration` 流程。

## 源码阅读检查点

定位源码时至少回答以下问题：

- profiling `Type` 对应哪些 op_host 注册、tiling 函数和 device kernel 文件？
- host tiling 是否有 runtime knowledge-base、platform 分支、compile-time 宏或 shape 特化？
- tile/block 参数是固定常量、启发式搜索、知识库命中，还是由硬件平台查询得到？
- 哪些输入是主数据流量，哪些是 scalar、metadata、mask、scale、bias、workspace 或可选输入？
- 输出是否经过 fixpipe、cast、dequant、atomic add、reduce、transpose、layout conversion 或 GM workspace merge？
- 小 shape、低 tile 数、长 K、长序列、非对齐、稀疏/量化、GQA/MQA、batch/varlen 等路径是否触发不同模板？

## 硬件建模检查点

设计方案必须说明目标 SoC 的硬件假设来源，优先读取 `configs/ascend_*.json` 和 `docs/info.md`。

需要显式考虑：

- AI Core 并行度与 occupancy：tile 数是否足以填满 core，是否存在 tail imbalance。
- Cube 计算峰值：dtype、HF32/非 HF32、量化 TOPS、pipeline efficiency。
- Vector/AIV 开销：softmax、activation、elementwise、reduction、dequant、layout rearrange。
- 存储层级：GM/HBM、L2、L1、UB、L0A/L0B/L0C 容量和重复搬运。
- 数据搬运：MTE/DMA、GM-to-L0 on-the-way、ND2NZ/NZ、workspace spill。
- 输出路径：fixpipe、cast、atomic、reduce-scatter/merge、format conversion。
- 固定开销：launch、同步、模板调度、小 shape latency floor。

## 方案输出模板

设计文档建议按以下结构写入 `docs/`：

```text
# <Kernel Family> 评估方案

## 范围
## profiling 解析规则
## 源码实现路径
## tiling 与模板分支
## 硬件架构假设
## 成本模型
## 报告字段和诊断标签
## 验证计划
## 已知限制和残留
## 实施任务拆分
```

## 设计约束

- 不允许只根据实测样本做 per-shape 拟合。
- 不允许把物理下界当作当前 kernel 耗时。
- 不允许在未读 tiling 或 kernel 源码的情况下声称已建模具体实现。
- 如果源码缺失，必须在方案中标注 `source_unavailable`，并把相关部分降级为 fallback。
- 如果硬件参数不确定，必须标注配置假设和对误差的影响。
- 方案完成后，应能直接指导代码实现和报告字段设计。
