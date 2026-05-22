# Kernel 评估工具

本仓库用于根据导出的 profiling CSV 估算昇腾算子 kernel 耗时。当前重点覆盖 `MatMul`、`GroupedMatmul` 和 Attention 家族，设计原则是从 kernel 实现、tiling 逻辑和硬件约束出发建立可解释模型，而不是对历史样本做黑盒拟合。

## 仓库范围

- 硬件平台：`Ascend 910B4`、`Ascend 910B4-1`、`Ascend 910C`
- profiling 样本：`example_profilings/`
- 评估入口：`tools/eval_ops.py`
- 架构说明：`docs/architecture.md`
- MatMul 设计文档：`docs/matmul_eval_design_zh.md`
- Attention 设计文档：`docs/attention_kernel_eval_design.md`

## 目录说明

- `tools/eval_ops.py`：统一 CLI 入口
- `tools/op_eval/`：共享配置、profiling 解析、API 和报告逻辑
- `tools/matmul_eval/`：MatMul / GroupedMatmul 解析、tiling、成本模型和报告
- `tools/attention_eval/`：Attention 解析、source strategy replay、成本模型和报告
- `configs/`：平台配置和模型相关参数
- `docs/`：架构、设计、误差分析和硬件补充文档
- `example_profilings/`：已有 profiling 样本和说明
- `eval_results/`：各轮基线刷新后的汇总结果

## 建模约束

- `estimated_us` 表示当前 kernel 路径的运行时间估计。
- `ideal_lower_bound_us` 只表示物理下界，不代表当前 kernel 一定能达到。
- 新增或修改模型项时，必须能对应到源码实现、tiling 分支或硬件执行机制。
- 不允许为了拟合个别样本引入无关超参、校准项或经验补丁。

## 当前重点能力

### MatMul / GroupedMatmul

- shape / dtype / format 解析
- runtime knowledge-base 命中
- advanced tiling 近似
- fallback analytic search
- GroupedMatmul 路由上下界建模
- 量化 MatMul 路径支持

### Attention

- Q/K/V shape 解析
- QK/PV FLOPs 与 vector 工作量估算
- HBM 最小流量与当前 kernel 流量放大
- `ops-transformer` source strategy replay
- FIA / FA / PFA / IFA / QSFA 路径支持

## 常用命令

MatMul：

```bash
python3 tools/eval_ops.py \
  --op-kind matmul \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output matmul_eval_report_910b4.csv \
  --unresolved-output matmul_eval_unresolved_910b4.csv
```

Attention：

```bash
python3 tools/eval_ops.py \
  --op-kind attention \
  --profiling example_profilings/910C \
  --config configs/ascend_910c.json \
  --output attention_eval_report_910c.csv \
  --unresolved-output attention_eval_unresolved_910c.csv
```

GroupedMatmul：

```bash
python3 tools/eval_ops.py \
  --op-kind grouped_matmul \
  --profiling example_profilings/910B4 \
  --config configs/ascend_910b4.json \
  --output grouped_matmul_eval_report_910b4.csv \
  --unresolved-output grouped_matmul_eval_unresolved_910b4.csv
```

## 结果与基线

- 最新汇总结果见 `eval_results/LATEST`
- 历史刷新结果按时间戳和 commit 存在 `eval_results/<timestamp>_<commit>/`
- 当前工作记录和关键结论见 `session.md`
