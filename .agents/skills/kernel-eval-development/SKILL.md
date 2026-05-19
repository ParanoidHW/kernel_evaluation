---
name: kernel-eval-development
description: 用于按既定方案开发或修改本仓库内 Ascend/CANN kernel 评估器，实现 parser、tiling/source replay、成本模型、报告字段、CLI 注册、配置和文档；适用于任意算子族的评估模型落地。
---

# Kernel 评估开发

在已有方案或明确任务后，开发某一类 kernel 评估器时使用本 skill。若还没有方案，先使用 `kernel-eval-design`；开发后需要检视时使用 `kernel-eval-review`；误差收敛时使用 `kernel-eval-iteration`。

## 开发目标

把方案落地为可运行、可解释、可验证的评估实现：

- 能从 profiling CSV 中解析目标算子行。
- 能推断逻辑规格、物理存储、dtype、format、辅助输入和 unresolved 原因。
- 能结合 kernel 实现、tiling 实现和硬件配置估算当前 kernel 耗时。
- 能输出足够诊断字段，支持最大相对误差 tail 分析。
- 能保留当前 kernel 估计、fallback 估计和物理下界的语义分离。

## 开发流程

1. 阅读方案文档和 `docs/architecture.md`，确认算子族、源码依据、报告字段和验证命令。
2. 检查现有公共接口：`tools/op_eval/api.py`、`tools/op_eval/cli.py`、`tools/op_eval/profiling.py`、`tools/op_eval/types.py`。
3. 新建或修改 `tools/<family>_eval/`，按模块拆分 spec/parser/cost/tiling/report/calibration。
4. 实现 profiling 行过滤，必须基于 `Type`，不要只依赖 scope 风格 `Name`。
5. 实现 parser，输出 resolved spec；失败时输出 unresolved 行，并保留 input/output shape、dtype、format、name/type 和来源行号。
6. 实现 source/tiling replay。能从 CANN 源码或知识库得到真实 tiling 时标为 `actual_tiling`；不能得到时明确降级为 `fallback_tiling`。
7. 实现成本模型，拆分 compute、vector、GM/HBM、cache/tiling 重复搬运、workspace、format、sync、launch、occupancy 和 template/floor。
8. 实现报告字段，至少包含估计分量、tiling 来源、瓶颈、confidence、diagnosis、`duration_over_estimate` 和相对误差分析所需字段。
9. 在 `tools/op_eval` 注册新的 `op_kind` 或扩展现有算子族。
10. 更新 `configs/` 和 `docs/`，记录硬件假设、校准项、已知限制和验证结果。

## 代码结构建议

新算子族优先采用以下结构：

```text
tools/<family>_eval/
  __init__.py
  api.py
  common.py
  parser.py
  tiling_replay.py
  reporting.py
```

已有算子族可沿用当前结构，但应保持边界清晰：

- `common.py`：spec、shape/dtype/format 解析和共享常量。
- `parser.py`：profiling 行解析和 unresolved 诊断。
- `tiling_replay.py`：源码/知识库/host tiling replay。
- `api.py`：估计 API、成本模型和 profiling evaluator。
- `reporting.py`：CSV 行和 summary。

## 实现约束

- 不把 per-shape 拟合曲线写进模型。
- 不把 `ideal_lower_bound_us` 当作当前 kernel 的 `estimated_us`。
- 不把 fallback 估计伪装成 actual tiling。
- 不静默吞掉 unresolved；unresolved 报告必须能指导下一轮 parser 修复。
- 不把硬件未知项硬编码到代码中；优先放入 `configs/`，并在文档中说明来源或假设。
- 不为单个 tail 加不可解释常量；新增参数必须对应 kernel 或硬件机制。

## 报告字段检查

开发完成前确认报告至少能回答：

- 这行来自哪个 profiling 文件和行号？
- 解析出的逻辑规格是什么？
- 物理存储和辅助输入如何计入？
- 使用的是 actual tiling、fallback tiling 还是 optimal/lower-bound？
- 主导瓶颈是 compute、vector、HBM、launch、sync 还是模板/floor？
- 当前估计与物理下界差距多大？
- 如果误差很大，diagnosis 能否指出可能路径？

## 基础验证

每次开发结束至少运行：

```bash
python3 -m compileall tools
python3 tools/eval_ops.py --op-kind <family> \
  --profiling <profiling_dir> \
  --config <config.json> \
  --output <family>_eval_report_<soc>.csv \
  --unresolved-output <family>_eval_unresolved_<soc>.csv
python3 .agents/skills/kernel-eval-iteration/scripts/analyze_report_errors.py \
  <family>_eval_report_<soc>.csv
```

如果当前没有 profiling 样例，也要至少提供导入/编译检查和一个最小手工 spec 调用路径。

## 交付内容

开发完成时说明：

- 修改了哪些模块和配置。
- 新增或变更了哪些报告字段。
- resolved/unresolved 数量。
- 最大相对误差、p95、中位数和 top tail。
- 哪些问题需要进入 review 或 iteration。
