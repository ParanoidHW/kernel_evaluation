# 昇腾 910B/910C 硬件信息补充

## 说明

本文件用于记录仓库中建模时引用的硬件背景信息和额外备注。这里的信息只作为本地分析辅助，不替代正式平台规格说明。

## 当前关注点

- 910B4 与 910C 的 `aic_num` / `aiv_num` 不同，评估时必须区分 Cube 与 Vector 路径。
- qwen3-7b/qwen7b 当前使用单独的 `910B4-1` 配置，HBM 带宽按 1.6 TB/s 处理。
- profiling 中的 `Block Dim` 不能跨算子族直接比较；MatMul / Attention 主要看 Cube 路径，很多其他算子会反映 Vector block dim。

## 参考链接

- 昇腾 910B/C 公开信息整理：https://zhuanlan.zhihu.com/p/2004196636507789012
