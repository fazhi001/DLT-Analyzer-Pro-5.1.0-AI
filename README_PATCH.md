# DLT Analyzer Pro 5.1.0 第一阶段补丁

本补丁替换 `src/dlt_analyzer_pro/digit_model.py`，并新增
`tests/test_digit_credible_engine.py`。

## 主要变化

- 排列三完整枚举 1,000 种结果。
- 排列五完整枚举 100,000 种结果。
- 删除随机候选抽样对最终排名的影响；保留 `candidate_count` 参数仅用于兼容旧调用。
- 位置模型从单次留出验证升级为 2～5 个连续时间折的滚动样本外验证。
- 统计与机器学习融合权重从固定 65%/35% 改为 0%～70% 网格优化。
- 只有达到最低 Log Loss 改善且至少 60% 时间折获胜时才启用机器学习。
- 模型包保存动态权重、验证期数和时间折胜率；特征版本升级为 `digit-position-v2-walkforward`。
- 增加全空间枚举、确定性输出、融合归一化和动态权重边界测试。

## 应用位置

```text
src/dlt_analyzer_pro/digit_model.py
tests/test_digit_credible_engine.py
```
