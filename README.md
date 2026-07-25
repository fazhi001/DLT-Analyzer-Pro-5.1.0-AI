# DLT Analyzer Pro 5.0 三种彩票可信分析版

基于原版 `DLT Analyzer Pro 4.2.3 AI` 源码直接升级，保留大乐透全部统计、AI可信评估、回测、导入导出和官方更新功能，并加入排列三、排列五中心。

## 三种彩票

- 超级大乐透：原有前区/后区模型与可信基线保护保持不变。
- 排列三：百位、十位、个位分别建模。
- 排列五：万位、千位、百位、十位、个位分别建模。

排列三和排列五共用同一套特征工程、训练、验证、预测和回测代码，但数据表、预测记录和各位置模型分别保存。

## 排列中心功能

在主界面左侧点击 **排列三 / 排列五中心**：

1. `同步最新`：从中国体彩网检查新增排列五开奖，并同步生成同期开奖结果的排列三数据。
2. `一键补齐排列数据`：分页补齐可取得的排列历史数据。
3. 支持 CSV、XLSX、XLSM 导入和 Excel 导出。
4. 查看每个位置 0—9 的历史频率、遗漏和综合概率。
5. 训练 3 个或 5 个独立位置模型。
6. 只有样本外 Log Loss 优于统计基线的位置模型才会启用；其余位置自动回退统计融合。
7. 支持 1—200 注号码生成、滚动验证和报告导出。

## 数据结构

原大乐透数据表不变。新增：

- `digit_draws`：按 `game + issue` 隔离排列三、排列五开奖。
- `digit_predictions`：分别保存两种彩票的预测记录。
- `%LOCALAPPDATA%/DLTAnalyzerPro/models/pl3/`：排列三位置模型。
- `%LOCALAPPDATA%/DLTAnalyzerPro/models/pl5/`：排列五位置模型。

## 本地运行

```powershell
python -m pip install -r requirements.txt
python main.py
```

## 运行测试

```powershell
python -m pytest
python main.py --self-test
```

## Windows 构建

双击：

```text
build_windows_5.0.bat
```

或者在 GitHub Actions 中运行：

```text
Build DLT Analyzer Pro 5.0 Three Games
```

安装包输出：

```text
release/DLT_Analyzer_Pro_5.0.0_3Games_Setup_x64.exe
```

## 风险提示

彩票开奖具有随机性。本软件用于历史数据统计、模型验证和娱乐研究，不保证预测中奖，也不应被用于超出个人承受能力的投注。
