# Retail Demand Forecasting

基于公开 M5 零售数据复现门店商品级需求预测流程，目标是使用历史日销量、价格和日历信息预测未来 30 天销量。本项目是公开数据上的学习与复现项目，不包含 Amazon 内部数据，也不代表公司业务结果。

## 当前进度

- 已准备 Python 3.11 Conda 环境和数据处理依赖。
- 已下载 M5 数据，并核对销量、日历与周价格文件的结构和对应关系。
- 数据处理、基线、LSTM、TFT 和实验追踪将在后续阶段逐步实现，目前没有模型结果。

## 数据来源

使用 [M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data) 的公开 Walmart 零售数据。下载和使用须遵循数据页面的条款，原始数据不提交至本仓库。

主要输入文件放在 `data/`：

```text
data/
  sales_train_evaluation.csv
  calendar.csv
  sell_prices.csv
```

`sales_train_validation.csv` 是较短的重叠历史，不能与 evaluation 文件上下拼接。`sample_submission.csv` 是竞赛提交格式，不是实际销量。M5 官方预测窗口为 28 天，本项目的 30 天实验将在历史数据上独立按时间评估，不直接宣称与竞赛排行榜可比。

## 实现路线

1. 使用 DuckDB 和 Polars 选取固定商品范围，将销量宽表转为日粒度长表，关联日历与价格并保存 Parquet。
2. 按时间划分数据，构造只依赖预测起点以前信息的 lag 与 rolling 特征。
3. 比较 Naive、Seasonal Naive、ARIMA、LSTM 和 TFT，使用 MLflow 追踪实验。
4. 报告 MAE、RMSE、WAPE，并分析预测步长、销量层级、价格变化和事件日期的误差。

价格下降不直接标记为真实促销。未来价格只有在明确假设预测时已知价格计划时，才能作为未来协变量使用。

## 开发环境

```text
conda activate amazon
python --version
```

PyCharm 项目解释器与终端应使用同一个 `amazon` Conda 环境。环境目录、IDE 配置、本地凭据、原始数据和大型实验产物由 `.gitignore` 排除。
