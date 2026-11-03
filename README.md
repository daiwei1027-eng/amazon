# Retail Demand Forecasting

基于公开 M5 零售数据复现门店商品级未来 30 天销量预测，重点练习数据清洗、可复现抽样、时间窗口、无未来信息泄漏的评估和模型比较。

截至 2026-11-03，已跑通全量数据预处理、固定实验子集、MinIO 上传、PyTorch Dataset/DataLoader、Naive 基线评估与 TensorBoard 记录。**Naive 验证 MAE 为 0.954333 件/门店商品/天**。LSTM、TFT 和 MLflow 实验追踪尚未实现。

本项目是公开数据上的学习与复现项目，不包含 Amazon 内部数据，也不代表公司业务结果。

## 当前进度

| 环节 | 状态 |
| --- | --- |
| 销量宽转长、日历与周价格关联、日期类型及排序 | 已完成，生成全量 12 列 Parquet |
| 固定门店与商品抽样、产物读回检查 | 已完成 |
| MinIO Compose、上传脚本、真实往返集成测试 | 已跑通 |
| 日历、lag、滚动均值与标准差 | 已生成完整实验子集的 388,200 行、24 列特征文件，相关小表测试通过 |
| 56 天历史、30 天标签的窗口 Dataset | 已完成 |
| 按标签日期划分 train/valid/test Subset | 已完成，排除跨边界窗口 |
| 通用训练单轮、评估、fit 与 TensorBoard 接口 | 已实现；当前实测运行的是 Naive 评估分支 |
| Seasonal Naive、ARIMA、LSTM、TFT | 待实现 |
| MLflow、Airflow、TorchServe、FastAPI | 已规划端口，尚未在本项目部署 |

目前模型输入只有历史销量。工程特征尚未接入 Dataset；带优化器的训练分支也尚未通过实际可训练模型验证。

## 数据与实验范围

数据来源为 [M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data) 的公开 Walmart 数据。原始文件与生成产物均不提交 Git，使用须遵循数据页面条款。

主要原始输入：

```text
data/
  sales_train_evaluation.csv
  calendar.csv
  sell_prices.csv
```

`sales_train_validation.csv` 是同一批序列的较短重叠历史，不能与 evaluation 文件上下拼接；`sample_submission.csv` 是提交模板，没有真实销量标签。M5 原竞赛预测窗口为 28 天，本项目使用历史日期独立评估 30 天预测，不直接对标竞赛排行榜。

| 范围 | 全量处理产物 | 当前实验子集 |
| --- | ---: | ---: |
| 门店 | 10 | CA_1、CA_2 |
| 不同商品 | 3,049 | 100 个共同商品 |
| 门店商品序列 | 30,490 | 200 |
| 每条序列历史 | 1,941 天 | 1,941 天 |
| 日记录 | 59,181,090 | 388,200 |
| 当前基础字段 | 12 列 | 12 列 |

日期覆盖 2011-01-29 至 2016-05-22。商品清单先排序，再使用随机种子 42 抽取；当前包含 FOODS 42 个、HOBBIES 21 个、HOUSEHOLD 37 个。此为简单随机抽样的实际组成，尚未采用分层抽样。

开发设备为 i7-12700KF、约 32 GB 内存和 RTX 5060（8 GB 显存）。考虑本地硬件资源及反复开发、验证、调参的成本，正式实验采用固定子集，同时保留可复用的全量处理结果。评价重点是处理规则、实验边界、可复现性与结果解释；当前没有全量训练耗时或全量模型效果的实测结论。

## 时间窗口与防泄漏规则

完整实验子集与其特征文件均保留全部历史。按门店商品分组，构造 56 天历史输入和随后 30 天销量标签，再按**完整标签窗口的日期**划分样本。

| 集合 | 历史与标签规则 | 窗口数 |
| --- | --- | ---: |
| train | 标签结束日期不晚于 2016-03-23 | 359,200 |
| valid | 历史为 2016-01-28 至 2016-03-23；标签为 2016-03-24 至 2016-04-22 | 200 |
| test | 历史为 2016-02-27 至 2016-04-22；标签为 2016-04-23 至 2016-05-22 | 200 |

完整 Dataset 有 371,200 个候选窗口，其中 11,600 个标签跨集合边界的窗口被排除。每个验证、测试样本各预测 30 天，因此各有 6,000 个目标值。

- 验证与测试使用预测起点前已发生的历史；历史与前一阶段数据重叠是允许的。
- 不使用预测起点后的真实销量生成该次预测的输入，不能将验证期每天实际销量依次作为一次 30 天预测的 lag 输入。
- 训练 DataLoader 打乱的是合法训练窗口顺序，窗口内部时间顺序保持不变；验证不打乱。
- 目前没有标准化步骤。后续需要拟合的变换只能在实际训练部分拟合。
- 测试集已划定，尚未用于模型效果评估。

早期生成的 train/valid/test 日记录 Parquet 可作为阶段产物保留；当前 `scripts/split_m5.py` 已改为划分 Dataset 索引，不再负责生成这三个文件。

## 当前基线结果

Naive 对每条序列取预测起点最后一天的销量，重复为未来 30 天预测。使用无可学习参数的 `nn.Module` 接入统一评估接口，不运行优化器或反向传播。

| 模型 | 预测起点 | 验证区间 | 预测值数 | 验证 MAE |
| --- | --- | --- | ---: | ---: |
| Naive | 2016-03-23 | 2016-03-24 至 2016-04-22 | 6,000 | 0.954333 |

直接按原始销量核算：绝对误差总和为 5,726，MAE = 5,726 / 6,000。TensorBoard 中的 float32 标量约为 0.954333365，与该结果一致。该值是每个门店商品每日的平均绝对销量误差，不是准确率，也不是最终测试集成绩。

同日再次运行，使用者报告输入切换为完整特征文件，控制台仍输出 `Naive validation MAE: 0.954333`，完成 25 个验证批次。当前 Dataset 仍只提取历史销量，读取特征文件不代表 Naive 使用了全部特征，因此相同结果符合预期。

目前尚未实现 WAPE、RMSE 和其他模型的比较，不能仅据这一个数字判断模型优劣。

## 环境与安装

当前已核实环境为 Windows x86_64、Python 3.11.18、PyTorch 2.10.0+cu128、NVIDIA 驱动 577.00。CUDA 12.8 GPU 张量运算已通过小型检查。`Dataset` 和 `DataLoader` 包含在 torch 中，不需要额外安装名为 dataset 的包。

新建环境时：

```powershell
conda create -n amazon python=3.11
conda activate amazon
python -m pip install -r requirements.txt
```

`requirements.txt` 固定当前实际使用的 Python 直接依赖，包含测试工具；通过额外的 PyTorch 官方 CUDA 12.8 索引安装指定的 `torch==2.10.0+cu128`。Docker Desktop 和容器镜像单独管理。DuckDB、MLflow、TFT 等后续组件接入时再更新依赖。

## 当前可运行的本地流程

从项目根目录运行。先自行下载并放好上述三个 CSV：

```powershell
python -m scripts.inspect_m5
python -m scripts.inspect_m5_scope
python -m etc.features
python -m scripts.train
```

前三步依次生成全量基础表、固定实验子集及 `data/feature/sales_feature.parquet`。特征文件为 388,200 行、24 列，覆盖全部实验日期。训练入口构造窗口、按配置划分，再运行 Naive 验证分支并打印 MAE；本地评估不依赖 MinIO 服务正在运行。

关键配置在 `configs/default.yaml`，包括输入输出文件、门店与抽样种子、标签日期边界、历史长度、预测长度、batch size 和 epoch 数。

训练入口已核对并保存：`scripts/train.py` 按配置中的 `data.directory / features.directory / features.filename` 读取完整的 `data/feature/sales_feature.parquet`。Dataset 当前仍只提取历史销量，因此读取特征文件不会自动启用其中的其他特征。

工程特征当前尚未作为模型输入。接入多变量模型前，需明确数值列、缺失值与缩放策略；未来标签日期上的 lag/rolling 值不能直接作为一次 30 天预测的未来已知输入。

## MinIO 与 TensorBoard

当前 Compose 只部署 MinIO，使用固定的第三方源码构建镜像 `ghcr.io/coollabsio/minio:RELEASE.2025-10-15T17-29-55Z`，发布来源见 [Coollabs MinIO](https://github.com/coollabsio/minio)。数据通过具名卷 `minio_data` 持久化。

在项目根目录自行创建 `.env`，填写实际值：

```dotenv
MINIO_ROOT_USER=amazon_admin
MINIO_ROOT_PASSWORD=replace_with_your_password
```

启动：

```powershell
docker compose up -d minio
```

API 为 `localhost:9000`，控制台为 `http://localhost:9001`。在控制台创建 raw、silver 桶。当前上传入口选择配置中的特征文件；生成 `data/feature/sales_feature.parquet` 后可执行：

```powershell
python -m scripts.ingest_minio_bucket
```

当前配置的上传目标为 `silver/feature/sales_feature.parquet`，本次尚未核实该新对象的上传结果；较早的 `silver/feature/train_feature.parquet` 上传已确认。原始 CSV 自动入库和训练入口直接读取 MinIO/DuckDB 尚待实现。真实 `.env` 已被 Git 忽略，当前仓库尚未提供 `.env.example`。

Naive 每次运行会在 `logs/tensorboard/naive_<时间戳>/` 创建独立日志目录，只记录一个 `MAE/valid` 点。启动查看器：

```powershell
python -m tensorboard.main --logdir logs/tensorboard --port 6006 --host 127.0.0.1
```

打开 `http://localhost:6006`。后续可训练模型会逐 epoch 记录 `MAE/train` 与 `MAE/valid`。TensorBoard 曲线记录已接入，MLflow 尚未接入；两者职责分别记录，不混称为已完成的实验平台。

## 测试

```powershell
python -m pytest tests -q
```

2026-11-03 实测：**13 passed、1 skipped**。包括宽长表、日期转换、固定抽样、日历/lag/rolling 特征和窗口标签边界测试。MinIO 集成测试默认跳过。

MinIO 已启动时可显式运行真实文件往返测试：

```powershell
$env:RUN_MINIO_TESTS = "1"
python -m pytest tests/test_ingest.py -v
$env:RUN_MINIO_TESTS = $null
```

该测试使用独立 tests 桶和唯一对象名，完成小型 Parquet 上传、下载、字节比较与对象清理。最近一次实际通过记录见 2026-11-01 日志，本次没有重新执行。

## 代码入口与后续工作

| 文件 | 职责 |
| --- | --- |
| `scripts/inspect_m5.py` | 全量原始数据整理与基础 Parquet |
| `scripts/inspect_m5_scope.py` | 固定范围的共同商品抽样 |
| `etc/features.py` | 从完整实验子集生成日历、lag、rolling 特征文件 |
| `etc/dataset.py` | 按序列构造窗口，返回历史与标签张量 |
| `scripts/split_m5.py` | 按标签日期筛选 Subset 索引 |
| `etc/losses.py` | MAE 计算及形状检查 |
| `models/baseline.py` | Naive 前向预测 |
| `scripts/train.py` | Dataset/DataLoader 组装、训练/验证、fit、TensorBoard |
| `docs/` | 按日期记录完成证据、设计变化和未完成事项 |

下一阶段补充 WAPE/RMSE 与 Seasonal Naive、ARIMA 比较；验证可训练模型的梯度与优化分支，接入 LSTM、最佳 checkpoint、早停和 MLflow；随后开展 TFT 与调参。原计划中的 raw 入库、数据字典、DuckDB、Airflow 和推理服务仍需按阶段补齐。

价格下降暂不直接标记为真实促销。价格空值保留在数据中，尚未形成建模处理策略；未来价格只有在预测时确实已知价格计划的设定下，才可作为未来协变量。
