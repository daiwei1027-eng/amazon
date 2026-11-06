# Retail Demand Forecasting

基于公开 M5 Walmart 数据完成门店商品级未来 30 天销量预测，实践数据处理、防泄漏时间划分、PyTorch 建模、实验追踪和结果解释。本项目不包含 Amazon 内部数据，也不代表公司业务收益。

截至 **2026-11-06**，已完成两种朴素基线、单变量与 12 特征 LSTM 的训练和验证，接入 TensorBoard、MLflow、最佳模型及周期快照保存。当前多变量 LSTM 的最佳验证 MAE 为 **0.738959**；普通测试为 **27 passed、2 skipped**。TFT、ARIMA 代表性对照、最终测试及误差切片尚待完成。

## 验证结果

固定 CA_1、CA_2 两家门店的 100 个共同商品，共 200 条序列。在 2016-03-23 一次预测随后 30 天，验证区间为 2016-03-24 至 2016-04-22，共 6,000 个目标值。

| 模型 | 验证 MAE | 验证 RMSE | 验证 WAPE |
| --- | ---: | ---: | ---: |
| Naive | 0.954333 | 1.934166 | 96.5273% |
| Seasonal Naive，周期 7 天 | 0.970167 | 1.862301 | 98.1288% |
| 单变量 LSTM，受控对照 | 0.754507 | 1.582347 | 76.3156% |
| 12 特征 LSTM，受控对照 | **0.738959** | **1.529758** | **74.7430%** |

MAE、RMSE 的单位为件/门店商品/天，WAPE 以百分数展示；代码返回比例。以上均为验证结果，最终测试集尚未评估。

两组 LSTM 均使用训练 seed=42、warmup=28、hidden_size=64、1 层、AdamW、学习率 0.001、weight decay 0.0001、MAE、patience=5。训练窗口均为 353,600 个，batch size=32，验证 batch size=8。

| LSTM 输入 | Run 名称 | 最佳 epoch | 停止 epoch | 每轮训练耗时，约 |
| --- | --- | ---: | ---: | --- |
| 仅销量 | `LSTM_20261106_125457_191257` | 11 | 16 | 27–31 秒 |
| 12 个历史特征 | `LSTM_20261106_120946_436213` | 13 | 18 | 129–131 秒 |

本次单 seed、固定验证期的特征组对照中，多变量 MAE 降低约 **2.06%**，RMSE 降低约 **3.32%**，单轮训练耗时约为单变量的 4–5 倍。这支持该特征组在当前设置下有帮助，尚不能证明跨起点稳定性、统计显著性或单个特征的独立贡献。

较早的未统一训练 seed/warmup 的 LSTM 运行保留在学习日志中，不混入上述受控对照。更多证据见 [11 月 6 日学习日志](docs/2026-11-06-learning-log.md)。

## 数据与特征

数据来源：[M5 Forecasting Accuracy](https://www.kaggle.com/c/m5-forecasting-accuracy/data) 的公开 Walmart 数据。使用 `sales_train_evaluation.csv`、`calendar.csv`、`sell_prices.csv`；不将与 evaluation 重叠的 `sales_train_validation.csv` 再拼接进来。M5 竞赛预测窗口为 28 天，本项目独立采用 30 天历史留出评估，不直接比较竞赛排行榜分数。

| 范围 | 全量基础产物 | 当前固定实验子集 |
| --- | ---: | ---: |
| 门店 | 10 | CA_1、CA_2 |
| 不同商品 | 3,049 | 100 个共同商品 |
| 门店商品序列 | 30,490 | 200 |
| 每条序列历史 | 1,941 天 | 1,941 天 |
| 日记录 | 59,181,090 | 388,200 |
| 基础字段 | 12 列 | 12 列 |

日期范围为 2011-01-29 至 2016-05-22。抽样候选商品先排序，再使用 sampling seed=42 抽取；当前品类组成为 FOODS 42、HOBBIES 21、HOUSEHOLD 37，属于简单随机抽样。全量预处理产物保留，正式模型实验使用固定子集；没有全量训练效果或耗时的实测结论。

本地主要产物：

```text
data/processed/m5_processed_full.parquet
data/processed/m5_exp_sub.parquet
data/feature/sales_feature.parquet
```

特征文件为 **388,200 行、24 列**。当前配置已将原先笔误的 lag_27 改为 **lag_28**，并重新生成特征文件。lags 为 `[1, 7, 28]`，rolling windows 为 `[7, 14, 28]`；rolling 先在每条序列内 `shift(1)`，要求完整窗口，标准差使用 `ddof=1`。M5 的 wday=1、2 分别对应周六、周日。

当前多变量 LSTM 按以下顺序读取历史特征：

```python
[
    "units_sold", "sell_price", "is_weekend",
    "lag_1", "lag_7", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_28", "rolling_std_28",
]
```

- 原始价格空值保留。在模型的历史输入副本中，`sell_price` 空值填 0，作为价格不可用的占位值；本轮未加入 `price_missing` 特征。
- 子集中 82,649 条价格空值记录全部对应零销量，且均出现在各序列首次有效价格之前，支持“上架前阶段”的解释；数据本身未提供明确上架状态标签。
- 通过 28 天 warmup 排除 lag/rolling 历史不完整的输入窗口。单变量对照也使用相同 warmup。
- 当前未对数值输入做标准化；后续拟合缩放器、填充值或类别词表时，仅使用训练部分。
- 当前 LSTM 只消费历史区间。未来价格、未来真实销量派生的 lag/rolling、静态类别编码和未来日历输入尚未接入。

## 时间窗口与信息边界

先构造 56 天历史与随后 30 天标签的窗口，再按完整标签窗口的日期划分 Subset。

| 集合 | 标签规则 | warmup=28 的窗口数 |
| --- | --- | ---: |
| train | 标签末日不晚于 2016-03-23 | 353,600 |
| valid | 标签首日晚于 2016-03-23，末日不晚于 2016-04-22 | 200 |
| test | 标签首日晚于 2016-04-22，末日不晚于 2016-05-22 | 200，尚未评估 |

warmup=28 时，完整 Dataset 为 365,600 个候选窗口，排除跨标签边界的 11,600 个后得到上述划分。每条序列最早标签起点为第 84 行（从 0 计数），前 28 天用于生成历史特征，再提供完整的 56 天输入。

早期 warmup=0 的实验有 371,200 个候选窗口和 359,200 个训练窗口；固定验证、测试标签区间不变。基线不学习训练参数，对比使用相同验证目标。

验证历史为 2016-01-28 至 2016-03-23，测试历史为 2016-02-27 至 2016-04-22。预测起点之前的已知销量可以使用；标签区间的实际销量只用于评估。训练窗口可以打乱，窗口内部时间顺序不变，验证不打乱。模型和特征选择使用验证集，方案固定后才使用测试集。

指标按全部目标累计：MAE 累计绝对误差后除以元素数；RMSE 累计平方误差后统一开方；WAPE 累计绝对误差后除以真实销量绝对值之和，不平均各批次的 RMSE/WAPE。WAPE 分母为零时报告未定义。当前训练 criterion 为 MAE。

## 环境与安装

已用环境为 Windows x86_64、Python 3.11.18、PyTorch 2.10.0+cu128、CUDA 12.8，开发设备为 i7-12700KF、约 32 GB RAM、RTX 5060 8 GB。训练在 Windows Conda 环境中执行；Docker 运行 MinIO 和 MLflow 服务。

```powershell
conda create -n amazon python=3.11
conda activate amazon
python -m pip install -r requirements.txt
```

`requirements.txt` 固定已实际使用的 Python 直接依赖，包含 **mlflow==3.18.0**，与 Compose 服务版本对应；PyTorch CUDA wheel 使用文件中的额外索引。Docker Desktop 与 GNU Make 单独安装。SQLite 属于 MLflow 服务的存储配置，DuckDB/TFT/ARIMA 相关依赖尚未作为新功能预装。

2026-11-06 本地直接依赖版本核对及 `python -m pip check` 均通过，尚未重新创建干净环境验证整套安装流程。

## 本地运行

从项目根目录执行。首次准备数据时，将三个 CSV 放在 `data/` 下：

```powershell
python -m scripts.inspect_m5
python -m scripts.inspect_m5_scope
python -m etc.features
```

已有产物可以直接复用。变更 lag/rolling 配置时，重新生成特征文件，并同步 `features.feature_columns` 中的列名。

关键实验配置位于 `configs/default.yaml`：

| 实验 | `train.model` | `features.feature_columns` | `features.warm_up` |
| --- | --- | --- | ---: |
| 单变量 LSTM 对照 | `lstm` | `null` | 28 |
| 多变量 LSTM | `lstm` | 上述 12 列列表 | 28 |
| Naive | `naive` | `null` | 可保持 28 |
| Seasonal Naive | `seasonal_naive` | `null` | 可保持 28 |

基线需要一维销量历史，切换模型时同时切换输入列配置。模型类根据 `input_size` 接收 `[batch,56]` 或 `[batch,56,features]`；标签始终为 `[batch,30]`。

常用入口已统一到 Makefile：

```powershell
make help
make config
make up
make health
make train
```

`make status` 查看服务状态，`make logs` 查看 MLflow 日志，`make test` 运行普通测试，`make down` 停止服务并移除容器，保留命名卷。直接运行 `make` 显示帮助。Makefile 命令行使用 Tab 缩进。

配置 `train.logs_enable=true` 时，训练需要能访问 MLflow；设为 false 时跳过 MLflow 调用。TensorBoard 仍由现有 writer 独立记录。无 Make 时，训练与测试入口分别为 `python -m scripts.train` 和 `python -m pytest tests -q`。

## 实验记录、保存与恢复

一次训练对应一个 MLflow Run，由 `main` 创建和结束。参数记录包含模型、输入列及顺序、input_size、warmup、日期边界、sampling/training seed、模型与优化器设置；逐轮指标为 `train_mae`、`valid_mae` 等，基线只记录 step=0 的验证指标。

产物位置：

```text
logs/tensorboard/<模型类名_时间戳>/
data/<模型类名_时间戳>/checkpoints/best.pt
data/<模型类名_时间戳>/checkpoints/checkpoint_0001.pt
```

`fit` 只依据验证指标严格下降更新 best.pt；每隔 10 轮独立保存周期快照，并按 patience=5 早停。开启 MLflow 时，最佳和周期模型上传到该 Run 的 `checkpoints/` 子目录。

**fit 返回停止时最后一轮的 `(train_loss, valid_loss)`，不会自动恢复最佳参数。** 最终评估需要显式加载 best.pt。checkpoint 包含模型、优化器、epoch 与指标；完整断点续训入口、早停历史及随机状态恢复尚未实现。

恢复评估入口：

```powershell
python -m scripts.reload
```

运行前先在 `scripts/reload.py` 选择目标 checkpoint，并让 YAML 的输入列、顺序、warmup 和模型结构与那次训练一致。该脚本目前仍需手动指定检查点路径，不是完整的命令行评估工具。

- 单变量对照：`LSTM_20261106_125457_191257/checkpoints/best.pt`，input_size=1，输入列为 null。
- 多变量对照：`LSTM_20261106_120946_436213/checkpoints/best.pt`，input_size=12，输入列为上面的完整有序列表。

两份模型均已独立加载并在相同验证集上复评 MAE、RMSE、WAPE。仅切换文件路径而沿用另一组输入配置，会产生权重形状不匹配；即使维度相同，特征顺序也需要保持一致。未来应将训练配置快照和数据校验信息一并保存，减少手工切换。

当前 MLflow 已验证参数、逐轮指标和 checkpoint 文件记录。数据内容哈希、完整配置快照、设备等信息尚未全部自动记录；三指标复评目前打印在控制台，未全部写回原 Run。没有将普通 `.pt` artifact 宣称为已注册的 MLflow 模型版本。

## Docker 服务

当前 Compose 包含 MinIO 和 MLflow：

| 服务 | 本机访问地址 | 持久化 |
| --- | --- | --- |
| MinIO 数据 API | `127.0.0.1:9000` | `minio_data:/minio_data` |
| MinIO 管理网页 | [localhost:9001](http://localhost:9001) | 同上 |
| MLflow 网页及 Tracking API | [localhost:5000](http://localhost:5000) | `mlflow_data:/mlflow_data` |

MinIO 使用 `ghcr.io/coollabsio/minio:RELEASE.2025-10-15T17-29-55Z`，MLflow 使用 `ghcr.io/mlflow/mlflow:v3.18.0-full`。MLflow 的实验元数据位于容器内 `/mlflow_data/mlflow.db`，文件产物位于 `/mlflow_data/experiments_records`。它们都由命名卷持久保存；MinIO 数据与 MLflow 产物当前分别管理。

在根目录自行创建 `.env` 并填写实际值，不提交凭据：

```dotenv
MINIO_ROOT_USER=replace_with_your_user
MINIO_ROOT_PASSWORD=replace_with_your_password
```

`make health` 检查 MLflow `/health`、MinIO API `/minio/health/ready`，以及 MinIO 管理网页 `/`。这是可达性/就绪检查，文件上传能力另外通过集成测试确认。

数据上传入口为 `python -m scripts.ingest_minio_bucket`。当前训练直接读取本地 Parquet；新生成的 lag_28 特征对象是否已同步到 MinIO 尚未核实。raw 自动入库与 DuckDB 不作为当前冻结前置条件。

查看 TensorBoard：

```powershell
python -m tensorboard.main --logdir logs/tensorboard --port 6006 --host 127.0.0.1
```

## 测试

```powershell
python -m pytest tests -q
```

2026-11-06 整理本文时实测：**27 passed、2 skipped**。覆盖数据处理、抽样、日历/lag/rolling、时间边界、单变量/多变量 Dataset、价格输入填充、warmup、未来值修改不影响历史输入，以及一层/两层 LSTM 的输入形状、有限梯度与批次独立性。两层模型当前只有测试覆盖，尚无真实层数对照成绩。

两个真实服务集成测试默认跳过，显式运行方式：

```powershell
$env:RUN_MINIO_TESTS = "1"
python -m pytest tests/test_ingest.py -q
Remove-Item Env:RUN_MINIO_TESTS

$env:RUN_MLFLOW_TESTS = "1"
python -m pytest tests/test_mlflow_tracking.py -q
Remove-Item Env:RUN_MLFLOW_TESTS
```

MinIO 测试上传、下载、比较并清理自己的测试对象；MLflow 测试会创建 `artifact-check` Run 并上传 README，当前测试不会自动删除该 Run。今天已完成 MLflow 上传及训练记录的真实联调；最近一次 MinIO 真实往返通过记录仍为 11 月 1 日，本次普通回归未重跑外部服务测试。

## 代码与后续范围

| 文件 | 职责 |
| --- | --- |
| `scripts/inspect_m5.py`、`scripts/inspect_m5_scope.py` | 全量预处理与固定抽样 |
| `etc/features.py`、`etc/dataset.py` | 历史特征生成与窗口张量 |
| `scripts/create_dataset.py`、`scripts/create_dataloader.py` | 数据装配；对外使用 warm_up，转传 Dataset 的 warmup |
| `scripts/split_m5.py` | 按完整标签日期划分窗口 |
| `scripts/train_one_epoch.py`、`scripts/evaluate.py` | 训练、全局指标汇总与验证 |
| `scripts/train.py` | fit、早停、checkpoint、实验和参数装配 |
| `scripts/mlflow_tracking.py` | 活动 Run 内的文件上传 |
| `scripts/reload.py` | 加载选定模型，输出三个验证指标 |
| `models/baseline.py`、`models/SeasonNaive.py`、`models/lstm.py` | 两种基线和 LSTM |

当前已完成历史多变量输入与同架构特征组对照。下一项为 TFT：定义历史观测、未来已知日历和静态属性三类输入；随后完成 ARIMA 代表性对照、固定方案后的测试与三类误差切片，最后整理展示材料。

范围仍遵循[项目路线与验收标准](docs/project-roadmap.md)，其阶段起点描述保留 11 月 3 日状态，最新实现以本文和按日学习日志为准。Optuna 与 Airflow 为可选扩展；FastAPI、TorchServe、K8s、Kafka、Redis 不属于冻结前置条件。

当前限制包括单个训练 seed 和验证起点、子集覆盖有限、尚未标准化、完整数据版本自动追踪与断点续训未完成，以及 RMSE 零误差反传和 WAPE 全零训练批次的边界策略尚待处理。当前真实训练统一使用 MAE；最终测试和业务收益均未宣称完成。
