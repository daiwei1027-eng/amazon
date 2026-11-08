# Retail Demand Forecasting

基于公开 M5 Walmart 数据完成门店商品级未来 30 天销量预测，实践数据处理、防泄漏时间划分、PyTorch 建模、实验追踪和结果解释。本项目不包含 Amazon 内部数据，也不代表公司业务收益。

截至 **2026-11-08**，已完成两种朴素基线、单变量/12 特征/15 特征 LSTM，以及自定义 PyTorch TFT 的训练、验证和最佳 checkpoint 恢复。TFT 的 year_offset 版本最佳验证 MAE 为 **0.736058**，12 特征 LSTM 的 RMSE 最低，为 **1.529758**；普通测试为 **71 passed、2 skipped**。当前已接入 TensorBoard 和 MLflow，后续主要为 ARIMA 代表性对照、实验元数据补齐、最终测试与误差切片。

## 验证结果

固定 CA_1、CA_2 两家门店的 100 个共同商品，共 200 条序列。在 2016-03-23 一次预测随后 30 天，验证区间为 2016-03-24 至 2016-04-22，共 6,000 个目标值。

| 模型 | 验证 MAE | 验证 RMSE | 验证 WAPE |
| --- | ---: | ---: | ---: |
| Naive | 0.954333 | 1.934166 | 96.5273% |
| Seasonal Naive，周期 7 天 | 0.970167 | 1.862301 | 98.1288% |
| 单变量 LSTM，受控对照 | 0.754507 | 1.582347 | 76.3156% |
| 12 特征 LSTM，受控对照 | 0.738959 | **1.529758** | 74.7430% |
| 15 特征 LSTM，加入日历 | 0.741551 | 1.550951 | 75.0052% |
| TFT，原始 year | 0.738534 | 1.556120 | 74.7000% |
| TFT，year_offset | **0.736058** | 1.577538 | **74.4496%** |

MAE、RMSE 的单位为件/门店商品/天，WAPE 以百分数展示；代码返回比例。以上均为验证结果，最终测试集尚未评估。

11 月 6 日的单变量/12 特征 LSTM 对照，以及本日的 15 特征 LSTM、TFT，均使用训练 seed=42、warmup=28、hidden_size=64、1 层、AdamW、学习率 0.001、weight decay 0.0001、MAE、patience=5。训练窗口均为 353,600 个，batch size=32，验证 batch size=8；本日实际 TFT 为单头、Dropout=0。

| 模型/输入 | Run 名称 | 最佳 epoch | 停止 epoch | 参数量 | 每轮训练耗时，约 |
| --- | --- | ---: | ---: | ---: | --- |
| LSTM，仅销量 | `LSTM_20261106_125457_191257` | 11 | 16 | 19,102 | 27–31 秒 |
| LSTM，12 个历史特征 | `LSTM_20261106_120946_436213` | 13 | 18 | 21,918 | 129–131 秒 |
| LSTM，15 个历史特征 | `LSTM_20261108_092617_264492` | 4 | 9 | 22,686 | 后续轮 132–147 秒，首轮 168 秒 |
| TFT，原始 year | `TFT_20261108_114514_027332` | 4 | 9 | 685,673 | 8 分 47 秒–8 分 58 秒 |
| TFT，year_offset | `TFT_20261108_133756_506473` | 4 | 9 | 685,673 | 8 分 49 秒–8 分 57 秒 |

本次单 seed、固定验证期的同架构对照中，12 特征 LSTM 相对单变量的 MAE 降低约 **2.06%**，RMSE 降低约 **3.32%**，单轮训练耗时约为单变量的 4–5 倍。这支持该特征组在当前设置下有帮助，尚不能证明跨起点稳定性、统计显著性或单个特征的独立贡献。

year_offset 版 TFT 相对原始 year 版的 MAE 降低约 **0.335%**，RMSE 升高约 **1.38%**；相对 12 特征 LSTM 的 MAE 降低约 **0.392%**，RMSE 仍更高。RMSE 对较大误差更敏感，具体误差来自哪些序列或时间步仍需切片分析。TFT 参数量约为 15 特征 LSTM 的 30.2 倍，每轮耗时约 4 倍；本次没有全面指标优势。

两版 TFT 均使用未来日历和静态信息；当前 year_offset 版与 15 特征 LSTM 使用同一年份编码，但输入可用范围仍不同，不能将模型间差值全部归因于结构。同一目标集合的 WAPE 与 MAE 排名一致，不作为独立提升证据。

较早的未统一训练 seed/warmup 的 LSTM 运行保留在历史日志中。更多证据见 [11 月 6 日学习日志](docs/2026-11-06-learning-log.md)和 [11 月 8 日学习日志](docs/2026-11-08-learning-log.md)。当前模型选择尚未以最终测试集成绩作判断。

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

特征文件现为 **388,200 行、25 列**。11 月 6 日将 lag_27 笔误改为 **lag_28**；11 月 8 日新增 `year_offset=(year-2011)`，原始 date、year 保留。lags 为 `[1, 7, 28]`，rolling windows 为 `[7, 14, 28]`；rolling 先在每条序列内 `shift(1)`，要求完整窗口，标准差使用 `ddof=1`。M5 的 wday=1、2 分别对应周六、周日。

12 特征 LSTM 对照按以下顺序读取历史特征：

```python
[
    "units_sold", "sell_price", "is_weekend",
    "lag_1", "lag_7", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_28", "rolling_std_28",
]
```

当前 YAML 的 LSTM 列表在上述末尾追加 `month`、`year_offset`、`wday`，共 15 列。直接输入原始大数值 year 的 LSTM 探索运行曾退化为接近零预测，改用年份偏移后恢复正常训练；全部数值特征的统一标准化尚未实现。

TFT 通过独立的 TFTDataset 返回字典，实际使用：

| 输入 | 字段及顺序 | 批次形状 |
| --- | --- | --- |
| past_observed | units_sold、sell_price、lag_1、lag_7、lag_28、rolling_mean_7、rolling_std_7、rolling_mean_14、rolling_std_14、rolling_mean_28、rolling_std_28 | `[B,56,11]` |
| past_known / future_known | is_weekend、month、year_offset、wday | `[B,56,4]` / `[B,30,4]` |
| static | store_id、item_id 的整数 ID | `[B,2]`，long |
| target | 未来 units_sold | `[B,30]`，float32 |

当前 TFT 与 15 特征 LSTM 均使用 **year_offset**。较早的 TFT Run `TFT_20261108_114514_027332` 使用原始 **year**；恢复旧 checkpoint 时必须切回当次输入语义，不能只检查维度。cat_id、dept_id 等虽在表中，本轮没有作为 TFT 输入。

- 原始价格空值保留。在模型的历史输入副本中，`sell_price` 空值填 0，作为价格不可用的占位值；本轮未加入 `price_missing` 特征。
- 子集中 82,649 条价格空值记录全部对应零销量，且均出现在各序列首次有效价格之前，支持“上架前阶段”的解释；数据本身未提供明确上架状态标签。
- 通过 28 天 warmup 排除 lag/rolling 历史不完整的输入窗口。单变量对照也使用相同 warmup。
- 需要数据统计量的缩放、填充值估计和销量分层阈值，仅在训练部分拟合。静态 ID 使用固定实体目录 unique 后排序编号；当前全量与训练期的门店、商品集合一致，不使用未来销量统计。新类别策略尚未定义。
- LSTM 只消费历史区间；TFT 已消费未来日历和静态 ID。未来价格及未来真实销量派生的 lag/rolling 不作为已知输入。

TFT 的数值列逐变量 Linear 投影，静态 ID 经过 Embedding；三路 VSN、四路静态上下文 GRN、LSTM 编码/解码、共享 V 的可解释因果注意力及门控残差组成模型。当前输出头为 Linear(H,1)，按 MAE 学习每个未来时间步的单值预测，未实现原论文的多分位数损失。注意力和变量权重尚未导出为解释图。

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

`requirements.txt` 固定已实际使用的 Python 直接依赖，包含 **mlflow==3.18.0**，与 Compose 服务版本对应；PyTorch CUDA wheel 使用文件中的额外索引。Docker Desktop 与 GNU Make 单独安装。SQLite 属于 MLflow 服务的存储配置；TFT 使用项目自定义 PyTorch 模块，无需额外 TFT 库。本日未新增依赖，ARIMA 尚未接入。

2026-11-06 本地直接依赖版本核对及 `python -m pip check` 均通过，尚未重新创建干净环境验证整套安装流程。

## 本地运行

从项目根目录执行。首次准备数据时，将三个 CSV 放在 `data/` 下：

```powershell
python -m scripts.inspect_m5
python -m scripts.inspect_m5_scope
python -m etc.features
```

已有产物可以直接复用。变更 lag/rolling 或日历编码时，重新生成特征文件，并同步 `train.lstm.feature_columns` 和 `train.tft` 中相应的输入清单。

关键实验配置位于 `configs/default.yaml`：

| 实验 | `train.model` | 输入配置 | `features.warm_up` |
| --- | --- | --- | ---: |
| 单变量 LSTM 对照 | `lstm` | `train.lstm.feature_columns: null` | 28 |
| 12 特征 LSTM | `lstm` | `train.lstm.feature_columns` 为上述 12 列 | 28 |
| 15 特征 LSTM | `lstm` | 原 12 列，加 month、year_offset、wday | 28 |
| TFT | `tft` | `train.tft` 的历史观测、已知日历、静态字段三组配置 | 28 |
| Naive | `naive` | 入口自动选择单变量销量历史 | 可保持 28 |
| Seasonal Naive | `seasonal_naive` | 入口自动选择单变量销量历史 | 可保持 28 |

当前默认 `train.model=tft`。保留完整 YAML 配置并修改对应键即可选择模型。ForecastDataset 为 LSTM 返回 `[B,56]` 或 `[B,56,F]`，TFTDataset 返回上述字典；标签统一 `[B,30]`。训练和 evaluate 已兼容字典的逐 Tensor 设备转换，静态 ID 保持 long。

当前 train.py 已读取 TFT 的 hidden_size、num_layers，但尚未显式透传配置中的 dropout、num_heads；本次实际默认值为 0.0/1，与 YAML 相同。后续调整这两个参数前需同步构造调用，不能仅改配置就宣称实验参数生效。

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

一次训练对应一个 MLflow Run，由 main 创建和结束。共通参数包含模型、历史输入列及顺序、input_size、warmup、日期边界、sampling/training seed 和优化器设置；LSTM 分支还记录 hidden_size、num_layers。逐轮指标为 train_mae、valid_mae 等，基线只记录 step=0 的验证指标。TFT 专属结构参数及三类输入/映射的完整归档尚待补齐；其日志中的 input_size=15 是历史原始列数，内部 LSTM input_size 实际为 H=64。

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
python -m scripts.reload_tft
```

LSTM 使用 scripts/reload.py，TFT 使用 scripts/reload_tft.py，均只复评验证集。LSTM 脚本手动选择 checkpoint 路径，当前指向本日 15 特征运行；TFT 脚本通过 RUN_NAME 选择运行，当前默认指向 year_offset 版 `TFT_20261108_133756_506473`。两者从可变的当前 YAML 读取输入与结构，仍需匹配当次训练配置，不是全自动的模型恢复工具。

- 单变量对照：`LSTM_20261106_125457_191257/checkpoints/best.pt`，input_size=1，输入列为 null。
- 多变量对照：`LSTM_20261106_120946_436213/checkpoints/best.pt`，input_size=12，输入列为上面的完整有序列表。
- 15 特征 LSTM：`LSTM_20261108_092617_264492/checkpoints/best.pt`，input_size=15，原 12 列后追加 month、year_offset、wday。
- TFT，原始 year：`TFT_20261108_114514_027332/checkpoints/best.pt`，已知日历为 is_weekend、month、year、wday。
- TFT，year_offset：`TFT_20261108_133756_506473/checkpoints/best.pt`，三类输入按当前上表，已知日历为 is_weekend、month、year_offset、wday。

这些最佳模型均已加载并复评三项指标。原始 year 版 TFT 的恢复 MAE=0.7385341339111328，与保存值完全一致；year_offset 版保存 MAE=0.7360584182739258，使用者恢复输出 MAE=0.736058、RMSE=1.577538、WAPE=0.744496，MAE 与保存值在展示精度上一致。两版均 best epoch=4、停止于第 9 轮，未产生第 10 轮周期快照。

仅切换文件路径而沿用另一组输入配置，可能产生权重形状不匹配；维度相同也必须保持特征顺序、年份编码和静态编号一致。TFT checkpoint 未包含静态映射，reload 按训练时相同数据和排序规则重建。后续应将配置、输入 schema、映射和数据标识一并归档。

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

数据上传入口为 `python -m scripts.ingest_minio_bucket`。当前训练直接读取本地 Parquet；包含 year_offset 的当前 25 列特征对象是否已同步到 MinIO 尚未核实。raw 自动入库与 DuckDB 不作为当前冻结前置条件。

查看 TensorBoard：

```powershell
python -m tensorboard.main --logdir logs/tensorboard --port 6006 --host 127.0.0.1
```

## 测试

```powershell
python -m pytest tests -q
```

2026-11-08 整理本文时普通回归实测：**71 passed、2 skipped，4.62 秒**。覆盖原有数据、窗口、LSTM 检查，并新增：

| 测试文件 | 用例数 | 主要覆盖 |
| --- | ---: | --- |
| tests/test_tft_dataset.py | 10 | 三路输入、静态 ID 顺序、warmup、未来信息隔离、DataLoader 与时间划分 |
| tests/test_grn.py | 11 | 残差、上下文广播、Dropout、输入不变、有限梯度 |
| tests/test_vsn.py | 8 | 权重归一化、加权结果、形状、上下文与梯度 |
| tests/test_tft.py | 15 | 输出、全部参数梯度、因果性、批次独立、Dropout、AdamW 更新与 checkpoint 恢复 |

双层、四头和非零 Dropout 的测试覆盖不等于真实效果对照；本日真实 TFT 仍为单层、单头、Dropout=0。正式 train_one_epoch/evaluate 的字典输入另外完成真实验证 batch 联调，尚未固化为专门训练循环测试文件。

两个真实服务集成测试默认跳过，显式运行方式：

```powershell
$env:RUN_MINIO_TESTS = "1"
python -m pytest tests/test_ingest.py -q
Remove-Item Env:RUN_MINIO_TESTS

$env:RUN_MLFLOW_TESTS = "1"
python -m pytest tests/test_mlflow_tracking.py -q
Remove-Item Env:RUN_MLFLOW_TESTS
```

MinIO 测试上传、下载、比较并清理自己的测试对象；MLflow 测试会创建 artifact-check Run 并上传 README，当前测试不会自动删除该 Run。11 月 6 日已完成上传及训练记录联调，本日 TFT 真实训练也生成了 MLflow Run；本次普通回归显式关闭两个服务测试开关，没有重新执行外部服务测试。最近一次独立 MinIO 往返证据仍为 11 月 1 日。

## 代码与后续范围

| 文件 | 职责 |
| --- | --- |
| `scripts/inspect_m5.py`、`scripts/inspect_m5_scope.py` | 全量预处理与固定抽样 |
| `etc/features.py`、`etc/dataset.py`、`etc/tft_dataset.py` | 特征生成、单变量/多变量窗口和 TFT 字典输入 |
| `scripts/create_dataset.py`、`scripts/create_dataloader.py` | 数据装配；对外使用 warm_up，转传 Dataset 的 warmup |
| `scripts/split_m5.py` | 按完整标签日期划分窗口 |
| `scripts/train_one_epoch.py`、`scripts/evaluate.py` | 训练、全局指标汇总与验证 |
| `scripts/train.py` | fit、早停、checkpoint、实验和参数装配 |
| `scripts/mlflow_tracking.py` | 活动 Run 内的文件上传 |
| `scripts/reload.py` | 加载选定模型，输出三个验证指标 |
| `scripts/reload_tft.py` | 重建静态映射、恢复 TFT 最佳模型并复评验证集 |
| `models/baseline.py`、`models/SeasonNaive.py`、`models/lstm.py` | 两种基线和 LSTM |
| `models/grn.py`、`models/vsn.py`、`models/attention.py`、`models/tft.py` | 自定义 TFT 组件与完整单值预测 |

当前已完成历史多变量特征组对照及 TFT 训练/恢复闭环。下一项为按训练数据预先固定代表序列的 ARIMA 对照；同时补齐实验配置、静态映射和实际参数归档。之后固定方案，评估最终测试并完成品类、训练期销量水平和预测步长三类误差切片。

范围仍遵循[项目路线与验收标准](docs/project-roadmap.md)，当前进度已同步到 11 月 8 日，过程证据保留在按日学习日志中。Optuna 与 Airflow 为可选扩展；FastAPI、TorchServe、K8s、Kafka、Redis 不属于冻结前置条件。

当前限制包括单个训练 seed 和验证起点、子集覆盖有限、尚未标准化、完整数据版本自动追踪与断点续训未完成，以及 RMSE 零误差反传和 WAPE 全零训练批次的边界策略尚待处理。当前真实训练统一使用 MAE；最终测试和业务收益均未宣称完成。
