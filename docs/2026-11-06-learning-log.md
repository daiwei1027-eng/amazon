# 2026 年 11 月 6 日学习日志：MLflow、历史多变量 LSTM 与受控特征对照

今天将 MinIO、MLflow 和本地训练入口串联起来，完成实验参数、逐轮指标、最佳模型及周期快照的真实记录；拆分训练模块，补齐两种基线的三指标评估；随后把价格、日历、lag、rolling 接入 LSTM，完成与单变量模型的同架构、同训练 seed、同 warmup 特征组对照。

**多变量 LSTM 最佳验证 MAE 为 0.7389587675730387，相较本次单变量对照的 0.7545066070556641 降低约 2.06%；RMSE 降低约 3.32%。** 目前结论限于一个训练 seed、一个固定验证起点。TFT、ARIMA 代表性对照、最终测试集及误差切片尚未完成。

本文按 Asia/Shanghai 的 2026-11-06 记录。证据来自使用者提供的控制台输出与 MLflow 页面、当天代码审阅与测试，以及整理时对本地特征文件、依赖版本和 checkpoint 元数据的核对。核心实现继续由学习者完成；助手按明确委托拆分模块、补充测试及文档。Git 暂存和提交由学习者自行处理。

## 一、统一验证结果

验证范围保持为 CA_1、CA_2 的 100 个共同商品，共 200 条门店商品序列。历史长度 56 天，一次预测 30 天；预测起点 2016-03-23，标签为 2016-03-24 至 2016-04-22，共 6,000 个目标值。

| 模型 | MAE | RMSE | WAPE，比例 | WAPE，百分数 |
| --- | ---: | ---: | ---: | ---: |
| Naive | 0.954333 | 1.934166 | 0.965273 | 96.5273% |
| Seasonal Naive，周期 7 天 | 0.970167 | 1.862301 | 0.981288 | 98.1288% |
| 单变量 LSTM，本次受控对照 | 0.754507 | 1.582347 | 0.763156 | 76.3156% |
| 12 特征 LSTM，本次受控对照 | **0.738959** | **1.529758** | **0.747430** | **74.7430%** |

Naive、Seasonal Naive 使用相同验证数据与 evaluate 循环，直接创建模型，无需恢复训练参数。两组 LSTM 均加载各自 best.pt 后分别计算三个指标，复评 MAE 与训练期最佳结果一致。

Seasonal Naive 的 MAE、WAPE 比 Naive 差，但 RMSE 更好，说明指标对误差分布的侧重不同；具体来源留待逐序列和预测步长切片分析。WAPE 在同一目标集合上分母固定，其排名与 MAE 相同，不能将两者视为独立的两份提升证据。

本次多变量组相对单变量组 MAE 降低约 2.0607%，RMSE 降低约 3.32%。这支持整个历史特征组在当前设置下有帮助，不代表每个特征都有独立贡献，也不等于业务收益或跨季节稳定提升。

## 二、Docker Compose：增加 MLflow，保留 MinIO

实际文件为 `docker-compose.yml`。原先只定义 MinIO，`configs/default.yaml` 中的 mlflow 地址只是应用配置，不会自动启动服务。今天在同一个 services 下增加 mlflow 服务，两者通过宿主机端口访问。

| 服务 | 镜像 | 本机端口 | 数据卷 |
| --- | --- | --- | --- |
| MinIO | `ghcr.io/coollabsio/minio:RELEASE.2025-10-15T17-29-55Z` | 9000 数据 API、9001 管理网页 | `minio_data:/minio_data` |
| MLflow | `ghcr.io/mlflow/mlflow:v3.18.0-full` | 5000 网页与 Tracking API | `mlflow_data:/mlflow_data` |

端口映射均绑定本机 127.0.0.1。MinIO 启动目录同步改为 `/minio_data`，MLflow 使用：

```text
--host 0.0.0.0
--port 5000
--allowed-hosts 127.0.0.1:5000,localhost:5000
--backend-store-uri sqlite:////mlflow_data/mlflow.db
--serve-artifacts
--artifacts-destination /mlflow_data/experiments_records
```

SQLite 保存实验元数据，文件产物保存在 experiments_records；两者由 mlflow_data 卷持久化。MinIO 继续管理原有文件存储，当前 MLflow artifact 使用自身数据卷，没有接到 MinIO。

本次澄清了三种路径的区别：

- `local_path`：Windows 上要上传的实际文件。
- `artifact_path`：当前 Run 内的相对子目录，如 `checkpoints`，不是 Windows 绝对目录。
- `--artifacts-destination`：服务端在容器里的实际保存位置，Python 客户端不用重复传入。

卷名与容器目录可以不同，关键是挂载目标必须覆盖程序实际写入位置。最初将卷挂到 /mlflow_data、程序却写 /mlflow，或 MinIO 写 /data、卷挂 /minio_data，均已通过同步目录修正。

训练仍在 Windows 的 amazon Conda 环境中运行。Docker Desktop 负责容器运行与端口转发，不需要为当前项目额外进入 WSL 重建训练环境。

DuckDB 的原始职责是查询数据，不是 MLflow 的元数据存储。本次明确沿用 SQLite 作为当前单机实验追踪后端；DuckDB 保留为路线图中的可选项。

## 三、Makefile 与健康检查

新增 Makefile，统一提供 config、up、down、status、logs、train、test、health、help。设置 `.DEFAULT_GOAL := help`，直接运行 make 显示帮助；命令行使用 Tab。

| 命令 | 作用 |
| --- | --- |
| make config | 静默解析校验 Compose |
| make up / down | 启动或停止服务；down 不附带删除命名卷选项 |
| make status / logs | 查看服务状态、MLflow 日志 |
| make train / test | 执行训练入口或普通测试 |
| make health | 从 Windows 检查三个服务入口 |

健康检查分别为 MLflow `:5000/health`、MinIO API `:9000/minio/health/ready`、MinIO Console `:9001/`。Windows Makefile 显式使用 curl.exe；设置 HTTP 错误返回失败、连接和总时限，并用 NUL 丢弃正文。9001 检查网页可达性，存储就绪主要由 9000 的专用接口判断。

容器内 MLflow 的 healthcheck 使用 Python urllib 请求 `/health`；没有继续沿用 MinIO 的 `mc ready local`。可达性检查与实际参数/文件写入检查分别验证。

## 四、MLflow 从上传函数到真实训练记录

在 `scripts/mlflow_tracking.py` 实现 log_artifact_file，检查本地文件存在、当前进程有活动 Run，再调用 mlflow.log_artifact。初始化 Tracking URI、experiment 和 Run 的工作由外层入口完成，上传函数只负责将文件归到当前 Run。

先用 artifact-check 测试上传 README，通过 MLflow 的 Model training 视图进入 Run，确认 Artifacts 下存在 `tests/README.md`。GenAI 视图没有 assessment 数据与本次普通文件上传不是同一个概念；本项目选择 Model training 视图查看训练运行。

正式实验使用固定 experiment：`retail-demand-forecasting`；每次执行用模型类名和时间戳生成 run_name。没有为每个时间戳新建 experiment。一次完整训练只创建一个 Run，每轮使用 step=epoch+1；无优化器的基线验证使用 step=0。

当前记录流程：

1. main 读取配置，设置服务器和 experiment，在活动 Run 内记录参数并调用 fit。
2. train_one_epoch 训练完成、整轮统计得到 result 后记录 `train_<metric>`；不上传最后一个 batch 的 loss 作为整轮指标。
3. fit 在 evaluate 后记录 `valid_<metric>`；训练指标不重复记录。
4. 最佳和周期 checkpoint 在保存后上传到同一 Run 的 `checkpoints/`。

metric 使用 log_metric，文件使用 log_artifact；日志值和 step 作为函数参数传入，不能拼进一个文件路径字符串。参数字典通过 log_params 一次写入，当前包含模型、输入特征顺序、input_size、warmup、历史/预测长度、划分日期、模型和优化器配置、sampling/training seed 等。

logs_enable 控制 MLflow 的初始化、Run、参数、指标与文件上传。关闭时用 nullcontext 执行相同训练流程；TensorBoard writer 仍按原逻辑独立记录。开发过程中修正了以下问题：

- 关闭开关时 run_context 未定义，已将上下文选择移到条件分支外。
- 参数上传或最佳模型上传漏掉开关，已加入保护。
- artifact_path 错用本地 Path，已改为相对子目录 checkpoints。
- 周期保存后误传 best.pt，已改为上传刚写入的 checkpoint_file。
- 验证指标大小写不一致、重复训练指标与重复 if 条件，已在审阅中逐项整理。

本轮用户保留模型实例字典和统一 input_size 计算方式；没有继续要求只在选中 LSTM 时创建对应实例。模型输入与 Dataset 特征选择的一致性是本次功能要求。

当前属于 MLflow 基础记录与上传链路完成：配置文件快照、数据内容哈希、设备信息、最佳轮次摘要和三指标复评尚未全部自动写回正式 Run。上传的普通 .pt 文件是 artifact，没有宣称完成模型注册或完整 MLflow flavor 打包。

## 五、拆分训练模块并保留既定接口

经明确委托，将 train.py 中已有函数迁出，保留函数计算逻辑和 fit/main 行为：

| 文件 | 职责 |
| --- | --- |
| scripts/create_dataset.py | 参数检查与 ForecastDataset 创建 |
| scripts/create_dataloader.py | DataLoader 创建 |
| scripts/train_one_epoch.py | 单轮训练、整轮指标与训练指标记录 |
| scripts/evaluate.py | no_grad 验证、形状检查、整轮指标 |
| scripts/train.py | fit、早停、保存与 main 装配 |
| scripts/reload.py | 加载选定 checkpoint，统一计算三指标 |

reload.py 和 test_split_m5.py 已直接从对应模块导入，避免工具函数反向依赖训练入口。拆分时核对函数语法树，确认迁移没有改变计算逻辑，并做人工小数据前向、梯度和指标汇总检查。

fit 继续返回停止时最后一轮的 `(train_loss, valid_loss)`；optimizer=None 时仅验证一次。模型选择只看验证值，best_train_loss 仅记录对应最佳验证轮次的训练指标。训练结束不会自动恢复 best，后续评估显式加载 best.pt。

## 六、抽样 seed 与训练 seed 分开

原先 `data.sampling.seed=42` 只控制共同商品抽样，传给 Polars sample，不控制 PyTorch 模型初始化或窗口 shuffle。

今天增加 `train.seed=42`，在 main 读取配置后、创建模型和 DataLoader 前执行 Python random.seed、NumPy random.seed、torch.manual_seed，并把 training_seed 与 sampling_seed 分别记录到 MLflow。

seed 在每次实验开始设置一次，不在每个 epoch 重置。当前 num_workers=0，没有新增多 worker 初始化流程。基本随机源已固定，但没有启用和验证一整套 CUDA 确定性设置，也没有宣称跨环境或从头重训的逐位一致性。

## 七、核对并修正历史特征

实验子集与生成特征文件今天再次核对：均为 388,200 行、200 条序列，日期 2011-01-29 至 2016-05-22。基础表 12 列，特征表 24 列。

将配置里笔误的 lag_27 改为 lag_28 后，重新执行 features.py，实际输出：

```text
(388200, 24)
Saved: C:\Users\admin\PycharmProjects\amazon\data\feature\sales_feature.parquet
进程已结束，退出代码为 0
```

同时修正 feature_columns 中遗留的 lag_27。列数不变，因为是替换一列。最终特征文件包含 lag_1、lag_7、lag_28，空值数分别为 200、1,400、5,600；rolling mean/std 的 7、14、28 天空值分别为 1,400、2,800、5,600。

滚动统计仍在各门店商品组内先 shift(1)，min_samples 为完整窗口，std 使用 ddof=1。没有将标签区间真实销量产生的派生特征作为预测输入。

### 价格缺失的事实与本轮决策

实际核对结果：销量列无 null；价格空值 82,649 条，全部对应零销量，没有正销量同时价格为空的记录；价格空值也全部位于每条序列首次有效价格之前。

这些现象支持上架前阶段的业务解释，但没有明确上架状态标签。学习者决定本轮不增加 price_missing 特征；原始文件继续保留价格 null，仅在 Dataset 的历史输入副本中填 0，作为价格不可用的占位值。没有把真实价格零与缺失状态混写为已验证业务事实。

本轮未拟合填充值或标准化器；后续需要统计量的填充和缩放只能在训练部分拟合。未来价格没有作为已知输入。

## 八、Dataset 和 LSTM 接入 12 个历史特征

输入列及顺序为：

```python
[
    "units_sold", "sell_price", "is_weekend",
    "lag_1", "lag_7", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_28", "rolling_std_28",
]
```

ForecastDataset 增加可选 feature_columns 和 warmup：未提供列列表时保留单变量 `[56]`；提供 12 列时返回 `[56,12]`。标签始终只取未来 units_sold，形状 `[30]`，没有把未来 12 列全部当成预测目标。

最初只添加多列选择而未处理空值时，第一条序列首个 56 天历史转换后出现 189 个非有限值；随后引入 warmup 和价格输入填充。当前 max lag/rolling window 为 28，历史窗口本身需要完整 56 天，因此最早标签起点为 **28+56=84**，从 0 计数。

| 范围 | warmup=0，早期实验 | warmup=28，受控对照 |
| --- | ---: | ---: |
| 完整 Dataset | 371,200 | 365,600 |
| 训练窗口 | 359,200 | 353,600 |
| 训练批次，batch size 32 | 11,225 | 11,050 |
| 验证 / 测试窗口 | 各 200 | 各 200 |
| 跨标签边界排除 | 11,600 | 11,600 |

两组对照使用相同窗口集合，避免只在多变量组跳过 warmup 造成训练样本范围不同。实测两个 Dataset 的索引与销量标签完全一致；200 条序列的首个多变量输入和标签均有限。

外层 create_dataset 的参数最终采用 warm_up，并显式转传 Dataset 的 warmup。训练入口同步使用 warm_up，修正过一次关键字名称不匹配。LSTM 增加 input_size；二维输入补最后一维，三维输入原样使用，避免多变量分支局部变量未定义或产生四维输入。

```text
单变量批次：[32,56]    → [32,56,1]  → LSTM → [32,30]
多变量批次：[32,56,12] → 保持三维   → LSTM → [32,30]
标签始终为：[32,30]
```

本轮历史多变量输入已真实参与训练，但静态类别编码、未来已知日历输入尚未接入。没有因原始表有 store/item/category 就宣称这些字段已经被模型使用。

## 九、真实训练与最佳 checkpoint 核对

### 接入过程中的早期单变量运行

| Run | 最佳 epoch | 最佳验证 MAE | 停止 epoch | 说明 |
| --- | ---: | ---: | ---: | --- |
| LSTM_20261106_101556_631293 | 11 | 0.749808 | 16 | 验证逐轮指标记录；当时尚未记录完整参数 |
| LSTM_20261106_103238_457988 | 10 | 0.752621 | 15 | 记录 20 项参数与 checkpoint，并补三指标复评 |

这些运行处于补训练 seed/warmup 之前，继续保留历史证据。后者复评为 MAE=0.752621、RMSE=1.576146、WAPE=0.761248；它不替代后面的受控单变量组。

### 本次受控特征组对照

共同配置为 seed=42、lookback=56、horizon=30、warmup=28、hidden_size=64、num_layers=1、AdamW、lr=0.001、weight_decay=0.0001、MAE、epochs 上限 100、patience=5、周期保存间隔 10，train/valid batch size 为 32/8。

| 项目 | 单变量 | 多变量 |
| --- | --- | --- |
| Run | LSTM_20261106_125457_191257 | LSTM_20261106_120946_436213 |
| MLflow Run ID | 5fbb6012ebed4f43a13c1258eac0cb7a | c6e1269acce34f96b7b3960f1fabb4d5 |
| input_size | 1 | 12 |
| 总参数量，含预测头 | 19,102 | 21,918 |
| 最佳 epoch | 11 | 13 |
| 最佳 train MAE | 0.6282500124078351 | 0.6136359888583107 |
| 最佳 valid MAE | 0.7545066070556641 | 0.7389587675730387 |
| 停止 epoch | 16 | 18 |
| 最后一轮 valid MAE，控制台六位 | 0.761591 | 0.745315 |
| 每轮训练时间，约 | 27–31 秒 | 129–131 秒 |

两次均在最佳轮次之后连续五轮未改善时停止。最后打印的 Valid Loss 是停止轮次结果，符合既定返回接口；评价使用恢复后的 best.pt。

整理时实际读取两组 best.pt 与 checkpoint_0001.pt：多变量 best 在第 13 轮，周期快照在第 10 轮、valid MAE=0.7423627141316732；单变量 best 在第 11 轮，周期快照在第 10 轮、valid MAE=0.7646190999348959。权重形状证实分别为 12/1 个输入、hidden_size=64、1 层。

多变量完整训练循环约 39 分钟，单变量约 7–8 分钟，均按控制台进度条估算，未作为精确性能基准。数据准备与模型计算各自的耗时尚未剖析，不能把差异全部归因于网络参数量。

本次尚未进行真实的 1 层与 2 层效果对照。先前只核算过单变量两层模型参数量为 52,382，并在测试中覆盖两层前向和梯度；这些不等于已有两层训练成绩。

## 十、恢复复评与配置一致性

reload.py 复用同一个 Dataset/DataLoader 和 evaluate，恢复选定 best.pt 后循环 MAE、RMSE、WAPE。模型参数固定，仅改变评估指标，不需要重新训练，也没有改变一次训练统一 criterion 的设计。

本次先后遇到两个反向的 shape mismatch：

- 多变量 checkpoint 的 weight_ih_l0 为 `[256,12]`，恢复模型默认 input_size=1。
- 切回单变量 checkpoint 后，当前配置仍有 12 个输入，模型与 `[256,1]` 的 checkpoint 不匹配。

修正方式是同时对齐 checkpoint、Dataset feature_columns 和 LSTM input_size；warmup 均保持 28。仅改模型维度而不改数据输入仍会出错。最终使用者提供的实际复评输出为：

```text
单变量：
MAE: 0.754507
RMSE: 1.582347
WAPE: 0.763156

多变量：
MAE: 0.738959
RMSE: 1.529758
WAPE: 0.747430
```

两次运行均完成 25×3 个验证批次。当前权重文件可以恢复，不是文件损坏。

恢复入口仍有手工操作限制：检查点路径写在脚本中，模型结构与特征来自可变的当前 YAML。整理时磁盘上的 YAML 为 12 特征，而 reload.py 路径为单变量对照；直接再次运行前仍需先匹配这两者。后续宜保存每个 Run 的配置快照及特征顺序，减少手工切换。

## 十一、测试、依赖与文档更新

当天先完成训练模块迁移与导入检查，随后明确委托助手同步 Dataset 和 LSTM 测试。新增 test_dataset.py，扩展 test_lstm.py：

- Dataset：默认单变量兼容、排序与序列隔离、列顺序、价格输入填充且原数据不变、单变量/多变量窗口及标签一致、改变未来值不改变历史输入、warmup 和末尾窗口边界。
- LSTM：默认二维输入、显式三维单变量、12 特征三维输入、1/2 层输出与有限梯度、批次样本独立性。

两文件共 13 项专项检查通过。整理本文时再次运行普通测试，设置外部服务开关为空，结果为 **27 passed、2 skipped，2.52 秒**。新回归未启动完整训练，也未重新执行外部服务上传测试。

MinIO 与 MLflow 集成测试均默认跳过。MLflow 测试需要 RUN_MLFLOW_TESTS=1 才执行，依赖导入位于测试函数内；执行会创建测试 Run 并上传 README，当前没有自动清理 Run。MinIO 沿用 RUN_MINIO_TESTS=1 的独立往返测试，最近一次真实通过仍见 11 月 1 日记录。

依赖核对覆盖全部直接依赖，当前版本与 requirements 原有条目一致。本次仅补充正式使用的 **mlflow==3.18.0**，与 Docker 镜像版本对应；`python -m pip check` 返回 `No broken requirements found.`。未提前加入 TFT、ARIMA、HPO 依赖，也未重新创建干净环境安装整套 requirements。

README 同步更新当前进度、四行三指标结果、受控对照条件、特征与 warmup、单变量/多变量/基线配置切换、Makefile、Docker、MLflow、checkpoint 恢复、测试开关及未完成项。11 月 5 日和更早日志保留当日事实，没有追改其中的历史 lag_27 或旧模型状态。

## 十二、本地数据校验标识

整理时重新计算了两个文件的 SHA-256，作为本文的本地数据快照记录：

```text
data/processed/m5_exp_sub.parquet
ca9703d919af77ec2161513014ce0cb0397e1db4e2a96670e7b59807bf970ba2

data/feature/sales_feature.parquet
30cd88a7c45656563a4a7954c963e360c0eed983bf19e9ab6eccbb00106b391f
```

后者对应今天重建的 lag_28 版本。哈希记录在本文不等于训练入口已经自动记录数据版本；新生成的特征对象是否同步到 MinIO 本轮未核实。训练和本次复评均读取本地 Parquet。

## 十三、面试复盘与下一步

**MLflow 解决了什么？** 将参数、逐轮指标和模型文件关联到同一次运行；本次验证了真实训练接入和 artifact 展示，SQLite 与命名卷承载实验记录及产物。

**为什么需要同架构特征对照？** 本轮单变量与多变量均为一层、hidden_size=64，训练 seed、窗口集合及主要训练配置一致。结果支持整个特征组的作用；不能用不同架构之间的差值直接量化特征贡献。

**为什么不能直接沿用最早的单变量分数？** 早期运行未统一 seed 和 warmup，与当前多变量训练样本范围不同；本轮重跑单变量并采用对应的最佳 checkpoint，保持比较依据清楚。

**可复现完成到哪一步？** 已验证模型加载复评、基本随机源设置、固定数据与日期边界，并记录本地文件哈希；完整配置自动归档、跨环境确定性和完整断点续训仍未完成。

**下一步是什么？** 进入 TFT，先约定历史观测、未来已知日历、静态属性三类输入及训练期拟合的编码/缩放，再接入实际训练和统一评估。之后补 ARIMA 的预固定代表序列对照，固定方案后运行最终测试，完成品类、训练期销量水平和预测步长三类误差切片。

范围继续以 project-roadmap.md 的冻结条件为准；其中阶段起点描述仍为 11 月 3 日，当前进度以今日日志与 README 为准。本轮决定不强制增加 price_missing；Optuna、Airflow 仍为可选项。后续不以增加服务数量代替模型实验与结果解释。

本文、requirements 与 README 按使用者明确委托整理；代码实现未在本次文档任务中改动，Git 暂存、提交和推送由使用者自行完成。
