# 2026 年 11 月 3 日学习日志：窗口 Dataset、统一执行框架与首个基线结果

今天实现了历史滚动统计、特征保存与上传，并将数据划分职责调整到窗口 Dataset 层。完成 PyTorch Dataset/DataLoader、按标签日期划分 Subset、Naive 模型、MAE、训练单轮、验证与 fit 调度，接入 TensorBoard，获得第一份真实验证结果。随后将特征生成入口改为读取完整实验子集，生成覆盖全部实验日期的特征文件，并报告再次运行 Naive 得到相同指标。

**Naive 验证 MAE 为 0.954333 件/门店商品/天，覆盖 200 条序列、未来 30 天，共 6,000 个预测值。**普通测试实测为 **13 passed、1 skipped**。当前模型只使用历史销量，没有使用工程特征，也没有评估最终测试集。

## 一、完成历史滚动特征与完整实验特征文件

在 `etc/features.py` 新增 `add_rolling_features()`：先按门店、商品、日期排序，对每条序列的销量先 `shift(1)`，再计算窗口均值与标准差，整个操作使用 `over(["store_id", "item_id"])` 限定在组内。

当前窗口为 `[7, 14, 28]`，每个窗口生成 `rolling_mean_<window>`、`rolling_std_<window>`。`min_samples=window` 要求完整的非空历史，历史不足保留 null；标准差明确采用 `ddof=1`。

三天窗口测试的预期为：

| 商品 | 原销量 | 前三天均值 | 前三天样本标准差 |
| --- | --- | --- | --- |
| A | `[2, 4, 6, 8, 10]` | `[None, None, None, 4, 6]` | `[None, None, None, 2, 2]` |
| B | `[10, 20, 30, 40, 50]` | `[None, None, None, 20, 30]` | `[None, None, None, 10, 10]` |

该测试检查不包含当前行销量、不跨商品继承窗口，以及历史不足的空值位置。

将日历、lag 与 rolling 函数串联，最初生成训练期的 `data/feature/train_feature.parquet`：376,200 行、24 列，总销量 317,916。随后将输入改为 `data/processed/m5_exp_sub.parquet`，输出改为 `data/feature/sales_feature.parquet`，以便在完整历史上构造窗口后划分 Dataset。

最终特征文件核验为 **388,200 行、24 列**，覆盖 2011-01-29 至 2016-05-22，保留 200 条门店商品序列，总销量 330,222，价格空值 82,649。原有 12 列与实验子集排序后一致。新增字段及空值核验为：

| 特征 | 空值数 |
| --- | ---: |
| year、month、is_weekend | 均为 0 |
| lag_1 | 200 |
| lag_7 | 1,400 |
| lag_27 | 5,400 |
| rolling_mean_7、rolling_std_7 | 各 1,400 |
| rolling_mean_14、rolling_std_14 | 各 2,800 |
| rolling_mean_28、rolling_std_28 | 各 5,600 |

当前实际配置是 `lags: [1, 7, 27]`，因此生成的是 lag_27，不能在文档中写成 lag_28。窗口长度乘以 200 条序列，与相应的历史不足空值数量一致。

写入时遇到父目录不存在，使用完整文件路径的 `.parent.mkdir(parents=True, exist_ok=True)` 创建目录。随后通过上传脚本写入 `silver/feature/train_feature.parquet`，控制台输出确认上传成功。

上述上传对应较早的训练期文件。当前上传配置已指向 `silver/feature/sales_feature.parquet`；完整特征文件在本地生成已确认，本次尚未核实新对象上传结果。

## 二、统一的数据窗口约定

明确区分四个组件：Dataset 定义样本；DataLoader 组批；nn.Module 定义张量前向计算与参数状态；实验平台记录参数、指标和产物。Naive 可以使用 nn.Module 接口，但没有可学习参数，不运行反向传播。

本次样本约定为：过去 56 天销量预测随后 30 天销量。最小实现暂时只返回销量张量：

```text
单条样本：past_target [56]，target [30]
训练批次：past_target [32, 56]，target [32, 30]
验证批次：past_target [8, 56]，target [8, 30]
```

没有把 24 列直接转换成输入张量：其中包含字符串标识、日期以及价格和历史特征空值，后续加入多变量输入前需要明确数值列、缺失值和缩放策略。

### ForecastDataset 的结构

类独立放在 `etc/dataset.py`：

1. 按门店、商品、日期排序并 `partition_by()`，保存每条门店商品序列。
2. 用 `sample_index` 保存 `(series_index, target_start)`。
3. `__len__()` 返回窗口数量。
4. `__getitem__()` 根据索引取历史和未来标签，转换为 float32 Tensor 后以元组返回。

`series_index` 指向哪条门店商品序列；`target_start` 是该序列内未来标签的起始行号，不是日期，也不是全局样本编号。

对长度为 n 的序列，标签起点的合法范围是：

```text
lookback <= target_start <= n - horizon
```

因此使用 `range(lookback, n - horizon + 1)`。历史切片为 `[target_start-lookback:target_start]`，标签切片为 `[target_start:target_start+horizon]`。

样本索引使用 `append((i, m))` 添加一个元组，不能向 append 传两个位置参数。DataLoader 返回两个组批张量，所以使用 `past_target, target = ...` 解包，不能再用字典键读取。

## 三、将 split 的职责改为划分窗口索引

最初将日记录保存为三个独立 Parquet，标签日期本身没有分错，但每份验证、测试文件只有 30 天，不能独立构成 56+30 天样本，需要额外补历史。

本次改为从完整实验子集构造 Dataset，再按完整标签窗口的日期选择三个 Subset。文件存储范围与预测目标范围分开：一个评估样本需要 56 天已发生的历史和 30 天未来标签，只有最后 30 天参与预测误差计算。

### 当前固定边界

| 集合 | 历史与标签规则 |
| --- | --- |
| train | 所有标签日期不晚于 2016-03-23 |
| valid | 历史为 2016-01-28 至 2016-03-23；标签为 2016-03-24 至 2016-04-22 |
| test | 历史为 2016-02-27 至 2016-04-22；标签为 2016-04-23 至 2016-05-22 |

`split_dataset_by_time()` 接收完整 ForecastDataset 与三个截止日期，遍历样本索引，读取标签的首日与末日：

- 训练：`target_last <= train_end`。
- 验证：`target_first > train_end` 且 `target_last <= validation_end`。
- 测试：`target_first > validation_end` 且 `target_last <= test_end`。
- 其余跨集合边界的标签窗口排除。

函数返回三个 Subset，不复制实际窗口。训练边界必须包含等号，否则会丢掉标签正好结束于训练截止日的每条序列最后一个窗口。

当前实测：

| 范围 | 样本数 | 批次数 |
| --- | ---: | ---: |
| 完整 Dataset | 371,200 | — |
| train | 359,200 | 11,225，batch size 32 |
| valid | 200 | 25，batch size 8 |
| test | 200 | 本次没有进行预测评估 |

另外 11,600 个窗口跨越标签边界，被主动排除。验证、测试各 200 个窗口，每个窗口 30 个标签，对应各 6,000 个目标值。

### 防泄漏的理解

验证历史与训练期末尾 56 天相同是正常的。训练样本自身还要在训练期内留足未来 30 天标签，因此最后一个训练样本的历史截止 2016-02-22，标签为 2016-02-23 至 2016-03-23；验证样本的历史截止 2016-03-23。

测试预测起点为 2016-04-22，此时验证期销量已经发生，可以作为历史输入；测试期实际销量仍只能用于标签。测试标签长度为 30 天不是泄漏原因，关键是预测输入是否使用了该起点之后的信息。

单行特征先 shift(1) 也不能自动消除多步泄漏。例如在 3 月 23 日一次预测未来 30 天时，3 月 25 日的 lag_1 会引用尚未知晓的 3 月 24 日真实销量，不能作为未来已知输入。

## 四、窗口划分的小表测试

将测试从 DataFrame 日期筛选更新为 Dataset/Subset 的索引及实际张量检查。使用 3 月 21 至 27 日连续七天、销量 `[10,11,12,13,14,15,16]`，lookback=1、horizon=2，边界为 23、25、27 日。

| 全局样本索引 | 标签起点行号 | 标签日期 | 归属 |
| --- | ---: | --- | --- |
| 0 | 1 | 22～23 日 | train |
| 1 | 2 | 23～24 日 | 跨边界排除 |
| 2 | 3 | 24～25 日 | valid |
| 3 | 4 | 25～26 日 | 跨边界排除 |
| 4 | 5 | 26～27 日 | test |

测试确认三个 Subset.indices 为 `[0]`、`[2]`、`[4]`，并分别核对历史与标签销量。样本编号与 target_start 不同，不能将行号 3、5 当作验证和测试的样本索引。

PyTorch Tensor 转列表使用 `.tolist()`，Polars Series 使用 `.to_list()`。输入原始数据还必须包含门店、商品、日期和销量列，不能只提供日期。

## 五、Naive 模型与统一训练验证框架

### Naive 的前向计算

`models/baseline.py` 中实现无可学习参数的 NaiveBaseline：对每个 batch 样本取历史最后一个销量，复制为未来 horizon 天预测。

输入为 `[batch_size, 56]`，输出必须为 `[batch_size, 30]`，不能创建只有 `[30]` 的一维结果。`output[:, i]` 表示所有样本的第 i 天预测，`output[i]` 则指第 i 个样本。

当前实现用 `past_target.new_zeros((batch_size, horizon))` 创建输出，保持输入的 device 与 dtype；也可以用最后一列的 `repeat(1, horizon)` 表达同一计算。零销量作为真实最后观测保留，不改为最后一次非零销量。

### MAE

`etc/losses.py` 先检查预测与标签形状完全一致，再返回 `(prediction-target).abs().mean()`。返回标量 Tensor，以保留后续可训练模型需要的梯度；打印或统计时在外部调用 `.item()`。

### train_one_epoch、evaluate 与 fit

`scripts/train.py` 统一组装数据和执行流程：

| 函数 | 职责 |
| --- | --- |
| train_one_epoch | 训练模式、清空梯度、前向、损失、backward、optimizer.step，返回整轮 MAE |
| evaluate | 评估模式、no_grad、前向与整体误差汇总 |
| fit | 无优化器时只评估一次；有优化器时逐 epoch 训练、验证并记录结果 |

每批 MAE 已经是平均值，所以整轮累计 `loss.item() * target.numel()`，再除以全部元素数量。训练循环不能用 dataloader[0] 取批次，也不直接把带梯度 loss Tensor 跨批累加。

不同 epoch 的平均误差分别记录，不将它们相加后作为最终模型指标。当前 fit 返回最后一轮结果；Naive 返回 `(None, valid_loss)`，不能把未更新的初始累计值 0 当成验证误差。

优化器是 train_one_epoch 的关键字参数，调用时明确写 `optimizer=optimizer`。当前 Naive 没有可学习参数，实际运行的是 optimizer=None 分支，尚未以 LSTM 等模型验证梯度训练、最佳 checkpoint 或早停。

tqdm 包装放在每次训练和评估的循环中，main 中保留原始 DataLoader。只创建进度条不会自动执行迭代；此前出现 0/11225 表示还没遍历，并非训练卡住。

## 六、首个验证结果与 TensorBoard

运行 Naive，在固定验证起点一次预测未来 30 天，得到：

| 检查项 | 结果 |
| --- | ---: |
| 验证序列数 | 200 |
| 每条预测天数 | 30 |
| 预测值总数 | 6,000 |
| 绝对误差总和，独立按原始销量核算 | 5,726 |
| MAE，5,726 / 6,000 | 0.9543333333 |
| TensorBoard 中的 float32 标量 | 0.9543333649635315 |
| 验证真实销量总和 | 5,932 |

生成完整 `sales_feature.parquet` 后，使用者报告再次以 feature 为输入运行，25/25 个验证批次完成，退出代码为 0，控制台输出 `Naive validation MAE: 0.954333`。指标与原先结果一致：当前 Dataset 只提取历史销量，尚未把日历、价格、lag 或 rolling 列送入模型。

本次读取事件文件确认：run 为 `naive_20261103_115106_346113`，tag 为 `MAE/valid`，step 为 0。截图显示的 0.9543 是该数值的显示精度，不是准确率。

当前只有一个验证点是正常的：Naive 只评估一次，没有训练损失曲线。接入可训练模型后，逐 epoch 分别写入 `MAE/train`、`MAE/valid`。

通过 `torch.utils.tensorboard.SummaryWriter` 写入独立时间戳目录，main 用 with 管理 writer，结束时关闭并刷新日志。位置为 `logs/tensorboard/naive_<时间戳>/`，现有 .gitignore 已排除 logs。

查看命令：

```powershell
python -m tensorboard.main --logdir logs/tensorboard --port 6006 --host 127.0.0.1
```

TensorBoard 记录已完成；MLflow 尚未接入。当前没有 WAPE、RMSE、其他基线或最终测试集成绩。

## 七、环境与依赖记录

本次核对 Python 3.11.18、RTX 5060 8 GB、驱动 577.00。已安装 PyTorch 2.10.0+cu128，CUDA 12.8 可用，并实际执行 GPU 上三个 1 的求和，结果为 3.0。

Dataset 和 DataLoader 属于 torch，不额外安装名为 dataset 的包。当前根目录 requirements.txt 由空文件更新为实际直接依赖的固定版本：

| 依赖 | 版本 |
| --- | --- |
| numpy | 2.4.6 |
| polars | 2.1.0 |
| PyYAML | 6.0.3 |
| python-dotenv | 1.2.4 |
| torch | 2.10.0+cu128 |
| tensorboard | 2.21.0 |
| tqdm | 4.70.1 |
| minio | 7.2.20 |
| urllib3 | 2.8.0 |
| pytest | 9.2.0 |

依赖文件包含 PyTorch 官方 CUDA 12.8 的额外索引。没有把整个 Conda 环境的所有包都冻结进项目依赖，也没有提前加入尚未使用的 MLflow、TFT 等组件。

README 同步更新数据范围、窗口划分、实测指标、安装运行命令和未完成事项，替换之前“尚无模型结果”和“正式样本待确定”等旧状态。旧日志保留当日事实，不追改历史记录。

## 八、测试结果与当前实现边界

本次设置 `RUN_MINIO_TESTS=0`，实际运行普通测试集合，结果为 **13 passed、1 skipped**：

| 文件 | 通过数 |
| --- | ---: |
| test_inspect_m5.py | 8 |
| test_inspect_m5_scope.py | 1 |
| test_split_m5.py | 1 |
| test_features.py | 3 |
| test_ingest.py | 默认跳过，本次未重新运行 |

新增的 rolling 用例与重构后的 Subset 边界用例通过。当前没有独立测试覆盖 Naive 前向和 MAE 的全部输入情况，也没有用可训练模型测试 optimizer 分支；真实 Naive 验证结果则通过事件文件与独立销量计算交叉核对。

特征生成入口已经完成适配：从完整实验子集读取，生成 `sales_feature.parquet`，无需再读取旧的 split 文件名字段。当前特征文件覆盖验证、测试日期，并不意味着可以把这些日期上的历史销量派生特征作为未来已知输入；仍需遵守每次预测起点的信息边界。

后续已确认并核对磁盘文件：`scripts/train.py` 的读取路径已改为 `data.directory / features.directory / features.filename`，指向完整的 `sales_feature.parquet`。特征生成与训练入口已衔接；Dataset 当前仍只返回历史销量与销量标签张量，其他特征尚未作为模型输入。

## 九、面试复盘与下一步

**为什么按标签日期划分 Dataset？** 一个预测样本同时包含过去和未来标签，按行切文件容易遗漏历史上下文；按完整标签窗口决定归属，既保留合法历史，又排除跨集合边界的标签。

**为什么验证历史可以与训练数据重叠？** 在新的预测起点，这些销量已经发生。需要隔离的是未来预测标签，而不是禁止使用已经知道的历史观测。

**为什么 Naive 也继承 nn.Module？** 为了复用批量张量接口与评估流程；这不意味着它有可学习参数。参数和指标的实验追踪由日志或 MLflow 完成。

**为什么不是平均每个 epoch 的误差？** 每轮对应不同模型状态，最终结果应明确是最后一轮或最佳验证模型的结果。训练过程曲线逐轮记录，不能用跨轮累计值代替模型指标。

**MAE 0.954333 表明什么？** 当前样本和固定验证期中，单个门店商品每天的销量预测平均绝对误差约 0.954 件；这只是第一份比较基准，尚不能证明模型有业务收益或优于其他方法。

下一轮优先：

1. 增加 WAPE/RMSE，并实现 Seasonal Naive；在同一验证起点、同一标签集合上比较。按原计划继续代表商品的 ARIMA。
2. 用小型可训练模型验证训练分支，再实现 LSTM、checkpoint 与早停；配置和指标接入 MLflow。
3. 确定多变量输入列及价格空值、初始 lag/rolling 空值的处理方式；所有拟合变换只使用实际训练部分。
4. 后续范围按当日新增的项目路线执行，原方案的 raw 自动入库、DuckDB 与额外服务不再作为全部必做项。

当前已获得第一份可信的验证基线，尚未完成完整六周方案。本文、README 和 requirements.txt 由助手整理更新；Git 暂存与提交由自己完成。

## 十、当日方向调整：围绕面试实验闭环验收

明确项目应在算法实验、数据工程、时序建模、MLOps 和结果解释上，与 RAG Agent System 形成互补。新增 `docs/project-roadmap.md`，作为后续执行顺序与验收依据；此处计划调整不代表相应模型已实现。

必做项确定为 Naive、Seasonal Naive、ARIMA 代表性对照、LSTM、TFT，统一 MAE/RMSE/WAPE，最佳 checkpoint 与早停，MLflow 实验追踪，以及品类、销量水平、预测步长三类误差切片。LSTM/TFT 至少一个真正使用多变量特征，不能把读取 24 列文件当作特征已接入模型。

ARIMA 若只跑代表序列，须另列同一代表集合的模型对照；销量分层阈值使用训练期统计；测试集不参与调参；未来真实销量派生的 lag/rolling 不进入未来已知输入。深度模型是否优于基线由实验决定，不预设 TFT 必须胜出。

MinIO 保持现有能力，Optuna 10–20 trials 为推荐扩展，Airflow 最多一个可选 DAG。FastAPI、TorchServe、K8s、Kafka、Redis 移出完成标准；raw 自动上传、DuckDB 等按实际需要再补。没有删除已有代码或端口配置，也未引入尚未使用的新依赖。

执行顺序调整为：统一指标与 Seasonal Naive → LSTM 训练闭环与 MLflow → 多变量输入与 TFT → ARIMA 代表性对照与可选 HPO → 固定配置后的测试评估、误差分析和 README 展示。满足路线中的六项条件即冻结范围，转入面试准备。

继续采用教学方式，核心代码由学习者实现。20–30 个有效小时保留为粗略预算，在 LSTM 首次端到端实验后按实际成本重估，不将其作为完成承诺。
