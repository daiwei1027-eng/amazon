# 2026 年 11 月 5 日学习日志：Seasonal Naive、统一指标与 LSTM 训练闭环

今天完成了三种可选损失函数、训练与验证的整轮指标汇总、早停及 checkpoint 保存，实现每周周期的 Seasonal Naive 和单变量 LSTM，并完成首次真实 LSTM 训练。随后通过独立加载脚本恢复最佳模型，重新验证得到与保存值完全一致的 MAE。

**LSTM 最佳验证 MAE 为 0.7513453375498453，最佳轮次为第 17 轮，第 22 轮触发早停。相较 Naive 的验证 MAE，降低约 21.27%。** 这是一组固定实验子集、固定验证起点上的单次结果；尚未评估最终测试集，也没有验证跨起点稳定性。

本文依据当天代码审阅、人工数据检查、使用者提供的真实运行输出，以及最终磁盘文件核对整理。真实项目目录为 `C:/Users/admin/PycharmProjects/amazon`。核心代码由学习者实现；日志由助手整理，Git 暂存与提交由学习者自行完成。

## 一、数据与评估协议保持一致

沿用已固定的 M5 实验子集：CA_1、CA_2 两家门店的 100 个共同商品，共 200 条门店商品序列。此前核实的实验特征文件为 `data/feature/sales_feature.parquet`，388,200 行、24 列；今天没有重新运行全量预处理或重新抽样。

模型均使用过去 56 天销量，一次预测未来 30 天：

| 项目 | 当前约定 |
| --- | --- |
| 输入 | 历史销量，float32，形状 `[batch, 56]` |
| 标签与预测 | `[batch, 30]` |
| 训练标签截止 | 2016-03-23 |
| 验证标签 | 2016-03-24 至 2016-04-22 |
| 测试标签 | 2016-04-23 至 2016-05-22，今天没有评估 |
| 训练窗口数 | 359,200；batch size 32，共 11,225 批 |
| 验证窗口数 | 200；batch size 8，共 25 批、6,000 个目标值 |

Dataset 当前只返回历史销量与未来销量标签。读取 24 列特征文件，不代表日历、价格、lag、rolling 或静态属性已进入 LSTM。多步输入不能引用标签区间中的真实销量来构造未来特征。

## 二、三种损失与整轮指标汇总

在 `etc/losses.py` 提供三个函数，均先检查 prediction 与 target 的形状完全一致，并返回标量 Tensor，以便参与反向传播：

| 函数 | 计算方法 |
| --- | --- |
| `mae_loss` | `mean(abs(prediction - target))` |
| `rmse_loss` | `sqrt(mean((prediction - target) ** 2))` |
| `wape_loss` | `sum(abs(prediction - target)) / sum(abs(target))` |

WAPE 分母为零时抛出异常。当前值为比例，若以百分数展示需要明确乘以 100。

配置 `train.criterion` 选择一个 criterion，训练与验证使用同一种指标；当前正式实验选择 MAE。没有另设一套训练和验证目标。

### 修正跨批次聚合

原先按 `batch_loss * target.numel()` 累加再除以总元素数，只对 MAE 适用。RMSE 需要累计平方误差后统一开方，WAPE 需要累计分子与分母后统一相除。

最终采用根据传入 criterion 分支累计的方法，每次只统计选中指标所需的量：

| criterion | `total_error` 累计 | `total_weight` 累计 | 整轮结果 |
| --- | --- | --- | --- |
| MAE | 绝对误差之和 | 目标元素数 | `total_error / total_weight` |
| RMSE | 平方误差之和 | 目标元素数 | `sqrt(total_error / total_weight)` |
| WAPE | 绝对误差之和 | 真实值绝对值之和 | `total_error / total_weight` |

`train_one_epoch` 和 `evaluate` 已同时完成适配。统计所用误差为 `prediction.detach() - target.detach()`，避免保留计算图。

训练依然使用批次损失 `loss = criterion(prediction, target)`，接着执行 `loss.backward()`、`optimizer.step()`。用于整轮报告的 detached error 不参与反向传播。

验证阶段直接累计原始误差，不再逐批调用 criterion，因此某个验证批次真实值全零不会提前触发 WAPE 异常；最终检查整个验证集的分母。训练若选择 WAPE，真实值全零的单个批次仍会使当前损失函数报错，这个限制尚未另行处理。

### 本轮审阅修正

- 删除 `total_error == 0` 时抛异常的判断：误差为零表示预测完全正确，应返回 0。
- 在 `evaluate` 中补回预测与标签的 shape 检查：移除逐批 criterion 调用后，需要显式防止 PyTorch 广播导致静默计算错误。
- 当前 RMSE 损失直接对 MSE 开方，误差恰好为零时的梯度数值问题仍待处理；本次 MAE 训练不受影响。

人工数据示例 `prediction=[1,3,2]`、`target=[1,1,4]` 的预期为 MAE=4/3、RMSE=sqrt(8/3)、WAPE=4/6。固定预测时，改变 batch size 不应改变整轮指标，仅允许浮点累计误差。

## 三、fit、早停与 checkpoint 行为

`scripts/train.py` 的统一执行接口继续保留：

- `optimizer=None`：仅验证一次，TensorBoard step=0，返回 `(None, valid_loss)`；不训练、不保存最佳或周期模型。
- 有优化器：逐轮训练与验证，仅依据验证指标严格下降更新最佳轮次，未改善时累计 `bad_epochs`。
- 发生改善时重置 `bad_epochs=0`；达到 patience 后，在完成本轮记录、保存与打印后停止。
- `fit` 返回停止时最后一轮的 `(train_loss, valid_loss)`，没有改为最佳轮次结果。
- `best_train_loss` 仅保存最佳验证轮次对应的训练指标，不参与模型选择。

`checkpoint_path` 表示目录。最佳模型保存为 `best.pt`，改善时覆盖；周期保存与最佳保存独立，每隔 `checkpoint_interval` 轮生成一个编号文件。

当前 checkpoint 包含 epoch、model_state_dict、optimizer_state_dict、train_loss、valid_loss、metric_name。周期编号为 `(epoch + 1) // checkpoint_interval`，文件名使用 `checkpoint_0001.pt` 等形式。

接口检查包括：epochs 为正数；patience 非空时为正数；checkpoint interval 非空时为正数，并要求提供保存目录。metric_name 根据 criterion 名称得到，TensorBoard tag 与训练打印随指标切换。

训练结束后模型仍是最后一轮状态，不会自动恢复最佳参数。使用最佳模型需要显式加载 `best.pt`。虽然保存了优化器和 epoch，但完整断点续训入口、历史最佳值、早停计数与随机状态恢复尚未实现。

## 四、实现 Seasonal Naive 并获得结果

在 `models/SeasonNaive.py` 实现无可学习参数的 `SeasonalNaiveBaseline`，接收 horizon 和 season_length，当前周期为 7 天。

核心计算：

```python
last_season = past_target[..., -self.season_length:]
repeat_count = (self.horizon + self.season_length - 1) // self.season_length
prediction = last_season.repeat(1, repeat_count)
return prediction[:, :self.horizon]
```

30 天预测需要把最后一周重复 5 次，再截取前 30 天。若最后一周是 `[1,2,3,4,5,6,7]`，预测即该组数重复四遍，再接 `[1,2]`。它不是取 30 天前的数据，也不读取未来第一周的真实销量。

初始化检查 horizon 与 season_length 为正；forward 检查历史长度至少覆盖一个周期。输出保持输入的 dtype 与 device，不混合不同样本。

在 `tests/test_seasonnaive.py` 添加两条序列的人工数据检查：用超过 7 天的历史，前面加入 99、98，确认模型取的是最后一周；检查输出形状、周期重复以及最后两列的截取。

本次学习中修正了两个错误：

- 重复次数应向上取整，`30 % 7` 只是余数 2，不能作为重复次数。
- `torch.equal` 的两个参数必须都是 Tensor，不能直接把 Python 列表当作第二个参数。

另外，初始化参数的合法性检查应放在 `__init__`；在 forward 使用实例参数时需要写成 `self.horizon`、`self.season_length`，不能直接引用另一个方法的局部变量。

实际完成 25/25 个验证批次，Seasonal Naive 验证 MAE 为 **0.970167**，比 Naive 稍差。这个结果不能单独证明数据没有周周期。

## 五、理解并实现单变量 LSTM

### 从 RNN 到门控记忆

普通 RNN 使用当前输入与上一隐藏状态生成新的隐藏状态。LSTM 额外维护 cell state，并通过遗忘门、输入门和输出门控制信息传递。

设当前输入为 x_t，上一隐藏状态为 h_(t-1)，上一记忆为 c_(t-1)。标准 LSTM 的核心更新为：

```text
候选记忆 = tanh(当前输入与上一隐藏状态的线性变换)
输入门、遗忘门、输出门 = sigmoid(各自的线性变换)

c_t = 遗忘门 × c_(t-1) + 输入门 × 候选记忆
h_t = 输出门 × tanh(c_t)
```

乘法均为逐元素运算。门值在 0～1 之间，对状态维度逐一加权；历史已经编码在状态中，并非直接为每个历史日期分配一个独立权重。

候选记忆提供本次写入的内容，输入门控制写入多少，遗忘门控制旧记忆保留多少。cell 更新包含加法；输出前使用 tanh，随后乘输出门。

图中的 z、z_i、z_f、z_o 是根据当前输入和上一隐藏状态，经可学习线性变换计算出的中间值。它们随时间步变化；权重与偏置由训练学习，并在各时间步共享。

使用 `nn.LSTM` 时，无需手工实现门函数或维护 `self.cell=0`。未显式传入初始状态时，模块按零状态开始处理当前批次中的序列。

### 模型结构与形状

在 `models/lstm.py` 实现 `LSTM`，由一层 `nn.LSTM` 和一个线性预测头组成：

```text
历史销量                       [batch, 56]
unsqueeze(-1)                  [batch, 56, 1]
LSTM(input_size=1, hidden=64)   [batch, 56, 64]
output[..., -1, :]             [batch, 64]
Linear(64, 30)                 [batch, 30]
```

`hidden_size=64` 表示隐藏状态与记忆状态的维度，不表示记住 64 天。`num_layers=1` 表示堆叠一层 LSTM；层数和时间步数不同。LSTM 的 output 是最上层在各时间步的隐藏输出。

`self.lstm(x)` 已按顺序计算全部 56 个时间步，随后 `output[..., -1, :]` 只是选出最后一步的结果。最后状态通过递推接收此前的信息，但不保证完整保留所有历史。

线性层仅改变最后一个维度。先取最后时间步，才能得到 `[batch, 30]`；如果将完整 `[batch, 56, 64]` 直接传入 Linear，会得到 `[batch, 56, 30]`。

当前模型直接一次输出未来 30 天，没有逐日递归预测，也不把未来真实销量作为输入。线性预测头未加入非负约束。

`tests/test_lstm.py` 保存了人工数据检查，覆盖输出 `[2,30]`、MAE 反向传播，以及 LSTM 输入权重梯度存在且有限。

## 六、统一模型选择与 AdamW

训练入口使用字典保存模型实例，通过配置选择当前模型：

```python
models = {
    "naive": NaiveBaseline(horizon),
    "seasonal_naive": SeasonalNaiveBaseline(horizon, season_length),
    "lstm": LSTM(horizon, lstm_hidden_size, lstm_num_layers),
}
model = models[config["train"]["model"]].to(device)
```

运行目录名根据实际选中模型的类名生成，修正之前用 Naive 类名命名 Seasonal Naive 日志的问题。

仅在选择 LSTM 时创建 AdamW，随后将实际 optimizer 传入 fit；两个基线保持 optimizer=None。优化器在模型移到目标设备后创建。

本次训练设置经最终代码与 checkpoint 元数据核对：

| 配置 | 值 |
| --- | --- |
| 模型 | 单变量 LSTM |
| 历史长度 / 预测长度 | 56 / 30 |
| hidden_size / num_layers | 64 / 1 |
| 优化器 | AdamW |
| 学习率 | 0.001 |
| weight_decay | 0.0001 |
| criterion | MAE |
| epochs 上限 | 100 |
| patience | 5 |
| checkpoint_interval | 10 |
| train / valid batch size | 32 / 8 |

初版代码误写为 `torch.optim.Adam`，最终已改为 `torch.optim.AdamW`。

当前 PyYAML 将配置中的 `1e-3`、`1e-4` 解析为字符串，导致优化器初始化发生 TypeError。最终配置改用 `0.001`、`0.0001`，确认读取为数值。

## 七、首轮真实 LSTM 训练结果

run 标识为 `LSTM_20261105_111313_139950`。真实训练控制台输出如下，数值保留日志中的六位小数：

| Epoch | train MAE | valid MAE |
| --- | ---: | ---: |
| 1 | 0.649755 | 0.779737 |
| 2 | 0.641775 | 0.766490 |
| 3 | 0.638155 | 0.766594 |
| 4 | 0.634515 | 0.757728 |
| 5 | 0.632840 | 0.757334 |
| 6 | 0.630409 | 0.770822 |
| 7 | 0.628757 | 0.761016 |
| 8 | 0.627559 | 0.769878 |
| 9 | 0.626106 | 0.755073 |
| 10 | 0.625017 | 0.756606 |
| 11 | 0.623895 | 0.761101 |
| 12 | 0.623268 | 0.754981 |
| 13 | 0.622203 | 0.754396 |
| 14 | 0.622001 | 0.754993 |
| 15 | 0.621371 | 0.757529 |
| 16 | 0.623050 | 0.762449 |
| **17** | **0.620695** | **0.751345** |
| 18 | 0.619566 | 0.761659 |
| 19 | 0.619026 | 0.755303 |
| 20 | 0.618640 | 0.752705 |
| 21 | 0.618193 | 0.761283 |
| 22 | 0.617721 | 0.760531 |

第 17 轮达到最佳验证结果，第 18～22 轮连续五轮没有改善，触发 patience=5 的早停。控制台最后显示 `Valid Loss0.760531`，这是 fit 返回的最后一轮结果，符合当前接口约定。

每轮训练 11,225 批，控制台训练进度条耗时大致 26～29 秒；22 轮训练循环约 10 分钟。该时间不包含完整启动与数据装配的精确计时，也不代表 TFT 或多变量模型的耗时。

### 当前同一验证集上的比较

| 模型 | 验证 MAE | 说明 |
| --- | ---: | --- |
| Naive | 0.954333 | 沿用此前实际验证结果 |
| Seasonal Naive，7 天周期 | 0.970167 | 当天实际验证 |
| LSTM | **0.751345** | 当天训练，第 17 轮最佳 checkpoint |

相较 Naive，最佳 LSTM 的验证 MAE 相对降低约 21.27%。这是预测误差的相对下降，不是准确率，也不是已实现的业务收益。当前只有单次实验、一个验证起点，不能据此声称跨季节稳定或具有统计显著性。

## 八、核对最佳模型、周期保存与加载复现

实际保存目录：

```text
data/LSTM_20261105_111313_139950/checkpoints/
```

整理日志时，助手读取磁盘上三个 checkpoint 的元数据，核对结果：

| 文件 | 保存轮次 | train MAE | valid MAE |
| --- | ---: | ---: | ---: |
| best.pt | 17 | 0.6206949167343627 | 0.7513453375498453 |
| checkpoint_0001.pt | 10 | 0.6250168763934548 | 0.7566058317820231 |
| checkpoint_0002.pt | 20 | 0.6186395691883679 | 0.7527048047383627 |

三个文件中的 metric_name 均为 MAE，优化器参数组均记录学习率 0.001、weight_decay 0.0001。实际文件证明最佳保存与周期保存已在本次真实训练中触发；没有执行从周期 checkpoint 断点续训的验证。

### 独立恢复脚本

在 `scripts/reload.py` 中读取 best.pt，使用 `map_location=device`、`weights_only=True`，重新创建与训练时相同结构的 LSTM，调用 `load_state_dict`，然后重新装配相同验证集进行 evaluate。

最初脚本缺少模型实例化，已补充。使用者实际运行后提供的输出为：

```text
Best epoch: 17
Saved MAE: 0.7513453375498453
Restored MAE: 0.7513453375498453

进程已结束，退出代码为 0
```

验证了同一 checkpoint 在相同数据、划分、模型结构与 MAE 下可以重新加载，并完全复现保存的验证指标。

该结论不等于从头重新训练也能得到完全相同的模型。本次训练入口尚未统一设置 Python、NumPy、PyTorch 与 DataLoader 等随机源；已有 sampling seed=42 只控制之前的数据抽样，不能替代训练随机种子。

最终磁盘核对时，reload.py 仍有两处后续可清理事项：criterion 取自当前配置而非 checkpoint 的 metric_name；`load_state_dict` 重复调用两次。当前配置仍为 MAE，重复加载相同参数不影响本次已复现的结果。后续将恢复流程复用到不同实验前，应统一指标来源与模型配置，并参数化当前写死的 checkpoint 路径。

## 九、验证记录与当前边界

当天验证分为不同层次，不能混写为一次完整回归测试：

| 检查 | 结论与范围 |
| --- | --- |
| 人工数据定向检查 | 审阅过程中通过 19 项检查，涉及固定预测下三个指标在不同分批方式下的训练/验证汇总、验证零误差、WAPE 全零批次、空集、MAE 参数更新与 Seasonal Naive 人工用例 |
| 验证 shape 检查 | 定向检查复现了形状不匹配被广播的问题，随后代码已显式添加 shape 检查 |
| LSTM 小型训练 | 助手使用数值类型正确的配置与 AdamW，在人工数据上跑两轮 CPU 训练，确认输出形状、有限梯度、参数更新与 fit 返回；两个基线的仅评估分支也通过 |
| 真实 LSTM 训练 | 使用者运行 22 轮，训练与验证完成，实际触发早停并保存 best 和两个周期 checkpoint |
| 最佳 checkpoint 恢复 | 使用者独立运行 reload.py，恢复后的 MAE 与保存值完全相同 |
| 完整普通测试集合 | 今天没有重新运行全部 tests；此前的 13 passed、1 skipped 属于 11 月 3 日记录，不能写成今天的回归结果 |

早先的小型保存检查曾因临时目录权限失败，后续重跑未获授权，当时没有完成保存验证。今天的真实训练与实际 checkpoint 核对提供了新的保存证据；没有重新执行被拒绝的检查。

目前仍未完成：

- 在真实模型上统一补齐 MAE、RMSE、WAPE 的正式结果表，保存逐序列、逐预测日结果。
- RMSE 零误差处的反传稳定性，以及 WAPE 全零训练批次的策略。
- 统一训练随机种子、数据与配置版本记录；完整断点续训。
- MLflow 实验追踪。TensorBoard 已接入，不能将其等同于 MLflow。
- 真正消费多变量特征的模型与特征组对照、TFT、ARIMA 代表性对照。
- 固定方案后的最终测试、三类误差切片与展示型 README。

## 十、面试复盘与下一步

**为什么要区分 loss 和整轮指标？** loss 是当前批次参与反向传播的标量；整轮指标需要按全部目标汇总。RMSE 与 WAPE 都不能通过简单平均各批次指标得到。

**为什么 LSTM 只取最后一步还能使用历史？** LSTM 已依次处理所有历史输入，每一步都接收上一时刻的状态；最后隐藏状态是递推后的表示，取最后一步只是在已有输出上索引。

**为什么最佳 MAE 与最后打印值不同？** 模型选择依据验证集，最佳值来自第 17 轮；fit 明确返回停止时第 22 轮的结果。后续推断需显式恢复最佳 checkpoint。

**本次证明了什么？** 在固定的 M5 实验子集与验证窗口中，单变量 LSTM 的 MAE 优于两个简单基线；训练、早停、最佳及周期保存已真实运行，最佳模型可以重新加载复现验证结果。

**还不能声称什么？** 不能声称多变量已经接入、最终测试表现已知、重训完全确定、业务收益已实现，或 TFT 与 ARIMA 已完成。

下一项任务确定为接入 MLflow：

1. 在统一训练入口记录模型、超参数、随机种子、代码与数据版本、采样和日期边界、实际输入特征、设备及耗时。
2. 在每轮记录训练与验证指标、最佳 epoch 和最佳验证值，同时保留现有 TensorBoard 与基线仅评估分支。
3. 记录配置、最佳 checkpoint 等产物；后续扩展逐目标预测和误差切片，并验证可以通过 run 找到模型与数据版本。
4. 完成实验追踪后进入多变量输入与 TFT，再补 ARIMA 代表性对照，最后固定方案、评估测试集并整理误差分析与 README。

范围继续遵循 `docs/project-roadmap.md`。Optuna 与 Airflow 为可选扩展，不增加额外服务作为冻结前置条件。本轮仅整理当天日志，没有执行 Git 暂存、提交或推送。
