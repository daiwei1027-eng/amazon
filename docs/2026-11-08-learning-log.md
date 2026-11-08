# 2026 年 11 月 8 日学习日志：日历特征、手写 TFT 与最佳模型恢复

今天完成了 15 特征 LSTM 的日历输入实验，以及自定义 PyTorch TFT 的 Dataset、GRN、变量选择、静态上下文、编码器/解码器、可解释注意力和回归预测头。TFT 已完成真实训练、早停、最佳模型保存和完整验证集恢复复评。

**TFT 先完成原始 year 版本，随后将年份改为 year_offset；最新最佳验证 MAE 为 0.7360584182739258。MAE 略低于原 12 特征 LSTM，但 RMSE 更高，参数量和训练耗时也明显增加。当前结果不能概括为 TFT 全面胜出。** 最终测试集仍未评估，ARIMA 代表性对照和误差切片尚未完成。

本文按 Asia/Shanghai 的 2026-11-08 记录。证据来自使用者提供的训练和 reload 输出、本地源码与 checkpoint 核对、CUDA 验证集复评及普通 pytest 回归。核心实现以学习者分步编写为主；助手按明确委托补充 VSN forward、TFT 缺失定义、测试、reload_tft 和文档。Git 暂存、提交与推送由使用者处理。

## 一、统一验证结果

固定实验子集仍为 CA_1、CA_2 的 100 个共同商品，共 200 条门店商品序列。使用 56 天历史，在 2016-03-23 一次预测未来 30 天；验证标签为 2016-03-24 至 2016-04-22，共 6,000 个值，真实销量合计 5,932 件。

| 模型 | 验证 MAE | 验证 RMSE | WAPE，比例 | WAPE，百分数 |
| --- | ---: | ---: | ---: | ---: |
| Naive | 0.954333 | 1.934166 | 0.965273 | 96.5273% |
| Seasonal Naive，周期 7 天 | 0.970167 | 1.862301 | 0.981288 | 98.1288% |
| 单变量 LSTM，11 月 6 日对照 | 0.754507 | 1.582347 | 0.763156 | 76.3156% |
| 12 特征 LSTM，11 月 6 日对照 | 0.738959 | **1.529758** | 0.747430 | 74.7430% |
| 15 特征 LSTM，本日 | 0.741551 | 1.550951 | 0.750052 | 75.0052% |
| TFT，原始 year，本日 | 0.738534 | 1.556120 | 0.747000 | 74.7000% |
| TFT，year_offset，本日追加 | **0.736058** | 1.577538 | **0.744496** | **74.4496%** |

旧四行沿用已完成的历史验证结果；本日新增的 15 特征 LSTM 及两版 TFT 均由对应 best.pt 复评。以上全部是验证集成绩，不是最终测试成绩或 M5 竞赛排行榜分数。

原始 year 版 TFT 相对 12 特征 LSTM 的 MAE 降低约 **0.0575%**，RMSE 升高约 **1.72%**。追加 year_offset 实验后，MAE 相对原始 year 版降低约 **0.335%**，RMSE 升高约 **1.38%**；相对 12 特征 LSTM 的 MAE 降低约 **0.392%**，RMSE 仍更高。绝对误差和平方误差的表现不同，需要后续逐序列、品类与预测步长切片定位，不能先归因为节假日或价格变化。同一目标集合的 WAPE 分母固定，其排名与 MAE 相同，不作为独立的第二份提升证据。

15 特征 LSTM 本次未超过 12 特征版本。TFT 还使用未来日历和静态 ID；原始 year 版的年份编码与 15 特征 LSTM 不同，追加的 year_offset 版则与其相同，但输入可用范围仍不同。因此模型间是完整方案对照，不是只改变结构的单因素实验。所有结论仍限于单个训练 seed 和一个固定验证起点。

## 二、日历特征与年份数值尺度

原先的特征表已保留 date、wday、year、month，但旧的 12 特征模型只选入 is_weekend。数据表有字段不等于模型已消费该字段；真正的输入由配置列名及顺序决定。

在 features.py 中新增：

```python
(pl.col("date").dt.year() - 2011).cast(pl.Float32).alias("year_offset")
```

原始 date 和 year 均保留，新增的 year_offset 将 2011–2016 映射为 0.0–5.0。2011 是本项目固定年份基准，不是在验证集上拟合的统计量。

当前特征文件为 **388,200 行、25 列**，仍为 200 条序列，日期 2011-01-29 至 2016-05-22。lags 继续为 1、7、28，rolling 为 7、14、28 天，先 shift(1)，std 使用 ddof=1。

本日 LSTM 输入按原 12 列末尾追加 month、year_offset、wday，共 15 列。一次直接使用原始 year 的探索运行出现训练 MAE 约 0.8566、验证 MAE 约 0.9894，预测接近零；当时全零预测的验证 MAE 为约 0.988667。门预激活检查支持原始年份的大数值造成 sigmoid/tanh 饱和的解释。修正年份偏移后，训练损失恢复正常下降。

这不等于完成了全部数值特征的标准化，也不能据此断言所有模型使用原始 year 都会失败。**首个成功 TFT Run 使用原始 year**，它经过逐变量投影和 GRN；随后追加 year_offset 版本，当前 TFT 与 15 特征 LSTM 均使用 year_offset。恢复任一版本时必须匹配训练时的选择，不能只因形状相同就替换列。

## 三、15 特征 LSTM 的真实结果

| 项目 | 值 |
| --- | --- |
| Run | LSTM_20261108_092617_264492 |
| 输入 | 原 12 列，加 month、year_offset、wday |
| hidden_size / num_layers | 64 / 1 |
| 参数量 | 22,686 |
| 最佳 epoch | 4 |
| 最佳 train MAE | 0.6271118062078324 |
| 最佳 valid MAE | 0.741550999323527 |
| 停止 epoch | 9 |
| 停止时 valid MAE，控制台六位 | 0.742960 |
| 首轮训练耗时 | 168 秒 |
| 后续每轮训练耗时 | 132–147 秒 |

配置继续使用 seed=42、warmup=28、AdamW、lr=0.001、weight_decay=0.0001、MAE、patience=5、batch size=32/8。reload.py 恢复该次 best.pt 后得到 MAE=0.741551、RMSE=1.550951、WAPE=0.750052。

排查训练变慢时，确认曾同时运行两份 python 训练进程，争用计算资源。随后停止旧进程并核对只剩新训练。参数增长、数据准备、并行任务都会影响速度，不能将耗时差全部归因于新增三个特征。

## 四、TFT 输入约定和独立 Dataset

新增 etc/tft_dataset.py，继承 ForecastDataset，复用排序、按门店商品分组、sample_index、长度和 warmup。原始 sales 保持完整，仍由 split_dataset_by_time 按完整标签窗口划分 train/valid/test；验证样本可以使用预测起点之前的历史。

当前三类输入为：

| 类型 | 实际字段 | 时间范围 |
| --- | --- | --- |
| 历史观测，11 列 | units_sold、sell_price、lag_1/7/28、rolling mean/std 7/14/28 | 历史 56 天 |
| 未来已知日历，4 列 | is_weekend、month、year_offset、wday；首个成功版本用原始 year | 历史 56 天和未来 30 天 |
| 静态属性，2 列 | store_id、item_id | 每条序列一组 |

cat_id、dept_id 等字段仍在数据表中，但本轮未作为模型输入。未来价格默认未知，未来真实销量及其离线 lag/rolling 不进入预测输入。

单个样本返回 `(inputs, target)`，批处理后为：

```text
inputs["past_observed"]  [B, 56, 11]  float32
inputs["past_known"]     [B, 56, 4]   float32
inputs["future_known"]   [B, 30, 4]   float32
inputs["static"]         [B, 2]       long
target                   [B, 30]      float32
```

日历列在配置中声明一次，Dataset 分别取历史与未来窗口。两段数组不能用 NumPy 的 + 拼接；当前接口保留为两个张量。价格 null 仅在历史输入副本中填 0，原数据不变；不新增 price_missing。

warmup=28 继续保留，首个 target_start 为 lookback+warmup=84。训练窗口 353,600，batch size=32 时为 11,050 批次；验证、测试各 200 个窗口，最终测试尚未评估。

## 五、静态映射、类别数量与字段顺序

本日区分了三种数据：

- static_mappings：字段内的原始 ID 到从零开始连续整数的映射。
- static_cardinalities：每个字段的类别数量，当前为 store_id=2、item_id=100。
- inputs["static"]：当前样本的两个整数 ID，不是整个映射字典。

Dataset 不再额外接收 static_features 参数，字段及顺序由 static_mappings 的键决定；模型的 static_cardinalities 从同一映射按相同顺序计算。测试同步了这项接口变更，并专门检查调换映射字段顺序时只有静态 ID 的排列变化。

本次训练和 reload 均从完整固定子集，对配置中的静态字段 unique 后排序编号。整理时确认全量与训练期的门店、商品 ID 集合相同；编号没有使用未来销量统计。本项目不为构建窗口而提前截断 sales。需要数据统计的缩放、销量分层等仍须仅在训练期拟合。

当前 checkpoint 没有保存映射。reload_tft 按训练时相同数据、字段顺序和排序规则重建；未知新类别处理和映射自动归档仍待完善。

## 六、自定义 PyTorch TFT 组件

TFT 由项目自己的 models 模块组成，不依赖 PyTorch Forecasting 或 Lightning。参考 [TFT 原论文](https://arxiv.org/abs/1912.09363)，采用其主要结构，并将输出设为本项目统一 MAE 训练的单值预测；原论文的多分位数输出与 quantile loss 本轮未实现。

| 文件 | 实现内容 |
| --- | --- |
| models/grn.py | GRN：可选上下文、ELU、门控、投影残差与 LayerNorm |
| models/vsn.py | 逐变量 GRN、打分 GRN、Softmax 和加权求和 |
| models/attention.py | 各头独立 Q/K、共享 V、各头平均及因果掩码 |
| models/tft.py | 输入投影、三路 VSN、静态上下文、双 LSTM、残差、静态融合、注意力及回归头 |

数值特征各自经过 Linear(1,H)，保留变量轴。历史和未来的同一日历变量共享投影参数；静态类别使用 Embedding。

```text
历史变量投影     [B, 56, 15, H] → 历史 VSN → [B, 56, H]
未来日历投影     [B, 30, 4, H]  → 未来 VSN → [B, 30, H]
静态 Embedding   [B, 2, H]      → 静态 VSN → [B, H]
```

静态结果分别通过四路 GRN 生成变量选择上下文、LSTM 初始 h/c、静态融合上下文。编码器处理历史，解码器接收编码器 h/c 并处理未来日历；两个 LSTM 独立创建，input_size 均为 H，不复用原 LSTM 预测器的最后时间步和预测头。

GRN 的上下文投影只处理额外 context；残差在门控后加入，不原地修改 x。VSN 的权重在变量轴 Softmax，返回融合结果及权重。

注意力把历史和未来接成 86 个时间步，使用上三角掩码屏蔽后续位置，保留当前及更早的位置。共享 V 并平均各头输出，区别于普通多头输出拼接。注意力后继续门控残差、逐位置 GRN，再与 post-LSTM 的 temporal 残差相加。

最终只取未来 horizon 个位置，经过 Linear(H,1) 和 squeeze(-1) 返回 `[B,30]`，没有词表 Softmax。当前注意力和变量权重可以作为后续解释分析的数据来源，尚未生成解释图或因果结论。

## 七、训练入口与字典输入联调

train.py 根据 model_key 为 TFT 构建 TFTDataset，为其他模型保留原 Dataset 路径。TFT 和 LSTM 均创建 AdamW；optimizer=None 的路径只用于朴素基线等无需训练的模型。

训练和验证循环都需要逐个移动字典中的 Tensor：

```python
if isinstance(past_target, dict):
    past_target = {
        name: tensor.to(device)
        for name, tensor in past_target.items()
    }
else:
    past_target = past_target.to(device)
```

分支之后不能再无条件对字典调用 .to(device)。本日曾在训练循环留下旧调用，以及漏改 evaluate，先后触发 AttributeError。

首次正式 TFT 尝试 `TFT_20261108_113155_691240` 完成约 9 分钟训练后，在第一轮验证报错，进程退出码为 1。由于 fit 先验证再保存最佳模型，该次没有 best.pt。对应 [MLflow Run](http://localhost:5000/#/experiments/2/runs/bb144c7715bd4f8ab7eb1f762186d777) 不纳入最终成绩表。

修正 evaluate 后，使用真实验证集的一个 batch、初始化 TFT 和 cuda:0 检查字典搬运、输出 `[8,30]`，MAE/RMSE/WAPE 均与直接计算一致。这是输入与评估链路检查，不是模型成绩。随后重新开始正式训练。

模型 pytest 已通过并不意味着训练入口已完成接入；本次报错位于训练/验证循环的数据转换阶段。后续可将实际 train_one_epoch/evaluate 的字典批次检查固化为专门的回归测试。

## 八、原始 year 版 TFT 运行与早停

| 项目 | 值 |
| --- | --- |
| Run | TFT_20261108_114514_027332 |
| MLflow Run ID | f58e7286895e4ddca6476f1bd569f19e |
| hidden_size / num_layers / num_heads | 64 / 1 / 1 |
| Dropout | 0.0 |
| 参数量 | 685,673 |
| 训练 / 验证 batch size | 32 / 8 |
| 最佳 epoch | 4 |
| 最佳 train MAE | 0.6286388303129562 |
| 最佳 valid MAE | 0.7385341339111328 |
| 停止 epoch | 9 |
| 最后 valid MAE，控制台六位 | 0.747036 |

[查看成功的 MLflow Run](http://localhost:5000/#/experiments/2/runs/f58e7286895e4ddca6476f1bd569f19e)。训练 criterion 为 MAE，seed=42，AdamW 的 lr=0.001、weight_decay=0.0001，epochs 上限 100，patience=5。

| Epoch | train MAE | valid MAE | 训练进度条耗时 |
| ---: | ---: | ---: | --- |
| 1 | 0.638706 | 0.744797 | 08:47 |
| 2 | 0.631892 | 0.746507 | 08:48 |
| 3 | 0.630937 | 0.750937 | 08:50 |
| 4 | 0.628639 | **0.738534** | 08:52 |
| 5 | 0.626444 | 0.739526 | 08:58 |
| 6 | 0.626264 | 0.761864 | 08:54 |
| 7 | 0.625920 | 0.749182 | 08:50 |
| 8 | 0.626418 | 0.738551 | 08:50 |
| 9 | 0.626120 | 0.747036 | 08:54 |

第 8 轮与第 4 轮很接近，但没有严格改善，等待计数不会清零；第 9 轮后累计五轮未改善，正确早停。fit 返回停止轮次结果，最后的 Valid Loss=0.747036 不替代最佳验证值。

每轮训练约 8 分 47 秒至 8 分 58 秒，9 轮进度条耗时合计 **79 分 43 秒**，不含数据装配、验证、模型保存和上传等额外时间。约为当前多变量 LSTM 单轮耗时的 4 倍，不是精确性能基准。

TFT 参数量约为 15 特征 LSTM 的 **30.2 倍**。历史 VSN 为 333,020，未来 VSN 为 88,952，静态 VSN 为 42,196；三个 VSN 合计 464,168，约占总参数的 67.7%。预测头虽只有 65 个参数，但前面的 GRN、双 LSTM、Embedding 和注意力贡献了主要容量。参数倍数不等于耗时倍数。

## 九、原始 year 版 reload_tft 与最佳模型复现

新增 [scripts/reload_tft.py](../scripts/reload_tft.py)，首次复评时 RUN_NAME 为 TFT_20261108_114514_027332，日历包含原始 year。随后入口已切换至下节的 year_offset 版。脚本读取当前 YAML 与本地特征表，按训练规则重建静态映射，创建 TFTDataset，使用同一时间划分和 evaluate，仅复评验证集。

```powershell
python -m scripts.reload_tft
```

助手执行及使用者独立执行均得到：

```text
Best epoch: 4
Saved MAE: 0.7385341339111328
MAE: 0.738534
RMSE: 1.556120
WAPE: 0.747000
```

实际恢复的 MAE 与保存值完全一致。完整精度的 RMSE 为 1.5561197167622811，WAPE 为 0.7470001354461896。验证共 200 个窗口，每项指标 25 个 batch。

后续更换 Run 时修改 RUN_NAME，并匹配当次数据、特征顺序、静态映射与结构参数。目前 checkpoint 只含 epoch、模型/优化器状态、train_loss、valid_loss、metric_name，没有完整特征 schema、映射或配置快照，不能只凭权重文件自动恢复所有输入语义。

原始 year 版停止在第 9 轮，未达到第 10 轮的周期保存条件，因此该成功 Run 的本地 checkpoints 目录只有 best.pt，没有 checkpoint_0001.pt。

## 十、追加 year_offset 版 TFT 实验

将已知日历的第三列由 year 改为 year_offset，其余列顺序保留为 is_weekend、month、year_offset、wday。原始 year 仍在特征表中，当前配置只更换模型选取的列。按照本次运行记录与当前配置，hidden_size=64、num_layers=1、num_heads=1、Dropout=0，seed=42、warmup=28、MAE、AdamW 与 batch size 均沿用前次设置。

| 项目 | 值 |
| --- | --- |
| Run | TFT_20261108_133756_506473 |
| MLflow Run ID | 20a569b16d75422095325d99ac2900e5 |
| 参数量 | 685,673，与原始 year 版相同 |
| 最佳 epoch / 停止 epoch | 4 / 9 |
| 最佳 checkpoint 的 train MAE | 0.6278446788845379 |
| 保存的 valid MAE | 0.7360584182739258 |
| 恢复 MAE / RMSE / WAPE | 0.736058 / 1.577538 / 0.744496 |
| 最后 valid MAE，控制台六位 | 0.743759 |

[查看 year_offset 版 MLflow Run](http://localhost:5000/#/experiments/2/runs/20a569b16d75422095325d99ac2900e5)。

| Epoch | train MAE | valid MAE | 训练进度条耗时 |
| ---: | ---: | ---: | --- |
| 1 | 0.638862 | 0.752401 | 08:56 |
| 2 | 0.629117 | 0.739323 | 08:57 |
| 3 | 0.627277 | 0.745178 | 08:49 |
| 4 | 0.627845 | **0.736058** | 08:49 |
| 5 | 0.624515 | 0.745000 | 08:51 |
| 6 | 0.622415 | 0.736321 | 08:54 |
| 7 | 0.620600 | 0.738154 | 08:53 |
| 8 | 0.619380 | 0.743188 | 08:51 |
| 9 | 0.619435 | 0.743759 | 08:52 |

第 6 轮没有低于第 4 轮，最后五轮均未严格改善，因此第 9 轮早停。每轮训练约 8 分 49 秒至 8 分 57 秒，9 轮进度条耗时合计 **79 分 52 秒**，不含装配、验证、保存与上传。改用 offset 未改变模型容量，本次耗时与原始 year 版接近。

使用者将 reload_tft 的 RUN_NAME 切换到该 Run，保持已知日历为 year_offset 后复评，输出：

```text
Best epoch: 4
Saved MAE: 0.7360584182739258
MAE: 0.736058
RMSE: 1.577538
WAPE: 0.744496
```

本次整理独立核对了 checkpoint 的 epoch、保存损失、状态规模和文件哈希；上述恢复指标来自使用者实际运行输出，MAE 与保存值在六位小数展示精度上一致。本次文档追加没有重新训练或重新执行三指标复评，也没有把此前普通 pytest 结果记作新一轮测试。

相对原始 year 版，MAE 下降而 RMSE 上升，表明两类损失的权衡发生变化。RMSE 对较大误差更敏感，但尚无逐目标误差数据定位原因，不能直接断言是某类商品或某个极端点导致。当前仍按 MAE 选择 checkpoint，保留两版结果供后续切片对照。两次均在第 9 轮停止，本地各只有 best.pt；重载原始 year 版时需同时切回原始 year 列，不能仅更改 RUN_NAME。

## 十一、测试、依赖与可复现边界

| 新增测试文件 | 用例数 | 重点 |
| --- | ---: | --- |
| tests/test_tft_dataset.py | 10 | 取值、列/ID 顺序、warmup、序列隔离、未来输入隔离、DataLoader 与时间划分 |
| tests/test_grn.py | 11 | 输出维度、残差投影、上下文广播、不改输入、Dropout、有限梯度 |
| tests/test_vsn.py | 8 | 静态/时序形状、归一化权重、加权结果、上下文、Dropout、梯度 |
| tests/test_tft.py | 15 | `[B,horizon]`、全参数梯度、单样本单步、因果性、批次独立、AdamW 更新、checkpoint 往返 |

本次普通回归实测：**71 passed、2 skipped，4.62 秒**。MinIO、MLflow 真实服务测试开关在该次回归中显式关闭；没有把本日训练 Run 的成功等同于重新执行了所有服务集成测试。单层/双层、单头/四头、非零 Dropout 仅有代码测试覆盖，本日实际 TFT 训练仍是单层、单头、Dropout=0。

```powershell
python -m pytest tests -q
python -m pytest tests/test_tft_dataset.py tests/test_grn.py tests/test_vsn.py tests/test_tft.py -q
```

TFT 使用现有 PyTorch 组件，本日没有新增第三方 Python 依赖；requirements 继续沿用已固定的版本。没有新建干净环境重装，也未重新运行 MinIO 往返测试。最近一次独立 MinIO 通过证据仍见 11 月 1 日日志。

固定 seed、数据与配置后，best.pt 恢复复评已经验证；跨环境逐位确定性、完整随机状态恢复及断点续训仍未完成。TFT 配置中的 dropout、num_heads 目前在 train.py 构造调用中尚未显式透传，当前模型默认 0.0/1 与 YAML 相同；后续改参数前需要同步入口。TFT 专属结构参数、静态映射、完整配置和数据哈希也尚未全部自动归档到 MLflow。

## 十二、本地校验标识

以下为整理时的本地快照，记录在文档不等于训练已自动上传这些标识。

```text
data/processed/m5_exp_sub.parquet
ca9703d919af77ec2161513014ce0cb0397e1db4e2a96670e7b59807bf970ba2

data/feature/sales_feature.parquet（25 列，包含 year_offset）
538793eae25a397f1203b32be8cff4cc28b5d56441caad42c960a7b658968206

data/LSTM_20261108_092617_264492/checkpoints/best.pt
273db4368432e0af2ccaa74c40acc7f844a82615a5762b2a38be32b9a82cfe04

data/TFT_20261108_114514_027332/checkpoints/best.pt
7ddd6bfb85d5c80983deb76377e3106237abe32247e44c0d56f2584214dc7760

data/TFT_20261108_133756_506473/checkpoints/best.pt
b7108dd86edbcf8c4967c95b1453ee7f8628da3469ae3b55b974cd043a4a7862
```

11 月 6 日的 24 列特征快照哈希保留在历史日志中。今天新增 year_offset 后文件发生变化，不能继续将旧哈希当作当前文件版本；新特征对象是否同步到 MinIO 本次未核实。

## 十三、下一步与项目状态

已完成 Naive、Seasonal Naive、单变量/多变量 LSTM、自定义 TFT 的验证结果与最佳模型恢复。项目不要求 TFT 必须胜出，本轮结果应同时报告精度、参数量和训练成本。

后续按既定范围推进：

1. 补 ARIMA 的代表性序列对照。序列选择规则从训练数据确定，并在同一代表集合补算其他模型，单列覆盖范围和拟合失败。
2. 补齐实验元数据：配置、输入 schema、静态映射、实际模型参数、数据校验标识和三指标复评关联；明确重建条件，减少手工切换。
3. 按验证结果固定方案后，才评估最终测试集；测试结果不用于反复挑模型。
4. 保存逐序列、逐预测日结果，完成品类、训练期销量分层、Day 1–7 / 8–14 / 15–30 的误差切片。
5. 完成展示材料与复现核查。Optuna、Airflow 仍为可选扩展，不增加额外服务作为冻结前置条件。

当前仍存在单 seed、单验证起点、子集覆盖有限、未统一数值标准化、元数据归档不完整，以及 RMSE 零误差反传和 WAPE 全零训练批次策略未处理等限制。当前真实训练统一使用 MAE，尚无最终测试成绩、跨季节稳定性、统计显著性或业务收益结论。

本文与 README、项目路线进度按使用者委托更新，原有历史日志保留当时事实。
