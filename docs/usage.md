# LBA 用法

LBA v2 直接从 dataset 构造动态 batch loader：

```python
from lba import LBA


def sample_length(sample):
    return len(sample["input_ids"])


loader = LBA(
    dataset,
    len_fn=sample_length,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    max_padded_length=8192,
)

for batch in loader:
    train_step(batch)
```

`LBA` 是唯一公开 loader 类，也是 `torch.utils.data.DataLoader` 子类。除必填的
keyword-only `len_fn` 和 LBA planner 选项外，它接受标准 DataLoader 配置；
`batch_size`、`shuffle`、`sampler` 等参数仍遵守 PyTorch 原有的互斥规则。

LBA 的输出 batch 数取决于运行时长度规划，不能提前准确计算。因此即使它是
DataLoader 子类，`len(loader)` 仍明确不可用。训练入口应使用显式 step 或 epoch
预算，不能用 loader 长度控制训练。

## 长度与 Collate

`len_fn` 在最终 `collate_fn` 之前接收 raw dataset sample，必须返回正整数。使用
`spawn` multiprocessing context 时，`len_fn` 会进入 worker，必须可 pickle；应使用
模块顶层函数或 callable class，不能使用 lambda 或局部函数。

预算约束的是：

```text
max_length_in_batch * dynamic_batch_size
```

它不会检查最终 `collate_fn` 的 tensor shape。普通 planned batch 不超过预算；单个
sample 自身超过预算时，LBA 会把它作为 singleton 输出并发 warning。

map-style dataset 的 worker、batched `__getitems__`、sampler 和
`persistent_workers` 配置会传给内部 source loader。最终动态 batch 仍由用户提供的
`collate_fn` 构造。配置 `pin_memory=True` 时，LBA 在最终 collate 后 pin batch，不
会把内部 length records 送入 pin-memory queue。

## 长度预算

模型有明确 token 或 padded-length 上限时，推荐显式设置：

```python
loader = LBA(
    dataset,
    len_fn=sample_length,
    batch_size=32,
    collate_fn=collate_fn,
    max_padded_length=8192,
)
```

省略 `max_padded_length` 时，LBA 会读取 warmup source batches，并按 source batch
size 和平均 sample length 推断预算。`warmup_batches` 默认根据 batch size 选择；读取
的 warmup samples 会继续进入 planner，不会因预算推断丢失。空输入无法推断预算，
此时必须显式配置正预算。

## 自定义 Batch Cost

当计算量不是线性的 `max_length * batch_size` 时，可以替换 budget model：

```python
def attention_cost(max_length: int, batch_size: int) -> int:
    return max_length * max_length * batch_size


loader = LBA(
    dataset,
    len_fn=sample_length,
    cost_fn=attention_cost,
    max_batch_cost=2_000_000,
    cost_window_batches=8,
    batch_size=32,
    collate_fn=collate_fn,
)
```

`cost_fn` 必须返回正整数，不能执行 I/O，并且必须随 `max_length` 和
`batch_size` 单调不减。planner 会在 hot path 多次调用它，并依赖单调性二分
最大可行 batch size。custom cost 模式要求显式 `max_batch_cost`，不执行 warmup
预算推断，也不能同时配置 `max_padded_length`。

`len_fn` 仍是排序和 padding-quality 轴。`max_padding_ratio` 仍按长度计算，
不表示 cost utilization。单个 sample 的 singleton cost 超预算时仍按 oversized 语义输出。

`cost_window_batches > 1` 会缓存对应数量的已规划 batch，并按 estimated cost
降序交给 collate。DDP 各 rank 使用相同窗口时，相近的局部 cost quantile 会倾向于落在
同一步；这个过程没有新 collective，也不会跨 rank 移动 sample。

## Distributed Cost Window

如果 DDP 各 rank 的局部 cost 分布本身不同，可以对完整 plan 做 block-level matching：

```python
loader = LBA(
    dataset,
    len_fn=sample_length,
    cost_fn=attention_cost,
    max_batch_cost=2_000_000,
    distributed_cost_window_batches=8,
    prefetch_batches=8,
    batch_size=32,
    collate_fn=collate_fn,
)
```

`distributed_cost_window_batches=None` 默认关闭；启用值必须至少为 `2`，只支持
map-style dataset，并与 `cost_window_batches > 1` 互斥。它也可以使用默认的
`max_length * batch_size` estimated cost，不强制配置 custom `cost_fn`。在没有初始化
distributed process group 的 iteration 中配置该选项会直接报错。

设窗口为 K 时，每个 rank 先积累 K 个已规划 batch。LBA 每个完整 block gather 一次
metadata，全局按 cost 降序，并把每 `world_size` 个相邻 plan 放到同一个 DDP step；跨
block 的 step rotation 避免固定 rank 总是接收高 cost plan。source 结束时不足 K 但非空
的 partial block 同样 gather 一次。final flush 不走这条路径，继续使用原有公共尾部
重规划和 equal-step 协议。

本地分配的 plan 复用原 sample。远端 plan 在接收 rank 主进程重新调用
`dataset[index]`，因此会重复 read/decode/transform，且不会使用 source worker 的
batched `__getitems__`。LBA 随后重跑 `len_fn`，有效长度变化会直接报错。dataset lookup
必须确定、无副作用、可在 worker 外执行并保持相同有效长度。该模式没有 forward
barrier，也不增加固定的 per-step collective；建议
设置 `prefetch_batches >= K`，并用真实训练的 loader wait、ready queue、step-start
spread 和总 wall time 判断重复读取是否值得。

## Planner 模式

默认 `planner_mode="quality"` 是稳定基线。它使用 recent-window fast path，并在
fast path 没找到达标 batch 时运行代表候选 fallback，优先保持较低 padding：

```python
loader = LBA(
    dataset,
    len_fn=sample_length,
    batch_size=32,
    max_padded_length=8192,
    max_padding_ratio=0.05,
    planner_mode="quality",
)
```

`max_padding_ratio` 是 fast-path readiness threshold，不是所有输出 batch 的硬上限；
fallback 和 final flush 可能超过它。

只有训练侧 loader wait、GPU utilization 和 LBA planner 统计共同表明 producer 成为
瓶颈时，才切换 throughput 模式：

```python
loader = LBA(
    dataset,
    len_fn=sample_length,
    batch_size=32,
    planner_mode="throughput",
    max_candidate_windows=256,
    limited_search_fallback_after=8,
    limited_search_fallback_pool_size=1024,
)
```

throughput 模式限制 steady-state recent-window search，但可能把更多选择工作推到
final flush。adaptive fallback 可以减少 flush 债务，不代表该模式在所有 workload
上都更快。

## Lightning DDP

在 Lightning data hook 中直接返回 LBA，并保持默认 distributed sampler 注入：

```python
import lightning as L


class DataModule(L.LightningDataModule):
    def train_dataloader(self):
        return LBA(
            self.dataset,
            len_fn=sample_length,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
            max_padded_length=8192,
        )


trainer = L.Trainer(
    devices="auto",
    strategy="ddp",
    use_distributed_sampler=True,
)
```

Lightning 会把 LBA 识别为 DataLoader，按 rank 重建并注入 `DistributedSampler`，也会
在 epoch 开始时调用 sampler 的 `set_epoch()`。调用侧不应再手工构造一套 rank
sampler，也不需要额外的 sampler epoch callback。

`DistributedSampler(drop_last=False)` 在 dataset size 不能整除 world size 时会补齐
index，补齐项会在不同 rank 间形成重复样本。这是 PyTorch sampler 的既定语义，LBA
不会去重。训练工程应根据数据守恒要求选择可整除数据、接受重复，或显式选择 drop
策略。

DDP steady state 中，每个非空 source batch 对应一个 planned batch。默认 plan 留在
本 rank；启用 distributed cost window 后，完整 block 和 source 尾部非空 partial block
会交换 metadata 并重新分配整个 plan。final flush 仍汇总各 rank 尾部 records，统一
规划后分发相同步数。默认 `drop_last_flush=True`：尾部不足以让每个 rank 都获得非空
batch 时丢弃并 warning；设置为 `False` 时直接报错。

所有 rank 必须同步消费和停止。某个 rank 单独 break，或在 dataset、`len_fn`、
`collate_fn` 中抛异常，都可能让其他 rank 卡在下一次 collective。显式 budget 与所有
影响 planner 控制流的选项必须跨 rank 一致。
`max_batch_cost`、local/global cost window 和 `max_batches` 同样必须一致，并会在
iteration 开始时校验。local window 排序本身不发起 collective，distributed window 则
每个完整 K-step block 和一个非空 partial source tail 各发起一次 metadata gather。

map-style dataset 的 indexed final flush 和 distributed cost window 都只交换 metadata，
接收 rank 会在主进程重新调用 `dataset[index]`。该读取必须确定、无副作用、可在 worker
外执行，并返回相同的有效长度。依赖 worker 或改变长度的随机 transform 不满足这个
契约；应把不改变有效长度的随机变换放到最终 `collate_fn`。

默认 process group 是 NCCL 时，LBA 会创建独立 Gloo group 同步 CPU metadata，运行
环境必须同时提供 Gloo。

## IterableDataset

IterableDataset 也使用同一个入口：

```python
loader = LBA(
    iterable_dataset,
    len_fn=sample_length,
    batch_size=32,
    collate_fn=collate_fn,
    max_padded_length=8192,
)
```

这个路径必须配置 batched loading；`batch_size=None` 不受支持。iterable 自身决定能否
重复迭代以及 cursor 如何推进。one-shot iterator 不会被 LBA 重建或回放；lookahead
可能已经消费最后一个输出 batch 之后的 source items。

Lightning 和 PyTorch 不会向 IterableDataset 注入 `DistributedSampler`。DDP 下，
dataset 必须按 distributed rank 和 DataLoader worker 自己分片，并保证每个 rank
产生相同数量的非空 source batches。final flush 使用 object gather，因此尾部 sample
必须可 pickle。

`distributed_cost_window_batches` 不支持 IterableDataset，并会在 loader 构造时直接
报错；iterable steady state 继续使用 rank-local planning。

## `max_batches`

`max_batches` 限制单次 iteration 最多输出多少个最终 batch。source 先耗尽且输出尚未
达到上限时，final flush 仍可继续输出；达到上限后 planner 会关闭并丢弃剩余
lookahead。它适合显式 bounded segment，不等价于 `DataLoader.drop_last`。

## Checkpoint Resume

LBA 不保存 source iterator cursor、planner pool 或 prefetch lookahead。epoch 边界恢复
可以沿用确定性 sampler 的正常契约，但 mid-epoch checkpoint 不保证和未中断运行具有
完全相同的后续 sample 序列；恢复后可能相对原运行重复或跳过 source samples。

如果任务要求 step checkpoint 精确续采，需要由 stateful dataset/sampler 保存游标，
并额外把 LBA pending planner state 纳入 checkpoint。Lightning 能恢复 model、
optimizer 和 loop state，不自动提供这项数据连续性保证。

## 日志与预取

每个 LBA 实例在首次迭代或访问日志属性时惰性创建一份 `.log` 和一份 `.jsonl`：

```text
~/.lba/logs/lba-YYYYmmdd-HHMMSS-ffffff-PID.log
~/.lba/logs/lba-YYYYmmdd-HHMMSS-ffffff-PID.jsonl
```

路径在 `loader.log_path` 和 `loader.log_event_path`。iteration 退出后，解析出的预算在
`loader.last_max_padded_length`；legacy 或 custom 模式实际使用的 cost budget 在
`loader.last_max_batch_cost`，planner 计数在 `loader.last_planner_stats`。

默认 `prefetch_batches=4`。设置为 `0` 可以关闭后台 producer。初始化
`torch.distributed` 后，LBA 会在启用后台 producer 前创建独立 Gloo metadata group。
这样 source-batch sync、planner、final `collate_fn` 和 pinning 可以提前进入 ready
queue，同时避免 producer thread 和训练线程在默认 process group 上交错发起
collective。
