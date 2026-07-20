# LBA 用法设计

LBA 的第一目标是让用户尽量少改训练代码。用户先照常创建原始 PyTorch
`DataLoader`，再用 `LBA` 包一层。

当前版本是稳定 v1。v1 推荐把默认 `planner_mode="quality"` 作为基线配置：先保证
低 padding、明确日志和 DDP step 对齐，再按 benchmark 结果决定是否打开吞吐取舍。

## 推荐用法

```python
from lba import LBA


def sample_length(sample):
    return len(sample["input_ids"])


loader = LBA(
    dataloader,
    len_fn=sample_length,
)

for batch in loader:
    ...
```

`DataLoader` 使用 `spawn` multiprocessing context 时，`len_fn` 会随 source
collator 一起传到 worker，必须可 pickle。此时使用模块顶层函数或 callable class，
不要使用 lambda 或局部函数。

如果模型有明确的最大 padded length 或 token budget，推荐显式传入：

```python
loader = LBA(
    dataloader,
    len_fn=len_fn,
    max_padded_length=8192,
    log_dir="outputs/lba_logs",
)
```

## 完整类名

```python
from lba import IterableLengthBatchingAdapter, LengthBatchingAdapter

loader = LengthBatchingAdapter(dataloader, len_fn=len_fn)
```

`LBA` 是 `LengthBatchingAdapter` 的短别名，两者行为完全一致。
`IterableLBA` 同样是 `IterableLengthBatchingAdapter` 的短别名。两种 adapter 是独立
的 iterable 类，不是 `DataLoader` 子类；它们不实现 `__len__`，因为动态规划后的
batch 数无法预先准确给出。

## Iterable 入口

调用侧已经生成 raw sample batches 时，可以绕过 `DataLoader` 重建：

```python
from lba import IterableLBA

loader = IterableLBA(
    source_batches,
    collate_fn=collate_fn,
    len_fn=sample_length,
    batch_size=32,
)
```

这里的 `batch_size` 只参与 `max_padded_length` 自动推断，不限制最终动态 batch 的
样本数；显式给出 `max_padded_length` 时可以省略。adapter 的可重复迭代能力取决于
`source_batches`：list、tuple 或其他可重入 iterable 可以再次迭代，generator 等
one-shot iterator 不会被重建或回放。只消费前缀时，下次 iteration 从当前游标继续；
lookahead / prefetch 可能已经读过最后一个输出 batch 之后的 source items，完整消费后
则保持耗尽。

`IterableLBA` 和 `LBA` 都支持 `max_batches`。source 先耗尽且输出尚未达到上限时，
final flush 可以继续产出，直到 cache 清空或达到上限；一旦达到上限，iteration 会关闭
planner 并丢弃剩余 lookahead。该选项适合按固定训练 step 切换阶段的调用侧，不应用
来模拟普通 `DataLoader.drop_last`。

## 长度预算

用户可以直接指定最大 padded length：

```python
loader = LBA(
    dataloader,
    len_fn=len_fn,
    max_padded_length=8192,
)
```

显式设置 `max_padded_length` 时，LBA 会发出 warning，提醒用户该值会覆盖
由原始 batch size 推导目标长度的流程。

省略该参数时，LBA 会先读取 warmup source batches，再按下面的方式推断：

```python
max_padded_length = ceil(mean(warmup_sample_lengths) * source_batch_size)
```

`warmup_batches` 默认是 `min(batch_size, 32)`；拿不到有效 source batch size 时，两者
分别退化为 `1` 和 warmup sample 数。warmup samples 会继续进入 planner，不会因为
推断预算而丢失。空输入无法推断预算；如果空迭代本身是合法结果，需要显式给出
`max_padded_length`。

预算约束的是 `len_fn` 返回值对应的
`max_length_in_batch * batch_size`，不会检查最终 `collate_fn` 产出的 tensor shape。
普通 planned batch 不超过预算；单个 sample 自身已经超出预算时，LBA 会把它作为
singleton 输出并发出 warning，详见 [边界情况](edge_cases.md#超长样本)。

## Padding 阈值

`max_padding_ratio` 默认是 `0.05`。也就是候选 batch 的 padding ratio 小于等于
5% 时，planner 可以直接提交这个 batch：

```python
loader = LBA(dataloader, len_fn=len_fn, max_padding_ratio=0.05)
```

如果更重视吞吐，可以调高这个值；如果更重视 padding，则调低这个值。它是 fast
path 的提交阈值，不是所有输出 batch 的硬上限；代表候选 fallback 和 final flush
仍可能产出高于该 ratio 的 batch。

## Planner 模式

默认 `planner_mode="quality"` 是 v1 稳定基线。它不限制 recent-window 候选枚举，
并在 fast path 找不到达标 batch 时继续使用不设上限的代表候选搜索，优先保持较低
padding，同时避免全量连续窗口枚举的 CPU 成本。

如果训练侧 loader wait / GPU utilization 与 LBA planner 统计共同表明 producer 是
瓶颈，可以显式切到 throughput 模式：

```python
loader = LBA(
    dataloader,
    len_fn=len_fn,
    planner_mode="throughput",
    max_candidate_windows=256,
    limited_search_fallback_after=8,
    limited_search_fallback_pool_size=1024,
)
```

`throughput` 模式只限制普通迭代中的 recent-window 搜索；当受限搜索找不到达标
batch 时，通常会等待更多 records，而不是立即做不设上限的代表候选 fallback。为了避免把
未解决的候选选择全部推迟到 final flush，throughput 模式默认在连续 miss 达到
`8` 次时允许一次不设上限的代表候选搜索；当 planner pool 达到
`min(max_cache_samples, 1024)` 条 records 时，会先取消本次 threshold search 的
candidate-window 上限，让 steady-state 阶段提前偿还一部分 flush 债务。final
flush 仍然使用不设上限的代表候选搜索并排出剩余样本。
`max_candidate_windows=None` 在 `quality` 模式下表示不限制；`throughput` 模式不
显式设置时默认使用 `256`。

这套 throughput/adaptive 策略没有替换 v1 默认值，因为缩小候选范围会把一部分
选择成本推迟到 flush，可能增加最终 batch 数或 padding；adaptive fallback 能减少
flush 债务，但会把更多搜索工作放回普通迭代。它适合有明确 CPU planner 瓶颈的
训练任务，不是无条件更优的策略。

## DDP 用法

DDP 下各 rank 的 source `DataLoader` 需要产生相同数量的非空 source batches，常见
做法是 map-style dataset 配合 `DistributedSampler`；相同约束也适用于
`IterableLBA.source_batches`。所有 rank 必须同步消费并同步停止；某个 rank 单独 break
iterator，或在 source、`len_fn`、`collate_fn` 中报错，可能让其他 rank 卡在下一次
collective。LBA 固定使用 default process group，没有 subgroup 参数，因此该 group 的
所有成员都必须参与。如果显式设置
`max_padded_length`，每个 rank 应使用相同值；如果让 LBA warmup 推断，LBA 会同步
各 rank 的预算并取最大值。其他影响规划或控制流的配置也必须相同，尤其是 padding
与 search 配置、`drop_last_flush` 和 `max_batches`；LBA 当前不会逐项跨 rank 校验。

final flush 时，LBA 会把各 rank 剩余样本重新规划成相同数量的 DDP steps。
`drop_last_flush=True` 是默认值：尾部无法给每个 rank 组成非空 step 的样本会被
丢弃并写入 warning；如果这类尾部丢弃不可接受，可以设置为 `False`，让相同情况
直接报错。这个选项不改变 source loader 自身的 `drop_last`，也不阻止
`max_batches` 或调用侧提前终止造成的丢弃。

map-style `DataLoader` 的 final flush 只交换 `(sample_index, length)` metadata，并在
接收 rank 的主进程重新调用 `dataset[index]`。LBA 只检查 index 是否存在，不会检测
dataset 是否可重放；调用侧必须保证该读取在 worker 外可用、确定、无副作用，而且
返回与第一次读取相同的 sample 和有效长度。随机或 worker-sensitive transform 应
移到不改变 `len_fn` 有效长度的最终 `collate_fn`，或者改用 `IterableLBA`，让 final
flush gather 原 sample object。

object gather 要求尾部 sample 可 pickle。NCCL default process group 旁需要可用的
Gloo backend，供 LBA 同步 CPU metadata；不满足这些前置条件时 LBA 会直接报错。

## 日志路径

每个 adapter 实例会创建一个人类可读日志和一个结构化事件文件；同一实例中实际进入
planner 的 iteration 会继续向这对文件写 summary，`max_batches=0` 则在此前返回：

```text
~/.lba/logs/lba-YYYYmmdd-HHMMSS-ffffff-PID.log
~/.lba/logs/lba-YYYYmmdd-HHMMSS-ffffff-PID.jsonl
```

用户也可以指定：

```python
loader = LBA(dataloader, len_fn=len_fn, log_dir="outputs/lba_logs")
```

`.log` 里默认只放训练时最需要扫的三类信息：padding 改善、planner 开销和健康
计数。完整的 before/after 长度统计、planner path 拆分、spill、oversized 和 DDP
事件写入同名 `.jsonl`，方便 benchmark 或回归脚本解析。

实际路径可以直接从 `loader.log_path` 和 `loader.log_event_path` 获取。底层 planner
iteration 实际退出后，解析出的预算在 `loader.last_max_padded_length`，planner 计数
在 `loader.last_planner_stats`。`max_batches=0` 不进入 planner，这两个字段保持初始值。
`loader.max_padded_length` 只返回构造时配置值；使用 warmup 推断时它仍是 `None`。

## 预取

默认 `prefetch_batches=4`，LBA 会用 bounded prefetch queue 在后台提前准备
batch。也可以显式调整 queue 深度：

```python
loader = LBA(dataloader, len_fn=len_fn, prefetch_batches=4)
```

需要严格同步迭代或排查线程相关问题时，可以设置 `prefetch_batches=0` 关闭后台
producer。初始化 `torch.distributed` 后 prefetch 会自动关闭，避免 producer 线程与
训练线程交错发起 collective。
