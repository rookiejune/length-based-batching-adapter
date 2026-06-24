# LBA 用法设计

LBA 的第一目标是让用户尽量少改训练代码。用户先照常创建原始 PyTorch
`DataLoader`，再用 `LBA` 包一层。

## 推荐用法

```python
from lba import LBA

loader = LBA(
    dataloader,
    len_fn=lambda sample: len(sample["input_ids"]),
)

for batch in loader:
    ...
```

## 完整类名

```python
from lba import LengthBatchingAdapter

loader = LengthBatchingAdapter(dataloader, len_fn=len_fn)
```

`LBA` 是 `LengthBatchingAdapter` 的短别名，两者行为完全一致。

## 显式目标长度

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

## Padding 阈值

`max_padding_ratio` 默认是 `0.05`。也就是候选 batch 的 padding ratio 低于
5% 时，planner 可以直接提交这个 batch：

```python
loader = LBA(dataloader, len_fn=len_fn, max_padding_ratio=0.05)
```

如果更重视吞吐，可以调高这个值；如果更重视 padding，则调低这个值。

## Planner 模式

默认 `planner_mode="quality"`，不限制 recent-window 候选枚举，并在 fast path
找不到达标 batch 时继续完整搜索，优先保持较低 padding。

如果真实训练里 planner producer 跟不上 GPU，可以显式切到 throughput 模式：

```python
loader = LBA(
    dataloader,
    len_fn=len_fn,
    planner_mode="throughput",
    max_candidate_windows=256,
)
```

`throughput` 模式只限制普通迭代中的 recent-window 搜索；当受限搜索找不到达标
batch 时，本次 `pop_ready` 会等待更多 records，而不是立即做完整 fallback。
final flush 仍然完整搜索并排出剩余样本。`max_candidate_windows=None` 在
`quality` 模式下表示不限制；`throughput` 模式不显式设置时默认使用 `256`。

## 日志路径

默认每次运行会写一个人类可读日志和一个结构化事件文件：

```text
~/.lba/logs/lba-YYYYmmdd-HHMMSS-PID.log
~/.lba/logs/lba-YYYYmmdd-HHMMSS-PID.jsonl
```

用户也可以指定：

```python
loader = LBA(dataloader, len_fn=len_fn, log_dir="outputs/lba_logs")
```

`.log` 里默认只放训练时最需要扫的三类信息：padding 改善、planner 开销和健康
计数。完整的 before/after 长度统计、planner path 拆分、spill、oversized 和 DDP
事件写入同名 `.jsonl`，方便 benchmark 或回归脚本解析。

## 预取

默认 `prefetch_batches=4`，LBA 会用 bounded prefetch queue 在后台提前准备
batch。也可以显式调整 queue 深度：

```python
loader = LBA(dataloader, len_fn=len_fn, prefetch_batches=4)
```

需要严格同步迭代或排查线程相关问题时，可以设置 `prefetch_batches=0` 关闭后台
producer。
