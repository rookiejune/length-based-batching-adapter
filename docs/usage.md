# LBA 用法设计

LBA 的第一目标是让用户尽量少改训练代码。用户先照常创建原始 PyTorch
`DataLoader`，再用 `LBA` 包一层。

当前版本是稳定 v1。v1 推荐把默认 `planner_mode="quality"` 作为基线配置：先保证
低 padding、明确日志和 DDP step 对齐，再按 benchmark 结果决定是否打开吞吐取舍。

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

默认 `planner_mode="quality"` 是 v1 稳定基线。它不限制 recent-window 候选枚举，
并在 fast path 找不到达标 batch 时继续完整搜索，优先保持较低 padding。

如果真实训练日志显示 planner producer 跟不上 GPU，可以显式切到 throughput 模式：

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
batch 时，通常会等待更多 records，而不是立即做完整 fallback。为了避免把
未解决的候选选择全部推迟到 final flush，throughput 模式默认在连续 miss 达到
`8` 次时允许一次完整搜索；当 planner pool 达到 `1024` 条 records 时，会先取消
本次 threshold search 的 candidate-window 上限，让 steady-state 阶段提前偿还
一部分 flush 债务。final flush 仍然完整搜索并排出剩余样本。
`max_candidate_windows=None` 在 `quality` 模式下表示不限制；`throughput` 模式不
显式设置时默认使用 `256`。

这套 throughput/adaptive 策略没有替换 v1 默认值，因为缩小候选范围会把一部分
选择成本推迟到 flush，可能增加最终 batch 数或 padding；adaptive fallback 能减少
flush 债务，但会把更多搜索工作放回普通迭代。它适合有明确 CPU planner 瓶颈的
训练任务，不是无条件更优的策略。

## DDP 用法

DDP 下各 rank 的 source `DataLoader` 需要产生相同数量的 source batches，常见做法
是 map-style dataset 配合 `DistributedSampler`。如果显式设置
`max_padded_length`，每个 rank 应使用相同值；如果让 LBA warmup 推断，LBA 会同步
各 rank 的预算并取最大值。

final flush 时，LBA 会把各 rank 剩余样本重新规划成相同数量的 DDP steps。
`drop_last_flush=True` 是默认值：尾部无法给每个 rank 组成非空 step 的样本会被
丢弃并写入 warning；需要严格保留样本时可以设置为 `False`，此时 LBA 会直接报错。

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
