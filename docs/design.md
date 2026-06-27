# LBA 设计

LBA 是 Length-based Batching Adapter。它包装用户原本的 PyTorch
`DataLoader`，按样本有效长度重新组织动态 batch。

## API

```python
from lba import LBA

loader = LBA(
    dataloader,
    len_fn=lambda sample: len(sample["input_ids"]),
    max_padded_length=8192,
)
```

`max_padded_length` 是硬预算，语义是：

```python
max_length_in_batch * batch_size <= max_padded_length
```

v1 默认策略：

- `max_padding_ratio=0.05`，默认偏向减少 padding。
- `prefetch_batches=4`，默认用 bounded queue 提前准备最终 batch；可设置为 `0` 关闭。
- `drop_last_flush=True`，DDP final flush 尾部无法给每个 rank 组成非空 step 时，
  默认丢弃尾部并发 warning；需要样本完整性时可改为 `False` 让训练直接报错。
- `planner_mode="quality"`，默认不限制候选窗口，保留完整搜索 fallback。
- `planner_mode="throughput"` 时，普通迭代只检查有限数量的 recent-window 候选；
  默认上限是 `256`，也可以用 `max_candidate_windows` 显式调整。
- throughput 模式默认启用 adaptive 偿还机制：连续 capped search miss 达到
  `8` 次时允许一次完整搜索；planner pool 达到 `min(max_cache_samples, 1024)`
  时取消本次 threshold search 的 candidate-window 上限，避免把候选选择成本
  全部推迟到 final flush。
- 第一阶段性能目标是 CPU batch 生产速度高过 GPU 消费速度，先按 `>= 5 it/s`
  作为最低目标。
- 默认 planner 继续偏向低 padding，不默认开启会增大 padding ratio 的近似搜索；
  追求吞吐的策略必须显式 opt-in，并在 benchmark 中同时报告 padding 和 planner
  开销。

## v1 稳定边界

当前版本定位为稳定 v1。稳定边界是：对外仍然表现为一个 `DataLoader` wrapper；
`len_fn` 在原始 `collate_fn` 前运行；最终 batch 继续交给用户原始 `collate_fn`；
默认 quality planner 保留完整 fallback；DDP 下 final flush 保证每个 rank 有相同
step 数。

v1 不承诺动态 batch 的精确样本顺序，也不承诺不同 planner 模式之间产生相同 batch
边界。只影响吞吐或候选近似的策略必须显式 opt-in，不能改变默认 quality 行为。

## 流程

1. wrapper 保存用户原始 `collate_fn`。
2. 内部 source loader 使用 record collate，把 raw samples 转成 `(sample, length)`。
   对 map-style dataset，source loader 复用原始 `batch_sampler`；对
   `IterableDataset`，source loader 复用 `batch_size` 和 `drop_last`。
3. PyTorch worker 继续负责 dataset 读取、decode、transform 和 `len_fn`。
4. 主进程为 records 分配 `arrival_id`。
5. 主进程 planner 维护全局 sample pool。
6. planner 选出动态 batch 后，wrapper 调用原始 `collate_fn`。

## Prefetch Producer

默认 `prefetch_batches=4`。当 `prefetch_batches > 0` 时，LBA 会启动一个后台线程，
提前运行 source loader、planner 和原始 `collate_fn`，并把最终 batch 放入 bounded
queue。需要严格同步迭代或排查线程相关问题时，可以设置 `prefetch_batches=0`。

第一版 producer 使用线程而不是进程：

- 避免把任意 Python sample 或 collated batch 额外 pickle 到子进程。
- 先验证 GPU 训练消费阶段能否为 CPU producer 留出足够时间。
- 如果线程 producer 仍然不能让 queue 保持非空，再讨论独立进程 producer。

## 145 Benchmark 结论

详见 [145 Benchmark 记录](benchmark_145.md)。

2026-06-19 在 `145.pami.group` 上的 Wikitext benchmark 显示，LBA 可以把
padded length 降低约 65% 到 66%，padding ratio 从约 67% 降到约 3.8%。

同时，当前实现的耗时主要来自主进程 planner，而不是 IO：

- Wikitext 20k 下，`num_workers=0` 和 `num_workers=4` 的 LBA 耗时几乎一样。
- 50k Wikitext 已经需要约 65 秒，不适合继续直接放大数据规模。
- 下一轮设计不应追求纯 CPU 迭代速度接近 baseline，而应先保证 batch 生产速度
  能稳定高过 GPU 训练消费速度。

这个结论决定了当前实现策略：先避免复杂的 planner 状态更新，接受足够好的次优
batch 选择，并用 DataLoader worker、异步 producer 和预取队列覆盖训练消费时间。
完整 per-worker planner 暂不做，因为还需要先解决 worker 结束时如何 flush 剩余
样本的问题。

## Planner

planner 使用按 `(length, arrival_id)` 排序的 sample pool，并维护 prefix sum 以及
arrival-id range-min 索引。prefix sum 用于计算窗口 raw length，range-min 用于
在候选比较时快速取得窗口内最早到达的 record。

候选连续窗口 `[left, right]` 的统计量：

```python
raw_length_sum = prefix[right + 1] - prefix[left]
max_length = records[right].length
batch_size = right - left + 1
padded_length = max_length * batch_size
padding_length = padded_length - raw_length_sum
padding_ratio = padding_length / padded_length
```

候选 batch 必须满足 `padded_length <= max_padded_length`。普通迭代中，
`pop_ready()` 先只枚举包含最近新增 records 的候选窗口；如果某个候选的
`padding_ratio <= max_padding_ratio`，可以走 fast path 直接提交。这个局部搜索
拿不到候选时，再完整搜索所有候选窗口作为 fallback。final flush 不使用 recent
限制，会从剩余 pool 中继续搜索直到清空。
默认 `max_padding_ratio=0.05`，这是根据 145 Wikitext benchmark 在 padding
质量和 producer 速度之间取得的折中。

`planner_mode="throughput"` 是显式 opt-in 的吞吐模式。它会给普通迭代的
recent-window 枚举加上 `max_candidate_windows` 上限；受限搜索未命中时，通常
本次 `pop_ready()` 直接返回 `None`，等待后续 records，而不是立刻进入完整搜索。
但如果连续 miss 达到 `limited_search_fallback_after`，或 planner pool 达到
`limited_search_fallback_pool_size`，会进入 adaptive 偿还路径：前者允许一次完整
搜索，后者取消本次 threshold search 的 candidate-window 上限。这样既能限制常规
steady-state 调用的 CPU work，又避免所有未解决的候选选择在 final flush 集中爆发。
flush 路径仍然完整搜索，避免尾部样本因为吞吐模式被跳过。

145 benchmark 后，当前 planner 的问题不是 sorted pool 和 prefix sum 本身，而是
每次 `pop_ready()` 后反复生成和比较大量候选窗口。当前先使用 recent-window
局部搜索降低 steady-state 成本，用 range-min 降低候选构造成本，并保留完整搜索
兜底；如果真实训练中仍然跟不上 GPU，再考虑更复杂的候选缓存、长度 bucket 或
aging 策略。

## 为什么不替换默认策略

这一轮尝试过的优化可以分成三类：

- 已纳入默认实现的结构优化：`CandidateIndex` 统一维护 sorted records、prefix
  lengths 和 arrival-id range-min，减少候选构造时的重复状态传递；日志和 DDP
  flush 也拆成独立 helper，降低 wrapper 和 coordinator 的职责混杂。这些优化不改
  batch 选择语义，因此适合进入 v1 默认。
- 保留为 opt-in 的 throughput 优化：给 recent-window 搜索加上候选窗口上限，能
  限制普通 `pop_ready()` 的 CPU work，但候选范围太窄时会让更多样本滞留到
  final flush，可能增加 batch 数和尾部搜索成本。
- 保留为 opt-in 的 adaptive 偿还：连续 capped-search miss 后允许完整 fallback，
  或在 pool 足够大时临时取消本次 threshold search 上限。它能明显减少 flush 债务，
  但会把额外搜索工作放回 steady-state，padding 和总耗时也不是稳定胜过 quality。

因此 v1 默认继续使用 quality planner。这个默认值已经是比较稳的实现：padding
质量好、flush 行为清晰、DDP 契约明确，也没有一个轻量改动能在 padding、吞吐和
final flush 三个维度同时稳赢。

## Spill

当内存 sample pool 超过 `max_cache_samples` 时，planner 将最早进入且暂未选中的
样本写入磁盘 shard。默认每个 shard 最多 `10_000` 个样本。spill 成功后，样本
从内存 pool 删除。

DDP 模式下，如果用户传入共享 `spill_dir`，adapter 会在其下按 rank 创建子目录，
避免不同进程写入相同 shard 文件名。

## 日志

默认日志目录为：

```text
~/.lba/logs/
```

每次 adapter 运行写两份日志：

- `.log`：给训练时人工扫读，固定输出 padding 改善、planner 开销和健康计数。
- `.jsonl`：给 benchmark 和排障脚本解析，记录完整 summary、spill、oversized
  sample、DDP final flush 等结构化事件。
