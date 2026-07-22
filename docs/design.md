# LBA 设计

LBA 是直接从 dataset 构造的 length-based `DataLoader`。它在标准 PyTorch
sampling/worker 语义之上，按样本有效长度重新组织动态 batch。

## API

```python
from lba import LBA


def sample_length(sample):
    return len(sample["input_ids"])


loader = LBA(
    dataset,
    len_fn=sample_length,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn,
    max_padded_length=8192,
)
```

对普通 planned batch，`max_padded_length` 是硬预算，语义是：

```python
max_length_in_batch * batch_size <= max_padded_length
```

单个 sample 的 `len_fn` 返回值已经超过预算时，planner 会把它作为 singleton 输出并
记录 oversized warning；这是唯一主动突破该预算的路径。预算只约束 `len_fn` 度量的
有效长度，不检查最终 `collate_fn` 输出的 tensor shape。

可选 `cost_fn(max_length, batch_size)` 用自定义计算成本替换
`max_length * batch_size`。custom cost 模式要求显式 `max_batch_cost`，
并与 `max_padded_length`、`warmup_batches` 互斥。`len_fn` 仍负责
sample pool 排序和 padding 指标；`cost_fn` 只接收候选 batch 的聚合形状，必须
返回正整数，并随两个参数单调不减。planner 依赖这个单调性二分搜索最大可行 batch size。

v2 只保留这一个公开入口。map-style dataset 和 `IterableDataset` 都通过
`LBA(dataset, ...)` 进入；不再公开接收既有 loader 或 raw batch stream 的旁路入口。
`LBA` 继承 PyTorch `DataLoader`，让 Lightning 等框架可以识别、检查并重建 loader。
动态规划后的 batch 数运行前不可知，因此 `__len__` 明确不可用。

`max_batches` 设置后，loader 最多产出指定数量的最终 batch；
source 先耗尽且输出尚未达到上限时，正常 final flush 会继续产出，直到 cache 清空或
达到上限。一旦达到边界，iteration 关闭当前 planner 并丢弃尚未输出的 sample cache，
不再额外 flush。这个模式用于调用侧需要按训练 step 切换阶段时清理 LBA lookahead
cache。DDP 下已经达到边界的 rank 会继续参与 source-batch 同步并丢弃新读到的本段
样本，直到所有 rank 都到达该 batch 边界。

v2 默认策略：

- `max_padding_ratio=0.05`，默认偏向减少 padding。
- `prefetch_batches=4`，默认用 bounded queue 提前准备最终 batch；可设置为 `0` 关闭。
- `cost_window_batches=1`，默认不缓存额外已规划 batch；设置为大于 `1` 时，
  每个局部窗口按 estimated cost 降序输出。
- `distributed_cost_window_batches=None`，默认不交换 steady-state plan；设置为至少
  `2` 时，map-style DDP 每个 rank 按 plan block 做全局 cost matching。它与
  `cost_window_batches > 1` 互斥。
- `adaptive=None`，默认关闭；设置 `AdaptiveConfig(...)` 时启用参数级自适应。
  在 `AdaptiveConfig` 内省略字段表示禁用，字段值为 `None` 表示由 LBA 默认策略自动调。
  `AdaptiveConfig()` 默认只自动调整 `max_padding_ratio`。如果启用 adaptive
  `distributed_cost_window_batches`，它与静态 `distributed_cost_window_batches` 和
  `cost_window_batches > 1` 互斥。
- `max_batches=None`，默认不限制本次 adapter 迭代的输出 batch 数；设置后用于 bounded
  segment，达到边界后不再继续 final flush。
- `drop_last_flush=True`，DDP final flush 尾部无法给每个 rank 组成非空 step 时，
  默认丢弃尾部并发 warning；如果这类尾部丢弃不可接受，可改为 `False` 让训练报错。
- `planner_mode="quality"`，默认不限制候选窗口，使用代表候选 fallback。
- `planner_mode="throughput"` 时，普通迭代只检查有限数量的 recent-window 候选；
  默认上限是 `256`，也可以用 `max_candidate_windows` 显式调整。
- throughput 模式默认启用 adaptive 偿还机制：连续 capped search miss 达到
  `8` 次时允许一次不设候选窗口上限的代表候选搜索；planner pool 达到 `min(max_cache_samples, 1024)`
  时取消本次 threshold search 的 candidate-window 上限，避免把候选选择成本
  全部推迟到 final flush。
- 默认 planner 继续偏向低 padding，不默认开启会增大 padding ratio 的近似搜索；
  追求吞吐的策略必须显式 opt-in，并在 benchmark 中同时报告 padding 和 planner
  开销。

## v2 稳定边界

稳定边界是：`LBA` 是直接从 dataset 构造的 DataLoader 子类；`len_fn` 在最终
`collate_fn` 前运行；最终 batch 继续交给用户显式传入的 `collate_fn`；默认 quality
planner 保留不设上限的代表候选 fallback；DDP 下 final flush 保证每个 rank 有相同
步数。

v2 不承诺动态 batch 的精确样本顺序，也不承诺不同 planner 模式之间产生相同 batch
边界。只影响吞吐或候选近似的策略必须显式 opt-in，不能改变默认 quality 行为。
`max_padding_ratio` 是 fast-path readiness threshold，不是所有输出 batch 的硬上限；
代表候选 fallback 和 final flush 可能超过它。

LBA 不 checkpoint source iterator cursor、planner pool 或 prefetched lookahead。
epoch-boundary resume 可以依赖确定性 sampler；mid-epoch resume 不保证精确 sample
continuity，需要调用侧额外提供 stateful dataset/sampler 和 pending planner state。

## Lightning 重建

Lightning 只向 DataLoader 注入 distributed sampler。v2 的构造边界必须让它捕获
dataset、标准 DataLoader 参数、`len_fn` 和 planner 参数，并在重建后把新 sampler
真正传入内部 source loader。只在外层暴露 `sampler` 属性而内部仍使用旧 sampler 不
满足该契约。

map-style dataset 可以使用 Lightning 默认的 `use_distributed_sampler=True`；Lightning
负责注入 `DistributedSampler` 和推进 `set_epoch()`。调用侧不能再同时维护手工 rank
sampler。dataset size 不能整除 world size 时，`DistributedSampler(drop_last=False)`
会补齐重复 index，LBA 保留这个显式语义，不做去重。

IterableDataset 不走自动 sampler 注入，必须按 distributed rank 和 worker 自己分片，
并保证所有 rank 的非空 source batch 数相同。

## 流程

1. LBA 保存 dataset、标准 DataLoader 配置、`len_fn`、最终 `collate_fn` 和 planner
   配置。
2. 内部 source loader 使用 record collate，把 raw samples 转成 `(sample, length)`。
   对 map-style dataset，source loader 复用 LBA 的 `batch_sampler`；对
   `IterableDataset`，source loader 复用 `batch_size` 和 `drop_last`。
3. PyTorch worker 继续负责 dataset 读取、decode、transform 和 `len_fn`。
4. 主进程为 records 分配 `arrival_id`。
5. 主进程 planner 维护 adapter-local sample pool；默认 DDP steady state 中各 rank
   独立规划。启用 distributed cost window 后，每个 block 只交换已完成 plan 的 metadata，
   不重新切分 plan 内的 samples。
6. global matcher 把整个 plan 分配给目标 rank；本地 plan 复用已读取 sample，远端 plan
   在接收 rank 主进程通过 `dataset[index]` 重取。随后 iteration pipeline 调用原始
   `collate_fn`。未启用 matcher 时直接进入 collate。
7. LBA 配置了 `pin_memory=True` 时，在最终 `collate_fn` 之后递归 pin
   最终 batch；内部 `LengthRecord` 不走 pin-memory queue。

final flush 不使用 distributed cost window，仍然聚合各 rank 剩余 records、统一重规划
并分配相同步数。

map-style source loader 在 adapter 内惰性创建并跨 iteration 复用，因此
`persistent_workers=True` 可以跨 epoch 保留 worker。索引 dataset wrapper 转发
`__getitems__` 和普通属性读写；`get_worker_info().dataset` 的对象身份仍是 wrapper，
需要原 dataset 时使用它的 `dataset` 属性。

## Prefetch Producer

默认 `prefetch_batches=4`。当 `prefetch_batches > 0` 时，LBA 会启动一个后台线程，
提前运行 source loader、planner 和原始 `collate_fn`，并把最终 batch 放入 bounded
queue。需要严格同步迭代或排查线程相关问题时，可以设置 `prefetch_batches=0`。

source `DataLoader` iterator 和它的 multiprocessing workers 在调用 `iter(LBA)` 的
线程中先创建，再由 producer thread 消费，避免在线程内 `fork` worker。任意 dataset
I/O 无法被 Python 线程强制中断；提前关闭 iterator 后 producer 若仍阻塞超过 1 秒，
LBA 会发出 warning，调用侧应给 source loader 配置有限 `timeout`。

DDP 下 producer thread 会继续执行 source-batch presence、budget sync、batch-limit、
distributed cost block 和 final-flush metadata collective。为了避免这些 metadata
collective 和训练
线程的 forward/backward collective 在默认 process group 上交错，`prefetch_batches > 0`
时 LBA 会先在调用 `iter(LBA)` 的线程中创建独立 Gloo metadata group，再启动 producer。
这个 prefetch 改动只改变 collective 使用的 group；distributed cost matcher 自身按
plan block 摊销 metadata gather，不增加固定的 per-step collective。

v2 producer 使用线程而不是进程：

- 避免把任意 Python sample 或 collated batch 额外 pickle 到子进程。
- 先验证 GPU 训练消费阶段能否为 CPU producer 留出足够时间。
- 如果线程 producer 仍然不能让 queue 保持非空，再讨论独立进程 producer。

v2 不做 per-worker planner；在引入前必须先定义 worker 结束时如何合并或 flush 剩余
records，不能让 worker-local cache 静默丢样本。独立进程 producer 同样只在训练侧
指标证明线程 producer 无法喂满消费端后再设计。

## 145 Benchmark 演进

详见 [145 Benchmark 记录](benchmark_145.md)。

2026-06-19 的初始实现单次 Wikitext benchmark 显示，LBA 可以把 padded length
降低约 65% 到 66%，padding ratio 从约 67% 降到约 3.8%。当时的主要耗时来自
主进程 planner，而不是 IO：

- Wikitext 20k 下，`num_workers=0` 和 `num_workers=4` 的 LBA 耗时几乎一样。
- 当时 50k Wikitext 需要约 65 秒，因此先优化 planner，再放大 benchmark。
- 早期阶段把 CPU batch 生产速度高过 GPU 消费速度作为目标，并用 `>= 5 it/s`
  作为最低观察线；这不是 v2 的吞吐承诺。

这些历史结果推动了 recent-window、range-min、lazy candidate、prefetch 和 adaptive
throughput 等后续工作。2026-07-19 的四次重复 2-GPU Wikitext 复测中，quality LBA
把 padding ratio 从 `68.24%` 降到 `3.50%`，但在 `compute_iters=0`、
`simulate_step_sec=0` 的最小模型 step 中仍比 baseline 慢；throughput-256 减少约
`8.7%` candidate checks，也没有形成稳定 wall-time 优势。
因此当前结论仍是先用真实模型的 token/sec、step/sec、GPU utilization 和 loader wait
判断 producer 是否成为瓶颈，不能从纯 loader benchmark 推导训练加速。

## Planner

`BatchPlanner` 维护按 `(length, arrival_id)` 排序的 sample pool。需要搜索时，
`CandidateIndex` 为当前 pool 构造 prefix sum 和 sorted lengths，并在首次需要时构造
arrival-id range-min view。prefix sum 用于计算窗口 raw length，range-min 用于在
候选比较时取得窗口内最早到达的 record。

候选连续窗口 `[left, right]` 的统计量：

```python
raw_length_sum = prefix[right + 1] - prefix[left]
max_length = records[right].length
batch_size = right - left + 1
padded_length = max_length * batch_size
padding_length = padded_length - raw_length_sum
padding_ratio = padding_length / padded_length
estimated_cost = cost_fn(max_length, batch_size)
```

legacy 模式的 `cost_fn` 等价于 `max_length * batch_size`，budget 是
`max_padded_length`。custom cost 模式使用用户回调和 `max_batch_cost`。
候选 batch 必须满足 `estimated_cost <= budget`；oversized singleton 在进入
候选搜索前单独处理。普通迭代中，
`pop_ready()` 先只枚举包含最近新增 records 的候选窗口；如果某个候选的
`padding_ratio <= max_padding_ratio`，可以走 fast path 直接提交。这个局部搜索
拿不到候选时，再使用不设 recent 限制和 candidate-window 上限的代表候选
fallback。代表候选不是全量连续窗口枚举；每个右端点会检查预算允许的最宽窗口、
padding threshold 对应的 tight 窗口、tight 前一个窗口，以及相邻 pair /
singleton，用较低成本覆盖常见的中间窗口质量问题。final flush 不使用 recent
限制，会从剩余 pool 中继续搜索直到清空。
默认 `max_padding_ratio=0.05`，这是根据 145 Wikitext benchmark 在 padding
质量和 producer 速度之间取得的折中。

`planner_mode="throughput"` 是显式 opt-in 的吞吐模式。它会给普通迭代的
recent-window 枚举加上 `max_candidate_windows` 上限；受限搜索未命中时，通常
本次 `pop_ready()` 直接返回 `None`，等待后续 records，而不是立刻进入不设上限的代表候选搜索。
但如果连续 miss 达到 `limited_search_fallback_after`，或 planner pool 达到
`limited_search_fallback_pool_size`，会进入 adaptive 偿还路径：前者允许一次不设上限的代表候选
搜索，后者取消本次 threshold search 的 candidate-window 上限。这样既能限制常规
steady-state 调用的 CPU work，又避免所有未解决的候选选择在 final flush 集中爆发。
flush 路径仍然使用不设上限的代表候选搜索，避免尾部样本因为吞吐模式被跳过。

没有 distributed cost window 时，DDP steady state 不允许 rank-local defer：capped
search miss 时立即运行不设上限的代表候选 fallback，保证每个非空 source batch 在每个
rank 都对应一个训练 batch。启用 global matcher 后，iteration 会有意积累一个 K-plan
block，再按全局 cost 分配整块 plan。

`cost_window_batches > 1` 时，iteration 只缓存已经完成 planner 选择的 plan，
按 estimated cost 降序后再进入 final collate/pin queue。它不改变 plan 边界、sample
归属、source-batch collective 次序或 final-flush 规划，也不增加 collective。不同 rank
的局部 cost 分布差异较大时，这个量化排序不能提供全局 balance 保证。

`distributed_cost_window_batches=K` 是 map-style DDP 的显式 opt-in。它要求 `K >= 2`，
与 `cost_window_batches > 1` 互斥，并在 iteration 开始时由所有 rank 无条件校验配置一致；
即使本 rank 配置为 `None` 也必须参加这次校验，避免部分 rank 进入 matcher 而另一些 rank
继续本地路径。非 distributed iteration 显式报错，`IterableDataset` 在构造 loader 时
直接拒绝。

每个 rank 积累 K 个已经完成的 plan 后，matcher 执行以下步骤：

1. `all_gather_object` 只交换 plan 的 `(index, length, arrival_id)`、estimated cost 和
   reason，不交换 sample object。
2. 全局按 estimated cost 降序排序；每 `world_size` 个相邻 plan 组成一个 DDP step。
3. 根据全局 step offset 轮转 plan 到 rank 的分配，避免固定 rank 长期接收同一 cost
   position。
4. 本 rank 原有 plan 直接复用；远端 plan 在接收 rank 主进程重新执行
   `dataset[index]` 和 `len_fn`；有效长度与 metadata 不同则报错，否则进入 final
   collate/pin queue。

完整 block 每 K 个输出 step 增加一次 metadata gather；source 耗尽时的非空 partial
block 也 gather 一次。它不引入 forward barrier 或固定的 per-step collective。final
flush 保持原协议，不进入这个 matcher。为了让 producer 能在 consumer 到达前准备一整个
matched block，建议 `prefetch_batches >= K`。

`adaptive` 的第一优先级是 `max_padding_ratio`。自动模式从默认候选值中的中间档位开始，
普通 steady-state planning 遇到 no-ready 或 fallback plan 超过当前阈值时放宽；连续低
padding batch 达到 patience 后收紧。更新只影响下一次 planner search，不修改 batch
budget、source sampler 或 source cursor。启用 adaptive `distributed_cost_window_batches`
时复用同一条 metadata gather 路径，每个 rank 根据 gather 后的同一份 metadata 计算
source step spread、matched step spread 和 improvement ratio，因此不需要额外 broadcast
决策；iteration 开始时仍会 object-gather 校验 adaptive config 完全一致。

远端 materialization 会重复 source rank 已经完成的 dataset read、decode 和 transform，
而且发生在接收 rank 主进程，无法复用 worker 的 batched `__getitems__`。因此该模式只在
跨 rank compute-duration spread 确实是瓶颈时启用，并同时 benchmark loader wait、ready
queue、总 wall time 和 remote record 数。dataset 必须在所有 rank 上 index-compatible，
且 `dataset[index]` 可确定、无副作用、worker 外可用并保持相同有效长度。iteration
启动时同一轮向量 min/max 还会校验 local/global cost window 和 `max_batches`，不额外
增加配置 collective 次数。

global matching 后，单个 rank 的 padding summary 中 `before` 属于 source-owner samples，
`after` 属于 matcher 分配给该 rank 的 samples，两者不再是严格相同的样本集合。跨 rank
聚合后仍保持全局样本守恒；不要把单 rank 的 padding reduction 当作守恒证明。

145 benchmark 后，当前 planner 的问题不是 sorted pool 和 prefix sum 本身，而是
每次 `pop_ready()` 后反复生成和比较大量候选窗口。当前先使用 recent-window
局部搜索降低 steady-state 成本，用 range-min 降低候选构造成本，并保留不设上限的代表候选
兜底；如果训练侧 loader wait / GPU utilization 和 LBA planner 统计共同表明 producer
仍然跟不上 GPU，再考虑更复杂的候选缓存、长度 bucket 或 aging 策略。

## 为什么不替换默认策略

这一轮尝试过的优化可以分成三类：

- 已纳入默认实现的结构优化：`CandidateIndex` 为 planner 的 sorted pool 提供 prefix
  lengths 和 arrival-id range-min view，减少候选构造时的重复状态传递；日志和 DDP
  flush 也拆成独立 helper，降低 wrapper 和 coordinator 的职责混杂。这些优化不改
  batch 选择语义，因此适合进入 v2 默认。
- 保留为 opt-in 的 throughput 优化：给 recent-window 搜索加上候选窗口上限，能
  限制普通 `pop_ready()` 的 CPU work，但候选范围太窄时会让更多样本滞留到
  final flush，可能增加 batch 数和尾部搜索成本。
- 保留为 opt-in 的 adaptive 偿还：连续 capped-search miss 后允许不设上限的代表候选 fallback，
  或在 pool 足够大时临时取消本次 threshold search 上限。它能明显减少 flush 债务，
  但会把额外搜索工作放回 steady-state，padding 和总耗时也不是稳定胜过 quality。

因此 v2 默认继续使用 quality planner。这个默认值已经是比较稳的实现：padding
质量好、flush 行为清晰、DDP 契约明确，也没有一个轻量改动能在 padding、吞吐和
final flush 三个维度同时稳赢。

## Spill

当内存 sample pool 超过 `max_cache_samples` 时，planner 将最早进入且暂未选中的
样本追加写入磁盘 shard。多次小 overflow 会继续填充当前 shard，默认每个 shard
最多 `10_000` 个样本。shard 使用 pickle 保存完整 sample record，因此触发 spill 的
sample 必须可 pickle，显式 `spill_dir` 也应视为包含训练样本的敏感目录。spill 成功
后，样本从内存 pool 删除。

非 DDP flush 会从多个 shard 惰性读取 records，只补满
`max_cache_samples` 允许的 planner pool，再规划并继续补池；候选因此可以跨 shard
组合，同时内存 pool 不突破配置上限。DDP final flush 的公共规划契约要求收集全部
剩余 records，因此 `drain_records()` 仍会全量加载本 rank 的 spill；indexed 模式
随后只交换 metadata，object 模式则交换完整 sample record。显式
`spill_dir` 下由当前 planner 创建的 shard 会在消费或关闭时清理，避免重复 flush
再次产出已消费样本。

DDP 模式下，如果用户传入共享 `spill_dir`，adapter 会在其下按 rank 创建子目录，
避免不同进程写入相同 shard 文件名。

## 日志

默认日志目录为：

```text
~/.lba/logs/
```

每个 LBA 实例在首次迭代或访问日志属性时惰性创建一对日志；同一实例中实际进入
planner 的 iteration 继续追加各自的 summary，`max_batches=0` 不进入 planner：

- `.log`：给训练时人工扫读，固定输出 padding 改善、planner 开销和健康计数。
- `.jsonl`：给 benchmark 和排障脚本解析，记录完整 summary、spill、oversized
  sample、DDP final flush 等结构化事件。
