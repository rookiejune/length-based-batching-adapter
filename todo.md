# LBA Todo

已经拍板或完成的设计、边界情况和 benchmark 结论放在 `docs/`：

- `docs/design.md`
- `docs/usage.md`
- `docs/v2.md`
- `docs/edge_cases.md`
- `docs/benchmark_145.md`

当前 todo 只记录还没落实的工作。

## 后续验证

- 真实训练侧的 loader wait / GPU utilization 与 LBA planner 统计如果共同表明 producer
  仍然喂不满 GPU，再补贴近模型计算的 benchmark。
- 后续真实训练 benchmark 需要同时观察 quality planner 的 padding 质量和
  candidate window checks。
- 若要评估端到端训练吞吐，优先记录真实模型的 token/sec、step/sec、GPU utilization、
  padding ratio、padded length、planner 时间、candidate window checks、loader wait、
  samples/sec。
- 在真实模型上拟合并验证 `cost_fn`，同时记录 estimated cost 与
  forward/backward duration 的相关性。
- 验证 `cost_window_batches` 对跨 rank compute-duration spread 的改善；如果各 rank
  cost 分布本身差异过大，再设计 block-level global matching，不能把局部排序当作全局保证。

## DDP final flush 契约

- 设计显式的 index metadata / object gather 选择。当前 map-style dataset 默认按
  index 重取 final-flush sample，要求 `dataset[index]` 可在主进程确定性重放；随机、
  worker-sensitive 或有副作用的 dataset 需要先改为稳定输入，或改成明确自行分片的
  `IterableDataset` 以使用 object-gather final flush。
- 如果新增公共选项，默认值需要同时权衡原 sample 守恒和大 sample object gather 的
  通信、内存成本，不能静默猜测 dataset 是否可重放。

## 非默认 planner 实验

### 1. 长度 bucket / 窗口索引

- 可以尝试按长度分桶或窗口索引，只在相近长度样本中找候选。
- 维护增量候选状态：新增 records 后只更新受影响的 bucket/window。
- 这类策略会改变 batch 选择语义，必须先作为非默认模式实现。

### 2. 控制样本滞留和尾部 flush

- 给长期没被选中的 records 加入 aging 机制，避免极端长度样本一直留在 pool。
- 在真实异构长度分布上对比 spill / no-spill 的 padding ratio、batch count 和
  flush time。
