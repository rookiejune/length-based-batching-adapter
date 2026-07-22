# LBA Todo

已经拍板或完成的设计、边界情况和 benchmark 结论放在 `docs/`：

- `docs/design.md`
- `docs/usage.md`
- `docs/v2.md`
- `docs/edge_cases.md`
- `docs/benchmark_145.md`

当前 todo 只记录还没落实的工作。

## 后续验证

- 2026-07-23 复旦 145 的 text-file token-work smoke 已经确认：index-only 后
  quality planner 仍能把 padding ratio 从 68.16% 降到 4.01%，candidate window
  checks 可记录，但 loader wait 明显偏高。下一步需要在真实模型训练中同时记录
  token/sec、step/sec、GPU utilization、padding ratio、padded length、planner 时间、
  candidate window checks、loader wait、samples/sec，判断 producer / materialization
  是否仍然喂不满 GPU。
- 在真实模型上拟合并验证 `cost_fn`，同时记录 estimated cost 与
  forward/backward duration 的相关性。
- 2026-07-23 复旦 145 的 text-file token-work smoke 中，
  `distributed_cost_window_batches=2` 将 step compute spread 从 2.39s 降到 0.61s，
  remote records 为 4,382。下一步仍需在真实模型上分别验证 local
  `cost_window_batches` 和 `distributed_cost_window_batches` 对跨 rank
  compute-duration spread 的改善；global matching 还要同时记录 remote read/decode
  数量、loader wait、ready queue、step-start spread 和总 wall time，不能只看 cost
  对齐。

## 非默认 planner 实验

### 1. 长度 bucket / 窗口索引

- 可以尝试按长度分桶或窗口索引，只在相近长度样本中找候选。
- 维护增量候选状态：新增 records 后只更新受影响的 bucket/window。
- 这类策略会改变 batch 选择语义，必须先作为非默认模式实现。

### 2. 控制样本滞留和尾部 flush

- 给长期没被选中的 records 加入 aging 机制，避免极端长度样本一直留在 pool。
- 在真实异构长度分布上对比 spill / no-spill 的 padding ratio、batch count 和
  flush time。
