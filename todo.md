# LBA Todo

已经拍板或完成的设计、边界情况和 benchmark 结论放在 `docs/`：

- `docs/design.md`
- `docs/usage.md`
- `docs/edge_cases.md`
- `docs/benchmark_145.md`

当前 todo 只记录还没落实的工作。

## Planner 优化主线

目标是降低 planner 的 CPU 搜索成本，同时保持当前较低 padding。不要把 batch
数量减少当成优化目标；batch 变多通常说明原始 fixed batch 的长度编排不合理。

### 1. 继续降低候选搜索成本

- 用 benchmark 观察 recent-window fast path 后的 `candidate_window_checks` 和
  `pop_ready_time_seconds` 是否仍然过高。
- 如果局部窗口仍不够，优先尝试按长度分桶或窗口索引，只在相近长度样本中找候选。
- 维护增量候选状态：新增 records 后只更新受影响的 bucket/window。
- 保留简单 fallback：当增量候选拿不到合适 batch 时，仍走一次完整搜索。

### 2. 控制样本滞留和尾部 flush

- 给长期没被选中的 records 加入 aging 机制，避免极端长度样本一直留在 pool。
- 检查 spill 后的 records 是否会影响候选质量和 flush 成本。
- DDP final flush 仍只作为尾部对齐机制，不把全程样本交换放进公共池。

### 3. Benchmark 回归

- DDP：145 上 2 GPU text-file 20k benchmark。
- 真实训练里如果线程 prefetch 仍然喂不满 GPU，再补更贴近模型计算的 benchmark。
- 对比指标：padding ratio、padded length、planner 时间、candidate window checks、
  loader wait、samples/sec。

## 暂不做

- 不做 per-worker planner，除非先设计清楚 worker 结束时的 flush 协议。
- 不做进程版 producer，除非真实训练里线程 prefetch 仍然喂不满 GPU。
- 不把 DDP 公共池扩展成全程跨 rank 样本调度；当前只解决最后 flush 对齐。
