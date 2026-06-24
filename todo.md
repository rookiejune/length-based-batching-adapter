# LBA Todo

已经拍板或完成的设计、边界情况和 benchmark 结论放在 `docs/`：

- `docs/design.md`
- `docs/usage.md`
- `docs/edge_cases.md`
- `docs/benchmark_145.md`

当前 todo 只记录还没落实的工作。

## Planner 优化主线

默认目标是保持当前较低 padding，而不是让纯 CPU 迭代速度接近 baseline。
`>= 5 steps/s` 的 batch producer 能力已经是一个可接受 baseline；如果真实训练
主要开销在 GPU，默认不应该牺牲 padding ratio 换 planner 吞吐。

下面这些优化可以做，但属于显式 opt-in 的 throughput tradeoff。任何会增大
padding ratio 的策略都不能默认开启，必须在配置名和文档里清楚暴露取舍。

### 1. 可选 throughput planner

- 增加 `planner_mode="quality" | "throughput"` 一类显式开关；默认保持
  `"quality"`。
- 或增加 `max_candidate_windows=None`，默认不限制；用户显式设置后才截断
  fast-path recent-window 枚举。
- throughput 模式可以限制每次 `pop_ready` 检查的候选窗口数量，例如扫描
  `128/256/512` 个候选窗口，换取更低 planner CPU 时间。
- 需要 benchmark 不同窗口上限对 padding ratio、padded length、steps/rank、
  planner 时间和 loader wait 的影响。

### 2. 长度 bucket / 窗口索引

- range-min 后主要成本在 fast-path recent-window 枚举，不在 tail flush。
- 可以尝试按长度分桶或窗口索引，只在相近长度样本中找候选。
- 维护增量候选状态：新增 records 后只更新受影响的 bucket/window。
- 这类策略会改变 batch 选择语义，必须先作为非默认模式实现。

### 3. 控制样本滞留和尾部 flush

- 给长期没被选中的 records 加入 aging 机制，避免极端长度样本一直留在 pool。
- 检查 spill 后的 records 是否会影响候选质量和 flush 成本。
- DDP final flush 仍只作为尾部对齐机制，不把全程样本交换放进公共池。

### 4. Benchmark 回归

- DDP：145 上 2 GPU text-file 20k benchmark。
- 真实训练里如果线程 prefetch 仍然喂不满 GPU，再补更贴近模型计算的 benchmark。
- 对比指标：padding ratio、padded length、planner 时间、candidate window checks、
  loader wait、samples/sec。
- throughput 模式 benchmark 必须额外标明是否启用近似搜索，以及相对默认
  quality 模式增加了多少 padding。

## 暂不做

- 不做 per-worker planner，除非先设计清楚 worker 结束时的 flush 协议。
- 不做进程版 producer，除非真实训练里线程 prefetch 仍然喂不满 GPU。
- 不把 DDP 公共池扩展成全程跨 rank 样本调度；当前只解决最后 flush 对齐。
