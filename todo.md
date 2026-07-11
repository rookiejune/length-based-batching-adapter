# LBA Todo

已经拍板或完成的设计、边界情况和 benchmark 结论放在 `docs/`：

- `docs/design.md`
- `docs/usage.md`
- `docs/v1.md`
- `docs/edge_cases.md`
- `docs/benchmark_145.md`

当前 todo 只记录还没落实的工作。

## 后续验证

- 真实训练里如果日志显示 producer 仍然喂不满 GPU，再补贴近模型计算的 benchmark。
- 2026-07-11 quality planner 从隐式“完整搜索”表述改为不设上限的代表候选搜索；
  后续真实训练 benchmark 需要同时观察 padding 质量和 candidate window checks。
- 若要评估端到端训练吞吐，优先记录真实模型的 token/sec、step/sec、GPU utilization、
  padding ratio、padded length、planner 时间、candidate window checks、loader wait、
  samples/sec。

## 非默认 planner 实验

### 1. 长度 bucket / 窗口索引

- 2026-07-08 优化后，默认 quality planner 的主要内部成本已经压到
  recent-window 枚举和索引刷新；继续优化需要改变搜索策略。
- 可以尝试按长度分桶或窗口索引，只在相近长度样本中找候选。
- 维护增量候选状态：新增 records 后只更新受影响的 bucket/window。
- 这类策略会改变 batch 选择语义，必须先作为非默认模式实现。

### 2. 控制样本滞留和尾部 flush

- 给长期没被选中的 records 加入 aging 机制，避免极端长度样本一直留在 pool。
- 检查 spill 后的 records 是否会影响候选质量和 flush 成本。
- DDP final flush 仍只作为尾部对齐机制，不把全程样本交换放进公共池。

## 暂不做

- 不做 per-worker planner，除非先设计清楚 worker 结束时的 flush 协议。
- 不做进程版 producer，除非真实训练里线程 prefetch 仍然喂不满 GPU。
- 不把 DDP 公共池扩展成全程跨 rank 样本调度；当前只解决最后 flush 对齐。
