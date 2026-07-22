# 145 Benchmark 记录

## 当前结论

当前 v2 基线是 [2026-07-20 v2 真实 DDP 复测](#2026-07-20-v2-真实-ddp-复测)：
2-GPU Wikitext quality LBA 把 padded length 从 `5,548,720` 降到 `1,826,009`，
padding ratio 从 `68.24%` 降到 `3.50%`。在 `compute_iters=4`、
`simulate_step_sec=0` 的轻量 DDP 模型 step 中，LBA 中位耗时为 `2.308s`，baseline
为 `0.934s`。每个 repeat 的样本与 raw length 守恒，训练默认 group 为 NCCL，LBA
metadata group 为 Gloo。这些结果验证 v2 的分布式契约、padding 和 planner 成本，不
证明真实模型端到端训练吞吐提高。

[2026-07-19 远程完整复测](#2026-07-19-远程完整复测) 仍提供 quality 与
throughput-256 的同参数历史对照；throughput-256 当时减少约 `8.7%` candidate checks，
但没有稳定 wall-time 优势。

## 环境说明

本文记录跨越 2026-06-19 到 2026-07-20，包含本地 microbenchmark、125/144 环境
排查和 145 远程复测。各节列出的环境覆盖这里的概览；不能把单一机器配置套到全文。

- 最新 v2 复测：`145.pami.group`，Python 3.12.0，PyTorch `2.9.0+cu128`，GPU 0、3
  两张 RTX 4090 D，package `2.0.0` 的本地未提交工作树。
- 2026-07-19 v1 复测：同机 GPU 5、6，package `1.0.0` / commit `06521f7`。
- 历史代码目录包括 `~/lba_benchmark_run/lba`、`~/repos/lba` 和隔离 debug checkout，
  只用于定位当时结果，不是当前安装路径约定。
- 原始 CSV 保存在 workspace 顶层 `debug/lba-review/remote-145/`，不随本仓库提交；
  文档中的表格是持久化结果记录。

## 数据集

端到端 DataLoader benchmark 使用原始 `batch_size=32`。LBA 产出的 batch size 是
动态的，但 `max_padded_length` 的默认推断基于这个原始 batch size。独立 planner 和
candidate microbenchmark 不适用这项约定，以各节参数为准。

### Synthetic

脚本内置的 lognormal 长度分布文本数据，用于观察 planner 本身的 CPU 开销和 padding 改善幅度。

### Wikitext-103

145 上已有 HuggingFace 缓存：

```text
~/.cache/huggingface/hub/datasets--Salesforce--wikitext/
```

从 Parquet 缓存中抽取了 200k 行非空文本：

```text
~/lba_benchmark_run/lba/outputs/datasets/wikitext103_train_200k.txt
```

benchmark 使用 `TextLineDataset` 按 offset 读取文本行，以便覆盖真实文本 IO。
脚本的 `--dataset hf` 模式会在运行时导入 Hugging Face `datasets`；该依赖不属于
LBA 核心安装，使用前需要单独安装。`synthetic` 和 `text-file` 模式不需要它。

## 指标

两条 benchmark 入口共同记录：

- `elapsed_sec`：完整迭代耗时。
- `time_to_first_batch_sec`：首个 batch 产出耗时。
- `raw_length_sum`：样本真实长度之和。
- `padded_length_sum`：batch padding 后总长度，计算为 `max_length * batch_size`。
- `padding_length_sum`：`padded_length_sum - raw_length_sum`。
- `padding_ratio`：`padding_length_sum / padded_length_sum`。
- `repeat_index` / `run_position`：当前 measured run 的重复编号和执行位置。
- effective planner config：实际解析后的 candidate window / fallback 上限和最终
  `max_padded_length`，不使用未解析的 CLI `None` 代替默认值。
- spill health：`planner_max_cache_size`、`planner_spill_events` 和
  `planner_spilled_records`。

单进程 `benchmark_lba.py` 另外记录：

- `loader_wait_sec` / `loader_wait_per_batch_sec`：消费端等待 `next(loader)` 的总时间
  和 batch 均值。
- `simulated_gpu_sec`：每个 batch 后模拟 GPU 消费的 sleep 时间。
- `samples_per_sec`：按完整 elapsed 计算的 sample 吞吐。

DDP `ddp_benchmark.py` 另外记录：

- `loader_wait_sec_sum` / `step_compute_sec_sum`：所有 rank 的 loader wait 和 step
  compute 累计时间。
- `simulate_step_sec` / `compute_iters`：固定 sleep 和 token-work model 的消费参数。
- `step_compute_sec_min/max/spread`、`rank_compute_iters_min/max/spread` 和
  `rank_step_delay_sec_min/max/spread`：rank-imbalance benchmark 的实际 step compute
  差异和每 rank 消费配置差异。
- `steps_per_rank`、`samples_per_sec`、`raw_tokens_per_sec` 和
  `padded_tokens_per_sec`：按最慢 rank 的 elapsed 计算的 DDP 结果。

当前脚本支持 `--repeats`、`--warmup-runs` 和 `--run-order alternate`。新的性能
对照建议至少使用一次 warmup 和四次 measured repeat，让 baseline / LBA 各先跑两次，
再按各自多次结果的中位数比较：

```bash
PYTHONPATH=src python benchmarks/benchmark_lba.py \
  --dataset text-file \
  --text-file /path/to/data.txt \
  --repeats 4 \
  --warmup-runs 1 \
  --run-order alternate
```

脚本会逐 repeat 校验 baseline 与 LBA 的 sample count 和 raw length 一致。DDP
benchmark 默认设置 `drop_last_flush=False`，无法守恒时直接失败；只有实验明确允许
尾部丢样本时才传 `--drop-last-flush`，此时 warning 不会被 benchmark 屏蔽。

标记为 2026-06-19 至 2026-07-08 的历史性能结果大多是在该重复测量协议加入前记录的
单次 run，保留为当时实现的回归记录；新的小幅性能结论应使用上述协议复测。

## 本地 Spill 回归

2026-07-19 在本地 `py39` / PyTorch 2.8 对 spill shard 聚合和 bounded flush 做了
固定 synthetic 回归。输入为 2,000 条长度均为 32 的 records，每次
`add_records()` 新增 1 条；`max_cache_samples=32`、`max_padded_length=1024`、
`max_padding_ratio=0.0`，默认 `shard_size=10_000`，使用显式临时 spill 目录。

| implementation | shards | batches | samples | add | flush | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| legacy per-write shard / per-shard flush | 1,968 | 1,969 | 2,000 | 0.1422s | 0.0916s | 0.2337s |
| current aggregated shard / bounded cross-shard flush | 1 | 63 | 2,000 | 0.0560s | 0.0045s | 0.0605s |

这组结果验证了同长输入下的 shard 聚合、样本守恒、bounded pool 和 batch step
爆炸修复。它不证明真实异构长度分布下 spill 与 no-spill 的候选质量完全一致；后者
仍需同时比较 padding ratio、batch count 和 flush time。

## 本地 Candidate 常数项回归

2026-07-19 在同一套本地 `py39` / PyTorch 2.8 环境对不改变候选集合的内部优化做了
隔离 microbenchmark：

| path | before | after | invariant |
| --- | ---: | ---: | --- |
| 8,192 个等长 records 的 threshold tie | 0.3932s | 0.0148s | 24,573 checks，winner `(0, 4095)` |
| 8,192 pool、末尾 32 个 recent 的枚举 | 3.556ms | 0.266ms | 96 个窗口完全一致 |
| 代表窗口枚举 | 7.089ms | 4.144ms | 26,746 个窗口及顺序一致 |
| 2,000 pool + 32 new，重复构造 5,000 次 | 1.1999s | 0.5465s | `(length, arrival_id)` 排序结果一致 |

前两项分别来自 tie 时按窗口扫描 arrival id 改为 lazy range-min，以及跳过不可能覆盖
recent records 的前后区间；第三项移除了每个 end 上的临时 set 和嵌套 generator；
最后一项使用 Timsort 处理已有 sorted run 和新增 records，替代 Python 层
`heapq.merge` materialize。这些数字只说明内部常数项，不代替真实训练的端到端
token/sec、step/sec 和 GPU utilization 复测。

## 2026-06-19 初始实现单次结果

| dataset | mode | samples | workers | elapsed | padded length | padding ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| synthetic | baseline | 20k | 0 | 0.030s | 4,686,560 | 82.75% |
| synthetic | LBA | 20k | 0 | 12.10s | 868,573 | 6.94% |
| wikitext | baseline | 20k | 0 | 0.117s | 5,300,032 | 66.75% |
| wikitext | LBA | 20k | 0 | 14.94s | 1,832,729 | 3.85% |
| wikitext | baseline | 20k | 4 | 0.214s | 5,300,032 | 66.75% |
| wikitext | LBA | 20k | 4 | 15.09s | 1,832,729 | 3.85% |
| wikitext | baseline | 50k | 4 | 0.393s | 13,284,848 | 67.36% |
| wikitext | LBA | 50k | 4 | 65.42s | 4,506,741 | 3.80% |

### 当时结论

LBA 对 padding 的改善非常明显。Wikitext 上，padded length 大约减少 65% 到 66%，
padding ratio 从约 67% 降到约 3.8%。

当时的瓶颈不是 IO。Wikitext 20k 下，`num_workers=0` 和 `num_workers=4` 的 LBA
耗时几乎一样，说明多进程读取不是限制，主进程 planner 才是主要瓶颈。

当时实现不适合继续直接放大数据规模。50k Wikitext 已经需要约 65 秒，因此后续先
优化 planner，再做更大规模 benchmark。

### 当时对 Planner 的启示

- 不能在每个 batch 后做高成本全局候选搜索。
- 需要让 planner 的候选维护接近增量式，而不是反复扫描整个 pool。
- `max_padding_ratio` 快速提交路径是必要的，但还不够。
- 需要讨论是否引入长度 bucket、局部窗口索引、候选缓存或更强的早停规则。
- benchmark 暂时应保留 20k/50k 规模，作为 planner 优化前后的回归数据。

## Prefetch Producer 测试

2026-06-19 在 145 上同步当前代码后，使用 `prefetch_batches` 做了一组测试。结果
当时备份在本地 `lba/outputs/remote_145/`；该路径是历史 checkout 内产物，不随当前
仓库发布。

### Wikitext 20k, simulated GPU 0.02s

这个设置相当于消费端目标约 50 it/s，比第一阶段 `>= 5 it/s` 更严格。

| prefetch | batches | elapsed | loader wait | wait / batch | padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 634 | 31.75s | 19.01s | 29.99ms | 7.04% |
| 4 | 634 | 19.02s | 3.40s | 5.36ms | 7.04% |

### Wikitext 2k, simulated GPU 0.2s

这个设置对应第一阶段约 5 it/s 的目标。

| prefetch | batches | elapsed | loader wait | wait / batch | padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 77 | 17.13s | 1.71s | 22.24ms | 3.57% |
| 4 | 77 | 15.52s | 0.11s | 1.37ms | 3.57% |

### 结论

`prefetch_batches=4` 对训练式消费场景有效。对于 5 it/s 目标，平均 loader wait
约 1.37ms，CPU producer 基本可以被 GPU 消费时间覆盖。

使用 `max_padding_ratio=0.1` 对照复跑时，LBA 为 634 batches、17.82s、padding
ratio 约 7.04%。这和旧记录中的 941 batches、3.85% 不一致，说明较宽松阈值会让
planner 更倾向产出更大的次优 batch。

### Planner 阈值原因

这个差异不是 prefetch 引起的。对照中的 planner 快速提交路径使用
`max_padding_ratio=0.1`，并在满足阈值的候选中优先选择更大的
`padded_length`。因此 planner 会倾向于把 batch 塞得更满，只要候选 padding
ratio 不超过 10%，就可能接受一个不是最低 padding 的窗口。

对照测试：

| setting | batches | elapsed | padded length | padding ratio |
| --- | ---: | ---: | ---: | ---: |
| 2026-06-19 code, `max_padding_ratio=0.1` | 634 | 17.82s | 1,895,454 | 7.04% |
| `max_padding_ratio=0.075` | 646 | 20.13s | 1,866,612 | 5.60% |
| `max_padding_ratio=0.05` | 666 | 25.11s | 1,833,619 | 3.90% |

`max_padding_ratio=0.05` 的 padded length 和旧记录 `1,832,729` 非常接近，
说明旧结果更像是更严格的快速提交阈值或更偏向低 padding 的候选选择策略。默认
`max_padding_ratio` 因此采用 `0.05`。

### `max_padding_ratio=0.05` 队列压力

`max_padding_ratio=0.05` 时，Wikitext 20k no-sim producer 速度约为
`666 / 25.11s = 26.5 it/s`，因此第一阶段 `>= 5 it/s` 目标足够。

继续用 `prefetch_batches=4` 做消费压力测试：

| simulated GPU | target it/s | elapsed | loader wait | wait / batch | padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.05s | 20 it/s | 34.47s | 0.50s | 0.75ms | 3.90% |
| 0.02s | 50 it/s | 25.98s | 9.54s | 14.33ms | 3.90% |

结论：`max_padding_ratio=0.05` 对 5 it/s 和 20 it/s 的消费速度都够用；到
50 it/s 时队列开始明显等 producer，实际吞吐回落到约 25.6 it/s。

## Recent-window Planner 回归

2026-06-20 在 145 上对比优化前的 `~/repos/lba` 和优化后的
`~/repos/lba_planner_opt`。两边都使用同一个 Wikitext text-file 缓存、`batch_size=32`、
`num_workers=4`、`max_padded_length=4096`、`max_padding_ratio=0.05`。

### 单进程 Wikitext

| setting | code | batches | elapsed | loader wait | padded length | padding ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20k, prefetch 0, no sim | before | 697 | 29.89s | 29.88s | 1,827,074 | 3.5569% |
| 20k, prefetch 0, no sim | after | 696 | 20.61s | 20.61s | 1,827,083 | 3.5574% |
| 50k, prefetch 0, no sim | before | 1,642 | 98.00s | 97.99s | 4,497,447 | 3.5990% |
| 50k, prefetch 0, no sim | after | 1,642 | 58.07s | 58.07s | 4,497,454 | 3.5991% |
| 20k, prefetch 4, sim 0.05s | before | 697 | 38.08s | 1.57s | 1,827,074 | 3.5569% |
| 20k, prefetch 4, sim 0.05s | after | 696 | 35.37s | 0.20s | 1,827,083 | 3.5574% |

结论：

- no-sim 20k 从 29.89s 降到 20.61s，约 1.45x；50k 从 98.00s 降到
  58.07s，约 1.69x。
- padding 基本不变，20k padded length 只增加 9，50k 只增加 7。
- prefetch + simulated GPU 场景下，总 elapsed 接近 `batch_count * 0.05s` 的消费
  下限；loader wait 从 1.57s 降到 0.20s。
- after 版本 20k no-sim 的 planner 字段为：`pop_ready_time_seconds=20.04s`、
  `candidate_window_checks=1,010,174`、`fast_path_batches=625`、
  `flush_search_batches=71`。50k no-sim 为：`pop_ready_time_seconds=56.86s`、
  `candidate_window_checks=2,491,895`、`fast_path_batches=1563`、
  `flush_search_batches=79`。

### DDP 2GPU Smoke

2 GPU、Wikitext 2k、`simulate_step_sec=0.2`、`compute_iters=0`：

| code | LBA elapsed | loader wait sum | steps/rank | padded length | padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| before | 14.73s | 2.03s | 68 | 198,756 | 2.9831% |
| after | 14.79s | 2.15s | 68 | 198,756 | 2.9831% |

DDP smoke 中总耗时主要由 68 个 simulated steps 决定；优化后步数、padding 和 final
flush 对齐行为保持一致。

## Range-min Candidate 回归

2026-06-20 在 145 上继续对比 `~/repos/lba_planner_opt` 和
`~/repos/lba_planner_rangemin`。后者在候选构造时用 arrival-id range-min 索引替代
逐窗口扫描 `min(arrival_id)`。

### 单进程 Wikitext

| setting | code | batches | elapsed | loader wait | pop ready | padded length | padding ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20k, prefetch 0, no sim | recent-window | 696 | 20.61s | 20.61s | 20.04s | 1,827,083 | 3.5574% |
| 20k, prefetch 0, no sim | range-min | 696 | 7.33s | 7.33s | 6.79s | 1,827,083 | 3.5574% |
| 50k, prefetch 0, no sim | recent-window | 1,642 | 58.07s | 58.07s | 56.86s | 4,497,454 | 3.5991% |
| 50k, prefetch 0, no sim | range-min | 1,642 | 18.90s | 18.90s | 17.67s | 4,497,454 | 3.5991% |
| 20k, prefetch 4, sim 0.05s | recent-window | 696 | 35.37s | 0.20s | 21.31s | 1,827,083 | 3.5574% |
| 20k, prefetch 4, sim 0.05s | range-min | 696 | 35.06s | 0.10s | 13.37s | 1,827,083 | 3.5574% |

结论：

- no-sim 20k 从 recent-window 的 20.61s 降到 7.33s，约 2.81x；50k 从
  58.07s 降到 18.90s，约 3.07x。
- 相比优化前最初版本，20k 从 29.89s 降到 7.33s，约 4.08x；50k 从 98.00s
  降到 18.90s，约 5.18x。
- padding 完全保持不变，说明 range-min 只优化候选构造成本，没有改变 batch
  选择语义。
- range-min 20k no-sim 的 source split：fast path 6.17s / 860,608 checks，
  flush 0.61s / 149,566 checks。50k no-sim：fast path 16.93s /
  2,316,862 checks，flush 0.74s / 175,033 checks。

range-min 后，flush 已不是主要成本；下一步应优先减少 fast-path recent-window
枚举的候选数量，例如按长度 bucket 或更窄的局部窗口索引。

## DDP 真实文本测试

DDP benchmark 已支持 `text-file` 数据源，可以直接复用 145 上落盘的 Wikitext
文本缓存，避免多进程 benchmark 每次重复走 HuggingFace dataset 构建。

2026-06-21 使用 range-min planner 在 145 上补跑 2GPU Wikitext 20k，
`compute_iters=0`，扫描不同 `simulate_step_sec`。LBA 的 step 数为 347/rank，
baseline 为 313/rank。

| simulate step | mode | elapsed | loader wait sum | step compute sum | padded length | padding ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.00s | baseline | 0.71s | 0.38s | 0.54s | 5,548,720 | 68.24% |
| 0.00s | LBA | 4.60s | 8.29s | 0.89s | 1,825,889 | 3.49% |
| 0.02s | baseline | 7.31s | 0.68s | 13.48s | 5,548,720 | 68.24% |
| 0.02s | LBA | 14.20s | 12.73s | 15.66s | 1,825,889 | 3.49% |
| 0.05s | baseline | 16.75s | 0.75s | 32.30s | 5,548,720 | 68.24% |
| 0.05s | LBA | 26.22s | 15.88s | 36.56s | 1,825,889 | 3.49% |
| 0.20s | baseline | 63.73s | 0.77s | 126.21s | 5,548,720 | 68.24% |
| 0.20s | LBA | 78.40s | 16.06s | 140.74s | 1,825,889 | 3.49% |

LBA planner source split 在这组中稳定为 fast path 626 次、807,626 candidate checks，
没有 full-search 或 flush-search 记录；`planner_pop_ready_time_seconds` 在
sim=0/0.02/0.05/0.20 下分别约为 5.52s、9.49s、12.47s、12.48s。

结论：

- DDP 20k 下，fast-path candidate 数量优化会改善 loader wait，尤其
  `simulate_step_sec <= 0.05` 的低/中 step 时间场景。
- 即使 `simulate_step_sec=0.20`，LBA 仍有约 16s 的 loader wait sum，但 wall time
  还同时受 LBA 产生更多 steps 影响。减少 planner wait 不是 DDP 总耗时的唯一杠杆。
- 这组没有 final flush 搜索成本，说明下一步仍应优先减少 steady-state fast-path
  候选枚举。

参考命令：

```bash
/home/zhuyin/anaconda3/envs/py312/bin/torchrun --nproc_per_node=2 \
  benchmarks/ddp_benchmark.py \
  --dataset text-file \
  --text-file /home/zhuyin/lba_benchmark_run/lba/outputs/datasets/wikitext103_train_200k.txt \
  --size 2000 \
  --batch-size 32 \
  --num-workers 4 \
  --max-padded-length 4096 \
  --max-padding-ratio 0.05 \
  --compute-iters 0 \
  --simulate-step-sec 0.2 \
  --output outputs/ddp_benchmark_2gpu_wikitext2k_textfile_sim02_mpl4096_mpr05.csv
```

`simulate-step-sec=0.2` 对应约 5 it/s 的训练消费速度。

| mode | samples | batches | steps/rank | elapsed | loader wait sum | padded length | padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 2,000 | 64 | 32 | 6.75s | 0.08s | 576,160 | 66.53% |
| LBA | 2,000 | 136 | 68 | 14.76s | 2.06s | 198,756 | 2.98% |

这个结果说明：在 5 it/s 的固定 batch 消费模型下，LBA 的 padded length 下降约
65.5%，但因为严格 padding 阈值会切出更多 batch，总步数从每 rank 32 步增加到
68 步，elapsed 也随之增加。这个 benchmark 主要用于确认 DDP 步数对齐、loader
等待和 padding 改善；真实训练吞吐还需要结合模型的 token 计算成本和梯度累积策略
一起看。

## DDP planner 性能优化复测

2026-07-08 在 145 上使用 2 张 4090D 复测 Wikitext text-file DDP benchmark。
GPU0 当时被其他进程占用，最终复测使用 `CUDA_VISIBLE_DEVICES=1,2`。数据集为
`/home/zhuyin/lba_benchmark_run/lba/outputs/datasets/wikitext103_train_200k.txt`，
`size=20000`、`batch_size=32`、`num_workers=4`、
`max_padded_length=4096`、`max_padding_ratio=0.05`。

该轮优化当时不改变 planner 策略：quality 模式仍检查不设上限的代表 recent candidate 集合，
throughput 模式仍保留原来的 limited-window 顺序。改动只减少内部候选枚举和
threshold fast path 的候选构造成本：

- unlimited recent candidate 从按 recent index 重复扫描，改为按 end index 单次枚举。
- threshold 搜索先比较可由 prefix length 直接算出的 padding key，只对最终胜出窗口构造
  `BatchCandidate`。
- final winner 的 `earliest_arrival_id` 使用窗口内扫描，避免常见 fast path 为每个候选都走
  range-min 查询。
- recent-prefix 和候选枚举的几个热点常数项做了小幅整理。

### 20k / no simulated step

同一条 benchmark 的 LBA quality 结果从最初的 4.89s 降到最终 1.61s；
planner `pop_ready` 从 5.68s 降到 1.04s。candidate window checks 没变，
说明优化保持了搜索集合，只降低了每个候选的内部成本。

| revision | mode | elapsed | loader wait sum | planner pop_ready | candidate checks | padded length | padding ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pre-opt | baseline | 0.72s | 0.44s | 0.00s | 0 | 5,548,720 | 68.24% |
| pre-opt | LBA quality | 4.89s | 8.79s | 5.68s | 807,626 | 1,825,889 | 3.49% |
| recent-prefix | LBA quality | 4.13s | 7.41s | 4.36s | 807,626 | 1,825,889 | 3.49% |
| lazy candidate | LBA quality | 1.97s | 3.34s | 1.66s | 807,626 | 1,825,889 | 3.49% |
| scanned winner | LBA quality | 1.80s | 3.05s | 1.32s | 807,626 | 1,825,889 | 3.49% |
| final | LBA quality | 1.61s | 2.69s | 1.04s | 807,626 | 1,825,889 | 3.49% |
| final | LBA throughput-64 | 1.81s | 3.13s | 1.28s | 787,239 | 1,826,612 | 3.53% |

### 20k / simulated step 0.2s

优化后再跑固定 0.2s/step 的 DDP 消费模型。此时 planner 成本已经明显下降，
但 wall time 仍慢于 baseline，主要原因是严格 padding 阈值让 LBA 产生更多 step：
baseline 为 313 steps/rank，LBA 为 347 steps/rank。

| mode | elapsed | steps/rank | loader wait sum | step compute sum | planner pop_ready | padded length | padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 63.76s | 313 | 0.78s | 126.25s | 0.00s | 5,548,720 | 68.24% |
| LBA quality | 72.82s | 347 | 5.60s | 140.00s | 2.96s | 1,825,889 | 3.49% |

结论：

- DDP smoke 和真实 text-file DDP benchmark 已在 145 上跑通。
- 当时默认的 quality planner 主要内部热点已经从每候选 range-min / candidate 构造，压回到
  recent-window 枚举和每批索引刷新。
- 继续做纯内部常数优化的收益已经变小；更大的吞吐改善需要改变 planner 策略，例如
  非默认长度 bucket/window index，或在训练侧通过 token budget / 梯度累积抵消更多 step。
- 在固定 per-step 成本模型里，LBA 仍会因为 step 数更多而变慢；在真实模型里是否更快取决于
  token 计算成本能否从 padded length 降低中收益。

参考命令：

```bash
CUDA_VISIBLE_DEVICES=1,2 \
PYTHONPATH=src \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
/home/zhuyin/anaconda3/envs/py312/bin/torchrun --standalone --nproc_per_node=2 \
  benchmarks/ddp_benchmark.py \
  --dataset text-file \
  --text-file /home/zhuyin/lba_benchmark_run/lba/outputs/datasets/wikitext103_train_200k.txt \
  --size 20000 \
  --batch-size 32 \
  --num-workers 4 \
  --max-padded-length 4096 \
  --max-padding-ratio 0.05 \
  --compute-iters 0 \
  --simulate-step-sec 0.0 \
  --output outputs/bench_20260708/ddp_2gpu_wikitext20k_sim00_quality_final.csv
```

## DDP 远程 Synthetic 复测

2026-06-25 使用当前 DDP 修复后的代码补跑 2GPU 真实远程 benchmark。144 上
`py312` / PyTorch `2.9.0+cu129` 的最小 NCCL DDP 会 `SIGSEGV`，因此切到
125 的 2 张 RTX 3090。125 上默认 NCCL 路径也不稳定，显式设置
`NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1` 后，最小 DDP、LBA smoke 和
`ddp_benchmark.py` 都能跑完。

本次没有在本地、144、125 找到旧的 Wikitext text-file 缓存，因此先使用脚本内置
synthetic lognormal 长尾长度分布。baseline 的 82% padding 来自长尾分布和
`batch_size=32` 下按 batch 最大长度 padding，不代表真实文本一定达到这个比例；
之前 Wikitext 20k 的 baseline padding 约为 68%。

结果文件已备份到本地：

```text
outputs/ddp_benchmark_2gpu_synthetic20k_sim002.csv
outputs/ddp_benchmark_2gpu_synthetic20k_sim005.csv
```

| simulate step | mode | elapsed | steps/rank | loader wait sum | padded length | padding ratio |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 0.02s | baseline | 7.56s | 313 | 0.82s | 4,487,712 | 82.13% |
| 0.02s | LBA | 17.34s | 351 | 17.89s | 817,804 | 1.94% |
| 0.05s | baseline | 16.91s | 313 | 0.79s | 4,487,712 | 82.13% |
| 0.05s | LBA | 28.85s | 351 | 20.43s | 817,804 | 1.94% |

LBA 在 synthetic 长尾数据上把 padded length 从 4.49M 降到 0.82M，padding
ratio 从 82.13% 降到 1.94%，说明长度聚类有效。但当前 wall time 仍明显慢于
baseline，主要原因是 LBA 产生更多 step，并且 planner/loader wait 仍较高：
`simulate_step_sec=0.02` 时 LBA 的 `planner_pop_ready_time_seconds` 为 14.12s，
`candidate_window_checks` 为 861,886；`simulate_step_sec=0.05` 时分别为 16.41s
和 861,886。下一步仍应优先减少 steady-state fast-path 候选枚举。

参考命令：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PYTHONPATH=src \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
~/miniconda3/envs/py312/bin/torchrun --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=29569 \
  benchmarks/ddp_benchmark.py \
  --dataset synthetic \
  --size 20000 \
  --seed 123 \
  --max-length 512 \
  --batch-size 32 \
  --num-workers 4 \
  --max-padded-length 4096 \
  --max-padding-ratio 0.05 \
  --compute-iters 0 \
  --simulate-step-sec 0.02 \
  --output outputs/ddp_benchmark_2gpu_synthetic20k_sim002.csv
```

## 2026-07-19 远程完整复测

本轮在 145 的 GPU 5、6 上使用隔离代码目录
`/home/zhuyin/debug/lba-20260719-review`，没有修改共享 checkout。环境为 Python
3.12.0、PyTorch 2.9.0+cu128 和 2 张 RTX 4090 D。默认 NCCL 路径直接通过，未设置
`NCCL_IB_DISABLE` 或 `NCCL_P2P_DISABLE`。

功能验证结果：

- 远端完整测试：`96 passed, 311 subtests passed`。
- NCCL smoke：两个 rank 均完成 3 steps，同时成功建立 Gloo metadata group。
- throughput `max_candidate_windows=1`：64 个 synthetic samples 全部守恒，两个
  rank 均为 17 steps，`planner_no_ready_calls=0`。
- CUDA pin-memory：32 个 samples 全部守恒，最终 8 个 CPU tensors 全部 pinned，
  `non_blocking=True` CUDA 传输和 synchronize 通过。
- spawn + persistent worker：两轮 iteration 复用同一个 worker 的测试通过。

### 重复 benchmark

以下均为 2 GPU、`size=20000`、`batch_size=32`、`num_workers=4`、
`max_padded_length=4096`、`max_padding_ratio=0.05`、`compute_iters=0`、
`simulate_step_sec=0`。每组先 warmup 1 次，再 measured 4 次并交替运行顺序；表中是
4 次中位数。每个 repeat 都校验 baseline/LBA 的 sample count 和 raw length 相同。

| dataset / planner | mode | elapsed | loader wait sum | samples/s | raw tokens/s | steps/rank | padded length | padding ratio | planner pop_ready | candidate checks |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| synthetic / quality | baseline | 0.644s | 0.356s | 31,069 | 1,245,713 | 313 | 4,487,712 | 82.13% | 0.000s | 0 |
| synthetic / quality | LBA | 1.822s | 2.769s | 10,976 | 440,096 | 344 | 821,615 | 2.40% | 1.551s | 1,339,418 |
| Wikitext / quality | baseline | 0.629s | 0.363s | 31,784 | 2,800,343 | 313 | 5,548,720 | 68.24% | 0.000s | 0 |
| Wikitext / quality | LBA | 2.079s | 3.263s | 9,619 | 847,468 | 346 | 1,826,009 | 3.50% | 1.681s | 1,308,198 |
| Wikitext / throughput-256 | baseline | 0.642s | 0.372s | 31,176 | 2,746,701 | 313 | 5,548,720 | 68.24% | 0.000s | 0 |
| Wikitext / throughput-256 | LBA | 2.064s | 3.225s | 9,691 | 853,859 | 346 | 1,823,899 | 3.39% | 1.642s | 1,194,136 |

Synthetic 每个 repeat 的 `raw_length_sum` 都是 801,898；Wikitext 每个 repeat 都是
1,762,087。quality 和 throughput 下 `planner_no_ready_calls` 均为 0，未发生 spill。

throughput-256 相比 quality 将 Wikitext candidate checks 减少约 8.7%，但四次
elapsed 范围重叠，2.064s 对 2.079s 的中位数差不足以证明稳定吞吐提升。它同时将
padding ratio 从 3.50% 小幅降到 3.39%；当前数据只支持“候选检查减少且质量未退化”，
不支持把 wall-time 差异当作确定结论。

### Text-file fork 回归

首次重复 Wikitext run 被严格守恒校验拦截：sample count 都是 20,000，但 raw length
不同。根因是 benchmark 的 `TextLineDataset` 在父进程缓存 reader 后，Linux `fork`
workers 继承并共享同一个 file offset；`__getstate__` 只覆盖 spawn，无法处理 fork。

修复前，父进程预读后连续三次 4-worker raw sum 为 1,760,265、1,761,937 和
1,754,617；正确值是 1,762,087。reader 改为按 PID 重开并增加回归测试后，145 上
连续三次都精确得到 1,762,087，随后完整测试和上述 Wikitext benchmark 均通过。
这是 benchmark dataset 的多进程读取修复，不改变 LBA planner 的选择语义。

本地原始结果保存在 workspace 顶层目录（不随本仓库提交）：

```text
debug/lba-review/remote-145/cap1.csv
debug/lba-review/remote-145/synthetic20k-quality.csv
debug/lba-review/remote-145/wikitext20k-quality-fixed.csv
debug/lba-review/remote-145/wikitext20k-throughput.csv
```

## 2026-07-20 v2 真实 DDP 复测

本轮用 LBA v2 的 `LBA(dataset, ...)` API 在 145 的 GPU 0、3 上运行，隔离代码目录为
`/home/zhuyin/debug/lba-v2-ddp-20260720-zxdzI4`。环境为 Python 3.12.0、PyTorch
2.9.0+cu128 和 2 张 RTX 4090 D；默认 DDP process group 使用 NCCL，未禁用 IB 或
P2P。

先运行 `ddp_smoke.py`。两个 rank 都完成 3 个 optimizer steps，且每个 rank 的结构化
日志都记录：

```json
{"default_backend": "nccl", "metadata_backend": "gloo", "world_size": 2}
```

这验证了训练梯度仍走 NCCL，而 LBA 的长度预算、source presence 和 final-flush
metadata 通过独立 Gloo group 同步。

正式 benchmark 使用 Wikitext text-file 20k、`batch_size=32`、`num_workers=4`、
`max_padded_length=4096`、`max_padding_ratio=0.05`、`compute_iters=4`、pin memory 和
`simulate_step_sec=0`。先 warmup 1 次，再 measured 4 次并交替 baseline/LBA 顺序；
下表为 4 次中位数：

| mode | elapsed | loader wait sum | samples/s | raw tokens/s | steps/rank | padded length | padding ratio | planner pop_ready | candidate checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.934s | 0.139s | 21,411 | 1,886,368 | 313 | 5,548,720 | 68.24% | 0.000s | 0 |
| LBA quality | 2.308s | 3.314s | 8,665 | 763,452 | 346 | 1,826,009 | 3.50% | 1.693s | 1,308,198 |

每个 repeat 的 baseline/LBA 都处理 20,000 samples，`raw_length_sum` 都是 1,762,087；
没有 sample drop、hang、spill 或 `no_ready`。final flush 使用 index metadata，两个 rank
步数一致。LBA 将 padded length 降低 67.09%，padding ratio 相对降低 94.87%，但当前
轻量模型下 wall time 仍受 planner 限制，不能据此声称训练吞吐提升。

原始 CSV、stdout、stderr 和每个 rank 的 JSONL 保存在：

```text
debug/lba-review/remote-145/v2-20260720/
```

## 2026-07-22 Custom Batch Cost DDP Smoke

本轮在 145 的 GPU 5、6 上验证 custom batch cost。环境为 Python 3.12.0、
PyTorch 2.9.0+cu128、2 张 RTX 4090 D；训练 process group 使用 NCCL，LBA metadata
使用独立 Gloo group。为了绕开机器当前 IB/P2P 状态，命令设置
`NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1`。

smoke dataset 有 128 个 map-style samples，source `batch_size=8`，
`prefetch_batches=4`、`cost_window_batches=4`、pin memory 和严格 final
flush。cost model 与预算为：

```python
def attention_cost(max_length: int, batch_size: int) -> int:
    return max_length * max_length * batch_size


max_batch_cost = 16_384
```

| metric | rank 0 | rank 1 |
| --- | ---: | ---: |
| steps | 17 | 17 |
| samples | 74 | 54 |
| maximum emitted cost | 16,184 | 16,384 |
| resolved max batch cost | 16,384 | 16,384 |
| resolved max padded length | `None` | `None` |
| isolated metadata group | yes | yes |

两个 rank 合计输出 128 个 samples，unique index 也是 128，没有静默丢失或重复。final
flush 可以让各 rank 样本数不同，但保持训练 step 数完全相同。所有非 singleton batch
都满足 custom cost budget。本次只验证 cost budget、window、prefetch、NCCL/Gloo
group 和 final-flush 契约，不用于证明真实模型的 compute-duration balance。

同一工作树另跑了原有 synthetic DDP benchmark：LBA 保持 62 steps/rank、
`padding_ratio=3.0796%`、`candidate_window_checks=47,800`、
`planner_no_ready_calls=0`，与引入 custom cost 前的 legacy 路径结果一致。

## 2026-07-22 Distributed Cost Window Smoke

本轮在 145 的 GPU 5、6 上验证 `distributed_cost_window_batches`。代码先同步到隔离目录
`/tmp/lba-global-cost`，没有修改共享 checkout；环境为 Python 3.12.0、PyTorch
2.9.0+cu128 和 2 张 RTX 4090 D，NCCL 默认路径直接通过。最终 py312 测试为
`139 passed, 328 subtests passed`。

先用 `benchmarks/ddp_smoke.py` 的 rank-dependent dataset 做 NCCL 对照。rank 0 的
steady plan cost 为 100，rank 1 为 2；两种模式都完成 3 steps、全局 8 samples：

| mode | rank 0 costs | rank 1 costs | steady cost spread | steps/rank | global samples |
| --- | --- | --- | ---: | ---: | ---: |
| default (`None`) | `[100, 100, 100]` | `[2, 2, 100]` | `98` | 3 | 8 |
| global (`K=2`) | `[100, 2, 100]` | `[100, 2, 100]` | `0` | 3 | 8 |

global 模式中包含两个 steady plan 的 block 只产生一次 `distributed_cost_block` event；NCCL
训练 collective 与 Gloo metadata group 均成功，未使用 forward barrier。这个 smoke 只
验证 cost quantile 对齐和 sample/step 守恒，不代表真实模型吞吐。

随后运行 synthetic map-style workload：`size=4096`、`batch_size=32`、
`num_workers=0`、`max_padded_length=8192`、`prefetch_batches=8`、`compute_iters=16`，
每个模式 warmup 1 次、测量 4 次；表中是 LBA measured runs 的中位数。两组均处理
4096 samples、95 steps/rank、raw length 167,459、padding ratio 2.456%，且 baseline/LBA
严格 workload 校验通过。

| mode | elapsed | time to first batch | loader wait sum | samples/s | steps/rank | padding ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| local (`distributed_cost_window_batches=None`) | 0.758s | 0.0110s | 0.0122s | 5,406 | 95 | 2.456% |
| global (`K=8`) | 0.829s | 0.0210s | 0.0331s | 4,942 | 95 | 2.456% |

该受控 workload 中 global wall time 增加约 9.4%、samples/s 降低约 8.6%，首 batch
延迟增加约 91%，loader wait sum 增加约 171%。随机分片的 rank cost 分布已经接近，
matching 收益不足以覆盖 block gather 和重读开销；即使 dataset 只是轻量整数 lookup 也
不应默认启用。另一方面，前述 rank-dependent smoke 的 steady cost spread 确实从 98
降到 0。后续必须在真实 cost imbalance workload 上同时记录 remote records、ready
queue empty ratio、step-start spread、模型 forward/backward duration 和总 wall time，
再决定 K 与是否启用该模式。

原始 CSV 保存在 workspace 顶层 debug 目录：

```text
debug/lba-cost-local.csv
debug/lba-cost-global.csv
```

## 2026-07-22 Rank-Imbalance DDP Benchmark

本轮在 145 的 GPU 5、7 上新增并验证 rank-imbalance benchmark。代码同步到隔离目录
`/tmp/lba-rank-imbalance`，没有修改共享 checkout；环境为 Python 3.12.0、PyTorch
2.9.0+cu128 和 2 张 RTX 4090 D，NCCL 默认路径直接通过。命令使用 synthetic
4096 samples、`batch_size=32`、`num_workers=0`、`max_padded_length=8192`、
`prefetch_batches=4`，并给两个 rank 设置不同消费 profile：

```bash
CUDA_VISIBLE_DEVICES=5,7 PYTHONPATH=src \
  /home/zhuyin/anaconda3/envs/py312/bin/torchrun \
  --standalone --nproc_per_node=2 benchmarks/ddp_benchmark.py \
  --dataset synthetic --size 4096 --seed 123 --max-length 512 \
  --batch-size 32 --num-workers 0 --max-padded-length 8192 \
  --max-padding-ratio 0.05 --prefetch-batches 4 --compute-iters 4 \
  --rank-compute-iters 4,32 --rank-simulate-step-sec 0.0,0.01 \
  --repeats 2 --warmup-runs 1 --run-order alternate \
  --output outputs/ddp_rank_imbalance_2gpu.csv
```

两个 measured repeats 都严格通过 baseline/LBA workload 校验。表中列出逐 run 结果；
`step_compute_sec_spread` 是各 rank 累计 step compute 的 max-min，`rank_step_delay`
确认 slow rank 每步额外 sleep `0.01s`。

| repeat | mode | elapsed | samples/s | steps/rank | padding ratio | step compute spread | rank step delay spread |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | LBA | 2.013s | 2,034 | 95 | 2.456% | 0.0056s | 0.010s |
| 0 | baseline | 0.947s | 4,325 | 64 | 82.557% | 0.0078s | 0.010s |
| 1 | baseline | 0.964s | 4,250 | 64 | 82.557% | 0.0091s | 0.010s |
| 1 | LBA | 1.958s | 2,092 | 95 | 2.456% | 0.0098s | 0.010s |

该 benchmark 主要验证新入口可以构造不同 GPU 消费时间并记录 rank profile / compute
spread。当前 synthetic workload 下 LBA 仍显著降低 padding，但因产生更多 optimizer
steps，rank-imbalance 场景总 wall time 高于 baseline；这说明后续判断 adaptive 策略
必须同时看 padding ratio、steps/rank 和 rank 同步等待，而不能只看单步 cost spread。
原始 CSV 已拉回本地 workspace 顶层：

```text
debug/lba-ddp-rank-imbalance-2gpu.csv
```

随后用同一 setup 复跑 `planner_mode="latency"`，验证 capped miss 不再等待未来样本的
低延迟路径：

| repeat | mode | elapsed | loader wait sum | steps/rank | padding ratio | planner pop_ready | candidate checks | no_ready |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | LBA latency | 2.171s | 0.0139s | 100 | 2.413% | 0.184s | 32,290 | 0 |
| 1 | LBA latency | 2.213s | 0.0144s | 100 | 2.413% | 0.190s | 32,290 | 0 |

相比 quality 模式，latency 模式在这个 workload 中把 candidate checks 从 156,749 降到
32,290，planner pop-ready time 从约 0.48s 降到约 0.19s，并保持 `no_ready=0`。
这条路径验证的是“不等、直接从当前缓存里选”的行为；总 wall time 仍主要由 step 数和
slow rank 的消费时间决定。原始 CSV：

```text
debug/lba-ddp-rank-imbalance-latency-2gpu.csv
```
