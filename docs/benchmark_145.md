# 145 Benchmark 记录

## 环境

- 机器：`145.pami.group`
- Python 环境：`py312`
- PyTorch：`2.9.0+cu128`
- 代码目录：早期单进程记录使用 `~/lba_benchmark_run/lba`，DDP 复测使用 `~/repos/lba`
- 本地结果备份：`lba/outputs/remote_145/`

## 数据集

所有结果都使用原始 `DataLoader` 的 `batch_size=32`。LBA 产出的 batch size
是动态的，但 `max_padded_length` 的默认推断基于这个原始 batch size。

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

## 指标

每个 run 记录：

- `elapsed_sec`：完整迭代耗时。
- `time_to_first_batch_sec`：首个 batch 产出耗时。
- `loader_wait_sec`：消费端等待 `next(loader)` 返回的总时间。
- `loader_wait_per_batch_sec`：平均每个 batch 的 loader 等待时间。
- `simulated_gpu_sec`：benchmark 中每个 batch 后模拟 GPU 消费的 sleep 时间。
- `raw_length_sum`：样本真实长度之和。
- `padded_length_sum`：batch padding 后总长度，计算为 `max_length * batch_size`。
- `padding_length_sum`：`padded_length_sum - raw_length_sum`。
- `padding_ratio`：`padding_length_sum / padded_length_sum`。

## 结果

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

## 结论

LBA 对 padding 的改善非常明显。Wikitext 上，padded length 大约减少 65% 到 66%，padding ratio 从约 67% 降到约 3.8%。

当前瓶颈不是 IO。Wikitext 20k 下，`num_workers=0` 和 `num_workers=4` 的 LBA 耗时几乎一样，说明多进程读取不是限制，主进程 planner 才是主要瓶颈。

当前实现不适合继续直接放大数据规模。50k Wikitext 已经需要约 65 秒，下一步应该先优化 planner，再做更大规模 benchmark。

## 对 Planner 的启示

- 不能在每个 batch 后做高成本全局候选搜索。
- 需要让 planner 的候选维护接近增量式，而不是反复扫描整个 pool。
- `max_padding_ratio` 快速提交路径是必要的，但还不够。
- 需要讨论是否引入长度 bucket、局部窗口索引、候选缓存或更强的早停规则。
- benchmark 暂时应保留 20k/50k 规模，作为 planner 优化前后的回归数据。

## Prefetch Producer 测试

2026-06-19 在 145 上同步当前代码后，使用 `prefetch_batches` 做了一组测试。结果
备份在本地 `lba/outputs/remote_145/`。

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
| current, `max_padding_ratio=0.1` | 634 | 17.82s | 1,895,454 | 7.04% |
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

本轮优化不改变默认 planner 策略：quality 模式仍检查完整 recent candidate 集合，
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
- 默认 quality planner 的主要内部热点已经从每候选 range-min / candidate 构造，压回到
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
