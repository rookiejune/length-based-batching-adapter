# LBA 边界情况

## 超长样本

如果单个样本的 `len_fn` 返回值超过 `max_padded_length`，LBA 会将该样本单独组成
batch。这个 singleton 是预算上限的显式例外，同时会：

- 发出 Python warning。
- 写入 LBA 日志文件。
- 只记录长度、预算、可选 dataset index 和 sample type，不记录 sample `repr`。

## 动态 Batch Size 与 `__len__`

LBA 会改变 batch size，训练代码不能假设每个 batch 的样本数固定。

最终 batch 数只有 planner 消费样本后才能确定。因此 LBA 虽然是 DataLoader 子类，
`len(loader)` 仍明确不可用。进度条、scheduler 和训练终止条件应使用显式 step/epoch
预算或运行时计数。

## 迭代顺序与守恒

LBA 会改变样本顺序，不保证 dataset sampler 的严格输出顺序。完整消费一个普通非
DDP iteration 且不设置 `max_batches` 时，planner 会 flush 所有收到的 samples。
source `drop_last`、`max_batches`、提前停止、异常和 DDP final-tail drop 都是样本不
守恒的明确例外。

## 空输入与预算推断

空 source 无法推断 `max_padded_length`，会直接抛出 `ValueError`。如果空 iteration
本身合法，应显式配置正预算；`max_batches=0` 是不会读取 source 的特例。

## 多进程

DataLoader worker 读取 raw samples，并在 source collate 中执行 `len_fn`；planner 在
主进程维护全局缓存。

使用 `spawn` multiprocessing context 时，`len_fn` 必须可 pickle。应使用模块顶层
函数或 callable class，不能使用 lambda 或局部函数。

map-style dataset 会被内部索引 wrapper 包装。wrapper 保留 `__getitems__` 快路径并
转发普通属性读写，但 `get_worker_info().dataset` 不再和原 dataset 对象相同；依赖
精确对象身份的 `worker_init_fn` 应通过 `info.dataset.dataset` 取得原对象。

`pin_memory=True` 应用于最终 `collate_fn` 的输出，而不是内部 records。

## Lightning Sampler 注入

Lightning 只会重建 DataLoader 并向 map-style dataset 注入 `DistributedSampler`。LBA
v2 的 DataLoader 子类身份使这条路径可用；注入后的 sampler 必须进入内部 source
loader。

不要同时在 DataModule 中手工分片，也不要额外调用 sampler `set_epoch()`。重复分片
会漏数据；两套 epoch 推进会让采样顺序难以审计。

当 dataset size 不能整除 world size 且 `DistributedSampler(drop_last=False)` 时，
sampler 会补齐 index。补齐项是有意的重复样本，LBA 不会去重。`drop_last=True` 则会
丢弃不能平均分配的尾部。调用侧必须显式选择符合训练目标的语义。

## 序列化与 Spill

pool 超过 `max_cache_samples` 时，LBA 使用 pickle 把完整 sample record 写入 spill
shard。sample 不可 pickle 时异常直接暴露。显式 `spill_dir` 包含训练 sample 的序列化
内容，应只使用可信目录并按训练数据敏感级别保护。

DDP object-gather final flush 同样要求尾部 sample 可 pickle。map-style indexed flush
只传 metadata，不受 sample object 传输大小影响，但必须满足确定性重取契约。

## DistributedDataParallel

初始化 `torch.distributed` 后，每个非空 source batch 在 steady state 对应一个
planned batch。final flush 汇总各 rank 剩余 records，重新规划并分发相同步数，避免
DDP backward 次数不一致。

map-style records 有 index metadata 时优先只交换 `(sample_index, length)`；否则所有
rank 统一使用 object gather，避免 collective 路径交错。

indexed 路径会在接收 rank 主进程重新调用 `dataset[index]`。LBA 只检查 index 是否
存在，不判断 lookup 是否稳定。调用侧必须保证读取可在 worker 外执行、确定、无副
作用，并返回相同 sample 和有效长度。改变长度、依赖 worker 或内部 cursor 的随机
transform 不符合该契约。

所有 rank 的 source batch stream 数量必须一致且每批非空；所有 rank 必须同步消费和
停止。某个 rank 单独 break，或 dataset、`len_fn`、`collate_fn` 报错，可能让 peers
卡在下一次 collective。LBA 使用 default process group，没有 subgroup 参数。

显式 `max_padded_length` 必须跨 rank 相同；自动推断会取各 rank 推断值的最大值。
`max_padding_ratio`、planner search limits、`drop_last_flush` 和 `max_batches` 等影响
控制流的配置也必须一致，当前实现不会逐项跨 rank 校验。

默认 `drop_last_flush=True`。final tail 无法给每个 rank 组成非空 step 时，LBA 丢弃
尾部并 warning；设置为 `False` 时直接报错。它不覆盖 source `drop_last`、
`max_batches` 或提前停止。

distributed 模式可以使用后台 prefetch。`prefetch_batches > 0` 时，LBA 会先创建独立
Gloo metadata group，再让 producer thread 执行 source-batch sync、planner、final
collate 和 pinning，避免 producer thread 和训练线程在默认 process group 上交错
collective。默认 process group 使用 NCCL 时同样需要 Gloo backend 同步 CPU metadata。
显式 `spill_dir` 会按 `rank-xxxxx` 子目录隔离。

## IterableDataset

IterableDataset 使用相同的 `LBA(dataset, ...)` 入口。必须配置 batched loading；
`batch_size=None` 不支持。

Lightning 和 PyTorch 不会为 IterableDataset 注入 `DistributedSampler`。dataset 必须
结合 distributed rank 和 `get_worker_info()` 自己分片，并保证所有 rank 产生相同数量
的非空 source batches。

repeatability 和 cursor 语义由 iterable 自身决定。可重入 iterable 可以重新开始；
one-shot iterator 不会被自动重建或回放。只消费前缀时，lookahead/prefetch 可能已经
读取最后一个输出 batch 之后的 items。distributed final flush 使用 object gather，
尾部 sample 必须可 pickle。

## Mid-Epoch Resume

LBA 不保存 iterator cursor、planner pool、spill consumption cursor 或 prefetched
lookahead。mid-epoch checkpoint 恢复不能保证与未中断运行具有相同后续 sample 序列，
可能重复或跳过 source samples。

需要精确续采时，调用侧必须提供 stateful dataset/sampler，并把 LBA pending state
纳入 checkpoint；仅恢复 model、optimizer、global step 和 sampler epoch 不足以恢复
数据位置。

## Collate 开销

最终 `collate_fn` 在主进程调用。collate 很重时可能成为吞吐瓶颈；当前版本不提供
worker-side final collate。
