# LBA 边界情况

## 超长样本

如果单个样本的 `len_fn` 返回值超过 `max_padded_length`，LBA 会将该样本单独组成
batch。这个 singleton 是预算上限的显式例外，同时会：

- 发出 Python warning。
- 写入 LBA 日志文件。
- 只记录长度、预算、可选 dataset index 和 sample type，不记录 sample `repr`，避免
  把训练数据写入日志。

## 动态 Batch Size

LBA 会改变 batch size。训练代码不应假设每个 batch 的样本数固定。

第一版不实现 `__len__`，避免进度条或训练框架拿到误导性的 batch 数。

## 迭代顺序

LBA 会改变样本迭代顺序，不保证原始 dataloader 的严格顺序。完整消费一个普通
非 DDP iteration 且不设置 `max_batches` 时，planner 会 flush 所有从 source loader
收到的 samples。source loader 自身的 `drop_last`、`max_batches`、调用侧提前停止、
异常，以及 DDP 默认的 final-tail drop 都可能让本次调用不再守恒。

## 空输入与预算推断

空 source 无法推断 `max_padded_length`，会直接抛出 `ValueError`。如果空 iteration
本身是合法结果，应显式配置正数预算；`max_batches=0` 是特例，它不会读取 source。

## 多进程

原始 dataloader 的 worker 用于读取 raw samples，并在 source collate 中执行
`len_fn`。planner 仍在主进程中维护全局缓存。

如果 `DataLoader` 使用 `spawn` multiprocessing context，`len_fn` 必须可 pickle。
应使用模块顶层函数或 callable class，不能使用 lambda 或局部函数。

map-style dataset 会被内部索引 wrapper 包装。wrapper 保留 `__getitems__` 快路径并
转发普通属性读写，但 `get_worker_info().dataset` 不再与原 dataset 对象同一；依赖
精确对象身份或类型判断的 `worker_init_fn` 应通过 `info.dataset.dataset` 取得原对象。

原 loader 的 `pin_memory=True` 应用于最终 `collate_fn` 的输出，而不是内部 records。

## 序列化与 Spill

pool 超过 `max_cache_samples` 时，LBA 使用 pickle 把完整 sample record 写入 spill
shard。触发 spill 的 sample 必须可 pickle；失败时异常会直接暴露，不会静默跳过。
显式 `spill_dir` 会包含原始 sample object 的序列化内容，应只使用可信本地目录，并按
训练数据的敏感级别保护。当前 adapter 创建的 shard 会在 planner 消费或关闭时删除。

DDP 的 object-gather final flush 同样要求尾部 sample 可 pickle，并会在每个 rank
物化公共 object pool。map-style indexed flush 只传 metadata，因此不受 sample object
序列化大小影响，但必须满足下面的确定性重取契约。

## DistributedDataParallel

当 `torch.distributed` 已初始化时，LBA 在常规迭代阶段仍然保持每个 source
`DataLoader` batch 后产出一个 planned batch。最后 flush 时，各 rank 会把剩余
records 聚合成一个公共 pool，重新规划后再按 rank 分发 flush batches，避免 DDP
backward 次数不一致导致 collective hang。map-style dataset 的 records 有 index
metadata 时优先只交换 `(sample_index, length)`；如果任一 rank 没有 index metadata，
所有 rank 会统一回退到完整 sample object gather，避免不同 collective 路径交错。

index metadata 路径会在接收 rank 的主进程重新调用 `dataset[index]`。LBA 只检查
index 是否存在，不会检测它是否稳定；调用侧必须保证该调用可在 worker 外执行、没有
副作用，并返回与 worker 首次读取相同的 sample 和 `len_fn` 有效长度。带随机
transform、依赖 `get_worker_info()` 或内部 cursor 的 map-style dataset 不满足这个
契约；可以把不改变有效长度的随机 transform 移到最终 `collate_fn`，或者改用
`IterableLBA` 让 final flush gather 原 sample object。

DDP 模式要求所有 rank 的 source `DataLoader` batch 数一致，而且每个 source batch
必须非空，通常应配合 `DistributedSampler` 使用。显式传入的
`max_padded_length` 必须在所有 rank 一致；自动推断时会使用所有 rank 推断值中的
最大值。其他影响 planner 和控制流的 LBA 配置也必须一致，尤其是
`max_padding_ratio`、planner search limits、`drop_last_flush` 和 `max_batches`；当前
实现不会逐项跨 rank 校验这些值。

默认 `drop_last_flush=True`。如果最后 flush 的尾部样本无法组成每个 rank 都有
非空 batch 的 DDP step，LBA 会丢弃这部分尾部样本并发出 warning。如果这类尾部
丢弃不可接受，可以设置 `drop_last_flush=False`，此时相同情况会直接报错。它不覆盖
source `drop_last`、`max_batches` 或调用侧提前停止。

公共 flush 池只用于最后少量尾部 records，不适合作为全程样本传输路径。LBA 的后台
prefetch 线程不会在 DDP 模式启用，避免 prefetch 线程和训练线程发起的 distributed
collective 交错。

默认 process group 使用 NCCL 时，LBA 需要 PyTorch 同时提供 Gloo backend，并创建
独立 Gloo group 同步 CPU metadata；Gloo 不可用时会直接报错。

如果显式配置 `spill_dir`，DDP 下每个 rank 会自动写入独立的 `rank-xxxxx`
子目录，避免多个进程写同名 spill shard。

## IterableDataset

LBA 会自动识别 `IterableDataset`，并使用原始 dataloader 的 `batch_size` 和
`drop_last` 构造内部 source loader。`batch_size=None` 的 unbatched iterable
loader 暂不支持，因为原始 `collate_fn` 通常不是面向样本列表的 batch collate。

`IterableLBA` 是另一个入口，不重建 `DataLoader`。它是否能重复迭代完全取决于传入
的 `source_batches`：可重入 iterable 可以重复消费，generator 等 one-shot iterator
不会被 adapter 自动重建。

## Collate 开销

v1 中原始 `collate_fn` 在主进程调用。如果用户的 `collate_fn` 很重，
包装后可能影响吞吐。后续可以单独设计 worker-side collate 优化。
