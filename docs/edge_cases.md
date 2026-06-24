# LBA 边界情况

## 超长样本

如果单个样本长度超过最大 padded length，LBA 应将该样本单独组成 batch，并同时：

- 发出 Python warning。
- 写入 LBA 日志文件。
- 记录 sample 自己的 repr。

## 动态 Batch Size

LBA 会改变 batch size。训练代码不应假设每个 batch 的样本数固定。

第一版不实现 `__len__`，避免进度条或训练框架拿到误导性的 batch 数。

## 迭代顺序

LBA 会改变样本迭代顺序。它会尽量保证样本不丢失，但不保证原始 dataloader
的严格顺序。

## 多进程

原始 dataloader 的 worker 用于读取 raw samples，并在 source collate 中执行
`len_fn`。planner 仍在主进程中维护全局缓存。

## DistributedDataParallel

当 `torch.distributed` 已初始化时，LBA 在常规迭代阶段仍然保持每个 source
`DataLoader` batch 后产出一个 planned batch。最后 flush 时，各 rank 会把剩余
records 聚合成一个公共 metadata 池，重新规划后再按 rank 分发 flush batches，
避免 DDP backward 次数不一致导致 collective hang。map-style dataset 会优先只交换
`(sample_index, length)`；如果任一 rank 的尾部 records 没有稳定 index，所有
rank 会统一回退到 object gather，避免不同 collective 路径交错。

DDP 模式要求所有 rank 的 source `DataLoader` batch 数一致，通常应配合
`DistributedSampler` 使用。显式传入的 `max_padded_length` 必须在所有 rank
一致；自动推断时会使用所有 rank 推断值中的最大值。

默认 `drop_last_flush=True`。如果最后 flush 的尾部样本无法组成每个 rank 都有
非空 batch 的 DDP step，LBA 会丢弃这部分尾部样本并发出 warning。需要严格保证
样本不丢失时，可以设置 `drop_last_flush=False`，此时相同情况会直接报错。

公共 flush 池只用于最后少量尾部 records，不适合作为全程样本传输路径。LBA 的后台
prefetch 线程不会在 DDP 模式启用，避免 prefetch 线程和训练线程发起的 distributed
collective 交错。

如果显式配置 `spill_dir`，DDP 下每个 rank 会自动写入独立的 `rank-xxxxx`
子目录，避免多个进程写同名 spill shard。

## IterableDataset

LBA 会自动识别 `IterableDataset`，并使用原始 dataloader 的 `batch_size` 和
`drop_last` 构造内部 source loader。`batch_size=None` 的 unbatched iterable
loader 暂不支持，因为原始 `collate_fn` 通常不是面向样本列表的 batch collate。

## Collate 开销

第一版设计中，原始 `collate_fn` 在主进程调用。如果用户的 `collate_fn` 很重，
包装后可能影响吞吐。后续可以单独设计 worker-side collate 优化。
