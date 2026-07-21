# LBA

Length-based `DataLoader` for PyTorch datasets with variable-length samples.

LBA measures each raw sample with a user-provided `len_fn` and emits dynamic
batches that keep `max_length_in_batch * batch_size` under a length budget. A
sample longer than the budget is emitted alone and reported as oversized. The
user-provided `collate_fn` still creates the final training batch.

```python
from lba import LBA


def sample_length(sample):
    return len(sample["input_ids"])


loader = LBA(
    dataset,
    len_fn=sample_length,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_fn,
    max_padded_length=8192,
)

for batch in loader:
    train_step(batch)
```

## Status

LBA 2.0 is a breaking API release. `LBA` is now the only public loader class,
it is constructed directly from a dataset, and it subclasses
`torch.utils.data.DataLoader`. This lets frameworks such as Lightning inspect
and reconstruct the loader to inject a `DistributedSampler`.

The dynamic number of output batches is not knowable before planning, so
`len(loader)` is intentionally unavailable even though `LBA` is a `DataLoader`
subclass. Use an explicit step or epoch budget instead of deriving training
control flow from loader length.

The package version is `2.0.0`. It requires Python 3.9 or newer and PyTorch 2 or
newer. The implementation includes:

- source record collation before the final `collate_fn`
- warmup-based or explicit `max_padded_length` resolution
- sorted-pool dynamic batch planning
- stable quality-mode planning with representative fallback search
- opt-in throughput-mode planning for CPU-bound producer workloads
- bounded background prefetch
- spill-to-disk support when the planner cache grows too large
- DDP final-flush replanning that keeps ranks on the same number of steps
- per-run log files with padding and planner timing summaries

## Installation

Install PyTorch first if you need a specific CUDA or platform build. LBA depends
on `torch>=2`, but GPU environments often need an explicitly selected PyTorch
wheel.

From GitHub:

```bash
python -m pip install "git+https://github.com/rookiejune/length-based-batching-adapter.git"
```

With SSH:

```bash
python -m pip install "git+ssh://git@github.com/rookiejune/length-based-batching-adapter.git"
```

For local development:

```bash
git clone git@github.com:rookiejune/length-based-batching-adapter.git
cd length-based-batching-adapter
python -m pip install -e ".[dev]"
```

## DataLoader Contract

`LBA` accepts a dataset, the required keyword-only `len_fn`, LBA planner
options, and standard `DataLoader` options such as `batch_size`, `shuffle`,
`sampler`, `num_workers`, `collate_fn`, `pin_memory`, and
`persistent_workers`. Standard PyTorch mutual-exclusion rules still apply. For
example, do not pass both `shuffle=True` and an explicit `sampler`.

`len_fn` receives a raw dataset sample before `collate_fn` runs and must return
a positive integer. With a `spawn` multiprocessing context, it must be
pickleable; use a module-level function or callable class instead of a lambda
or local function.

Map-style datasets retain batched `__getitems__`, worker, sampling, and
`persistent_workers` behavior through LBA's internal source loader. If
`pin_memory=True`, LBA pins the final collated batch rather than internal length
records.

## Configuration

Start with the quality planner and set `max_padded_length` explicitly when the
model has a known token or padded-length budget:

```python
loader = LBA(
    dataset,
    len_fn=sample_length,
    batch_size=32,
    collate_fn=collate_fn,
    max_padded_length=8192,
    max_padding_ratio=0.05,
    planner_mode="quality",
    log_dir="outputs/lba_logs",
)
```

Important LBA arguments:

| Argument | Meaning |
| --- | --- |
| `dataset` | Map-style or iterable PyTorch dataset. |
| `len_fn` | Required callable returning the effective length of one raw sample. |
| `max_padded_length` | Budget for `max_length_in_batch * batch_size`. If omitted, LBA estimates it from warmup records. An oversized sample is emitted as a singleton. |
| `warmup_batches` | Source batches used for budget inference. Warmup samples still enter the planner. |
| `max_cache_samples` | Maximum in-memory planner pool before old records spill to disk. Spilled samples must be pickleable. |
| `max_padding_ratio` | Fast-path readiness threshold. Fallback and flush batches may exceed it. |
| `prefetch_batches` | Background queue depth. Set to `0` for synchronous iteration. Under distributed execution, LBA uses an isolated Gloo metadata group before moving planning, final collation, and pinning into the producer thread. |
| `planner_mode` | `"quality"` is the default; `"throughput"` limits steady-state recent-window search. |
| `max_candidate_windows` | Optional cap on recent-window candidates. Defaults to no cap in quality mode and `256` in throughput mode. |
| `limited_search_fallback_after` | In throughput mode, allow an uncapped fallback after this many capped-search misses. |
| `limited_search_fallback_pool_size` | In throughput mode, remove the cap when the planner pool reaches this size. |
| `drop_last_flush` | Under DDP, drop a final tail that cannot create a non-empty batch on every rank. Defaults to `True` and warns. |
| `max_batches` | Maximum emitted batches for one iteration. Reaching it discards remaining lookahead instead of flushing it. |
| `spill_dir` | Planner spill directory. A temporary directory is used when omitted. |
| `log_dir` | Per-run log directory. Defaults to `~/.lba/logs/`. |

`planner_mode="throughput"` is an explicit tradeoff. It bounds ordinary recent
search but can defer more work to the final flush. Adaptive fallbacks reduce
that debt but do not make throughput mode universally faster. Switch only when
training-side loader wait, GPU utilization, and LBA statistics identify the
producer as a bottleneck.

## Lightning Distributed Training

Return LBA directly from a Lightning data hook and leave automatic distributed
sampling enabled:

```python
import lightning as L


class DataModule(L.LightningDataModule):
    def train_dataloader(self):
        return LBA(
            self.dataset,
            len_fn=sample_length,
            batch_size=32,
            shuffle=True,
            collate_fn=collate_fn,
            max_padded_length=8192,
        )


trainer = L.Trainer(
    devices="auto",
    strategy="ddp",
    use_distributed_sampler=True,
)
```

Lightning recognizes LBA as a `DataLoader`, reconstructs it with a
`DistributedSampler`, and calls the sampler's `set_epoch()` as epochs advance.
Do not also create a rank sampler or sampler-epoch callback in the data module.

PyTorch's `DistributedSampler(drop_last=False)` pads a dataset whose size is not
divisible by world size. Those padded indices are duplicates by design. LBA
does not deduplicate them. Use a divisible dataset, accept the duplicates, or
select an explicit drop policy according to the training contract.

During steady state, each rank emits one planned batch for each non-empty source
batch. At final flush, ranks gather remaining records into a shared pool,
replan it, and distribute flush batches so every rank performs the same number
of DDP steps. Map-style datasets exchange `(sample_index, length)` metadata when
indices are available; other cases gather sample objects.

The indexed path re-reads assigned tail samples through `dataset[index]` in the
receiving rank's main process. This lookup must be deterministic,
side-effect-free, valid outside a worker, and return the same effective length.
Move length-preserving random transforms to `collate_fn` when needed.

All ranks must consume and stop in lockstep. A rank-local early break or an
exception in the dataset, `len_fn`, or `collate_fn` can leave peers blocked in
the next collective. Explicit budgets and all planner options that affect
control flow must match across ranks. With distributed background prefetch, LBA
creates a separate Gloo group for metadata collectives before the producer
thread starts, so training collectives on the default group do not interleave
with LBA metadata collectives. NCCL default groups also use this Gloo metadata
group, so Gloo support is required.

`drop_last_flush=True` drops and warns about a final tail that cannot form a
non-empty step on every rank. Set it to `False` to fail instead.

## IterableDataset

The same `LBA(dataset, ...)` entry point accepts an `IterableDataset`, but
automatic distributed sampling does not apply. Lightning and PyTorch do not
inject `DistributedSampler` into iterable datasets. The dataset must shard
itself by distributed rank and DataLoader worker, and every rank must expose the
same number of non-empty source batches.

LBA requires batched iterable loading; `batch_size=None` is unsupported. The
iterable controls repeatability and cursor behavior. A one-shot iterator is not
rebuilt or replayed, and lookahead can consume source items beyond the last
emitted batch. Distributed final flush uses object gathering, so tail samples
must be pickleable.

## Checkpoint Resume

LBA does not checkpoint the source iterator cursor, planner pool, or prefetched
lookahead. Epoch-boundary resume works with the normal deterministic sampler
contract, but exact mid-epoch sample continuity is not guaranteed. Restarting
from a step checkpoint may replay or skip source samples relative to an
uninterrupted run.

Exact mid-epoch resume requires a stateful dataset or sampler plus explicit
checkpoint integration for its cursor and LBA's pending planner state. Do not
infer this guarantee from Lightning model and optimizer checkpoint support.

## Logs

Each loader creates one human-readable log and one matching JSONL event file.
Each planner iteration appends a summary to that pair:

```text
~/.lba/logs/lba-YYYYmmdd-HHMMSS-ffffff-PID.log
~/.lba/logs/lba-YYYYmmdd-HHMMSS-ffffff-PID.jsonl
```

The paths are available as `loader.log_path` and `loader.log_event_path`. After
an iteration exits, `loader.last_max_padded_length` contains the resolved budget
and `loader.last_planner_stats` contains its counters. Important JSONL events
include `run_start`, `summary`, `oversized_sample`, `spill`, and
`distributed_*`.

## Benchmarks

Use multiple measured runs, a warmup run, and alternating execution order:

```bash
PYTHONPATH=src python benchmarks/benchmark_lba.py \
  --dataset synthetic \
  --repeats 4 \
  --warmup-runs 1 \
  --run-order alternate
```

The DDP benchmark is a CUDA/NCCL entry point:

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=2 \
  benchmarks/ddp_benchmark.py \
  --dataset synthetic \
  --repeats 4 \
  --warmup-runs 1 \
  --run-order alternate
```

`ddp_benchmark.py` uses `drop_last_flush=False` by default so an
unsplittable tail fails instead of silently changing the measured workload.
Pass `--drop-last-flush` only when dropped tail records are intentional.

The latest controlled 2-GPU Wikitext benchmark before the v2 API migration
reduced padded length from `5,548,720` to `1,826,009` and padding ratio from
`68.24%` to `3.50%`. It did not show a stable wall-time advantage in a minimal
training step. These results validate planner behavior, not end-to-end model
throughput. See [the benchmark record](docs/benchmark_145.md).

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The suite can also run with the standard library runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Design Docs

- [Design](docs/design.md)
- [Usage](docs/usage.md)
- [Stable v2 Notes](docs/v2.md)
- [Edge Cases](docs/edge_cases.md)
- [145 Benchmark](docs/benchmark_145.md)

## License

[MIT](LICENSE)
