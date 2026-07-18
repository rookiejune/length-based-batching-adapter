# LBA

Length-based Batching Adapter for PyTorch `DataLoader`s with variable-length
samples.

LBA wraps an existing `DataLoader`, measures each raw sample with a user-provided
`len_fn`, and emits dynamic batches that keep
`max_length_in_batch * batch_size` under a length budget. The original
`collate_fn` is still used for the final batch.

```python
from lba import LBA


def sample_length(sample):
    return len(sample["input_ids"])


loader = LBA(
    dataloader,
    len_fn=sample_length,
)

for batch in loader:
    train_step(batch)
```

## Status

LBA is now a stable v1 package for variable-length sequence training. The v1
default, `planner_mode="quality"`, is the supported baseline: it favors low
padding, predictable planner behavior, and DDP step alignment over experimental
shortcuts. The current implementation includes:

- source record collation before the original `collate_fn`
- warmup-based or explicit `max_padded_length` resolution
- sorted-pool dynamic batch planning
- stable quality-mode planning with uncapped representative fallback search
- opt-in throughput-mode planning for CPU-bound producer workloads
- bounded background prefetch
- spill-to-disk support when the planner cache grows too large
- DDP final-flush replanning that keeps ranks on the same number of steps
- per-run log files with before/after padding and planner timing summaries

## Installation

Install PyTorch first if you need a specific CUDA or platform build. LBA depends
on `torch>=2`, but GPU environments often need an explicitly selected PyTorch
wheel.

From GitHub:

```bash
python -m pip install "git+https://github.com/rookiejune/lba.git"
```

With SSH:

```bash
python -m pip install "git+ssh://git@github.com/rookiejune/lba.git"
```

For local development:

```bash
git clone git@github.com:rookiejune/lba.git
cd lba
python -m pip install -e ".[dev]"
```

## Quick Start

Create your normal `DataLoader` first, including the `collate_fn` your training
code already expects:

```python
from torch.utils.data import DataLoader
from lba import LBA


def collate_fn(samples):
    ...


def sample_length(sample):
    return len(sample["input_ids"])


base_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

loader = LBA(
    base_loader,
    len_fn=sample_length,
    max_padded_length=8192,
    log_dir="outputs/lba_logs",
)

for batch in loader:
    train_step(batch)
```

`len_fn` receives a raw dataset sample before the original `collate_fn` runs.
It must return a positive integer. When the source `DataLoader` uses the
`spawn` multiprocessing context, `len_fn` must also be picklable; use a
module-level function or callable class instead of a lambda or local function.

For v1 usage, start with the defaults and set `max_padded_length` explicitly
when your model has a known token or padded-length budget. Leave
`planner_mode="quality"` unless benchmark logs show the LBA producer is the
actual training bottleneck.

LBA infers whether the wrapped loader uses a map-style dataset or an
`IterableDataset`. Map-style loaders reuse the original `batch_sampler`;
iterable loaders reuse `batch_size` and `drop_last`. Iterable loaders must be
batched (`batch_size` cannot be `None`), because LBA needs groups of raw samples
before applying the original `collate_fn`.

The internal source loader is reused across adapter iterations, so
`persistent_workers=True`, batched dataset `__getitems__`, and `in_order` keep
their source-loader behavior. When the wrapped loader has `pin_memory=True`,
LBA pins the final result after the original `collate_fn`, rather than trying to
pin internal length records.

## Public API

The package exposes short aliases and descriptive class names for both source
styles:

```python
from lba import (
    IterableLBA,
    IterableLengthBatchingAdapter,
    LBA,
    LengthBatchingAdapter,
)

assert LBA is LengthBatchingAdapter
assert IterableLBA is IterableLengthBatchingAdapter
```

Use `IterableLBA` when the caller already produces batches of raw samples and
does not want LBA to rebuild a `DataLoader`:

```python
loader = IterableLBA(
    source_batches,
    collate_fn=collate_fn,
    len_fn=sample_length,
    batch_size=32,
)
```

## Configuration

```python
loader = LBA(
    dataloader,
    len_fn=len_fn,
    max_padded_length=None,
    warmup_batches=None,
    max_cache_samples=8192,
    max_padding_ratio=0.05,
    prefetch_batches=4,
    planner_mode="quality",
    max_candidate_windows=None,
    limited_search_fallback_after=None,
    limited_search_fallback_pool_size=None,
    drop_last_flush=True,
    max_batches=None,
    spill_dir=None,
    log_dir=None,
)
```

| Argument | Meaning |
| --- | --- |
| `dataloader` | Existing PyTorch `DataLoader` to wrap. |
| `len_fn` | Required callable that returns the effective length of one raw sample. |
| `max_padded_length` | Hard budget for `max_length_in_batch * batch_size`. If omitted, LBA estimates it from warmup records. |
| `warmup_batches` | Number of original dataloader batches used for length-budget inference. Defaults to `min(batch_size, 32)` when possible, otherwise `1`. |
| `max_cache_samples` | Maximum in-memory planner pool size before spilling old records to disk. |
| `max_padding_ratio` | Padding threshold used when deciding whether a candidate batch is ready. Default is `0.05`. |
| `prefetch_batches` | Bounded background queue depth. Set to `0` for fully synchronous iteration. Disabled automatically when `torch.distributed` is initialized. |
| `planner_mode` | Planner search mode. `"quality"` is the default and keeps uncapped representative fallback search; `"throughput"` limits steady-state recent-window search. |
| `max_candidate_windows` | Optional cap on recent-window candidates inspected by each non-flush `pop_ready` call. Defaults to `None` in quality mode and `256` in throughput mode. |
| `limited_search_fallback_after` | In throughput mode, allow an uncapped representative fallback after this many capped-search misses. Defaults to `8`; set `None` outside throughput mode. |
| `limited_search_fallback_pool_size` | In throughput mode, remove the recent-window cap when the planner pool reaches this many records. Defaults to `min(max_cache_samples, 1024)`; set `None` outside throughput mode. |
| `drop_last_flush` | In distributed mode, drop final flush samples that cannot form a non-empty batch on every rank. Defaults to `True` and emits a warning when samples are dropped. |
| `max_batches` | Maximum final batches for this adapter iteration. Reaching the limit discards the remaining lookahead cache without a final flush. |
| `spill_dir` | Directory for planner spill shards. If omitted, LBA uses a temporary directory. |
| `log_dir` | Directory for per-run logs. If omitted, logs are written under `~/.lba/logs/`. |

`planner_mode="throughput"` is an explicit tradeoff: capped recent search keeps
ordinary `pop_ready` calls bounded, but LBA still runs an adaptive uncapped
representative fallback after repeated capped-search misses. When the planner pool grows too
large, LBA first removes the candidate-window cap for threshold search so more
ready work is paid down before final flush. Final flush uses the same uncapped
representative search so remaining samples are not silently skipped.

## Stable v1 Defaults

The stable v1 recommendation is to use the quality planner with the default
padding threshold:

```python
loader = LBA(
    dataloader,
    len_fn=len_fn,
    max_padded_length=8192,
    planner_mode="quality",
    max_padding_ratio=0.05,
)
```

The throughput planner is useful when benchmark logs show that candidate search
is limiting the training loop. It is not the default because simply narrowing the
candidate range can leave more work for final flush, increase the number of
emitted batches, or trade padding quality for producer speed. The adaptive
throughput fallback reduces that flush debt, but it also moves extra search work
back into steady-state iteration, so it is best treated as an opt-in safety valve
rather than a universally better strategy.

## Distributed Training

When `torch.distributed` is initialized, LBA keeps the steady-state path close
to normal iteration: each rank emits one planned batch after each source
`DataLoader` batch. At the final flush, ranks gather their remaining records into
a shared metadata pool, replan that pool, and distribute the flush batches so
every rank performs the same number of DDP steps. Map-style datasets exchange
only `(sample_index, length)` metadata for this flush path when every rank has
stable indices; if any rank lacks indices, all ranks fall back to object
gathering.

The indexed path re-reads assigned tail samples through `dataset[index]` in the
receiving rank's main process. Here, a stable index means that lookup is
deterministic, side-effect-free, valid outside a worker, and returns the same
sample and length. Map-style datasets with random or worker-dependent transforms
do not satisfy that contract; keep the indexed dataset deterministic and apply
such transforms in the final `collate_fn`, or use `IterableLBA` so the final
flush gathers the original sample objects.

Use source loaders that yield the same number of batches on every rank, such as
map-style datasets with `DistributedSampler`. Explicit `max_padded_length`
values must match on every rank; inferred budgets are synchronized with the
maximum inferred value.

CUDA training should continue to initialize the default process group with
NCCL. When LBA sees an NCCL default group, it creates a separate Gloo process
group only for CPU-side metadata synchronization, including small integer
reductions and final-flush object metadata. This keeps model gradient
collectives on NCCL while avoiding unnecessary CUDA traffic for Python and
batch-planning metadata.

Distributed iteration requires one planned batch for every non-empty source
batch. If a capped throughput search misses, that rank immediately runs the
uncapped representative fallback instead of deferring locally and allowing
rank step counts to diverge.

By default, `drop_last_flush=True` drops final flush samples that cannot form a
non-empty DDP step on every rank, and LBA emits a warning with the dropped sample
count. Set `drop_last_flush=False` to fail instead.

When `spill_dir` is configured under DDP, LBA writes each rank under a
`rank-xxxxx` child directory to avoid shard filename collisions.

## Logs

Each adapter run creates a human-readable log and a matching JSONL event file,
then emits a warning with both paths:

```text
~/.lba/logs/lba-YYYYmmdd-HHMMSS-PID.log
~/.lba/logs/lba-YYYYmmdd-HHMMSS-PID.jsonl
```

The `.log` file is optimized for scanning during training:

```text
2026-06-25 14:03:12 INFO lba summary: padding 82.10% -> 4.70% (94.28% reduction) saved_padding=+123456 batches=312->428 samples=9984
2026-06-25 14:03:12 INFO lba planner: total=840.000ms pop_ready_avg=0.180ms sort_avg=1.200ms paths=fast:397/fallback:21/flush:10 max_cache=8192
2026-06-25 14:03:12 INFO lba health: oversized=0 spill_events=0 spilled_records=0 no_ready=18 other_batches=0 event_log=...
```

The `.jsonl` file keeps the full structured details for benchmarks and
regression checks. Important events include:

- `run_start`: resolved log paths and adapter configuration.
- `summary`: before/after padding ratios, token sums, planner timings, path
  counts, spill counters, and health counters.
- `oversized_sample`: a sample exceeded `max_padded_length`; logs length,
  budget, optional dataset index, and sample type without dumping the sample.
- `spill`: planner cache overflow wrote records to disk.
- `distributed_*`: DDP budget synchronization, final flush mode, and dropped
  final flush records.

In the summary event, `padding.before` describes the original `DataLoader`
batches before LBA replans them, and `padding.after` describes emitted dynamic
batches. `padding_ratio` is `padding_length_sum / padded_length_sum`;
`mean_batch_padding_ratio` is the arithmetic mean of each batch's padding ratio.

## Benchmarks

For timing comparisons, use multiple measured runs, a warmup run, and alternating
execution order:

```bash
PYTHONPATH=src python benchmarks/benchmark_lba.py \
  --dataset synthetic \
  --repeats 4 \
  --warmup-runs 1 \
  --run-order alternate
```

Each CSV row records its repeat and run position, the effective LBA planner
limits, the resolved `max_padded_length`, and planner cache/spill counters. The
benchmark verifies that baseline and LBA process the same sample count and raw
length.

`ddp_benchmark.py` uses `drop_last_flush=False` by default so an unsplittable
tail fails instead of silently changing the measured workload. Pass
`--drop-last-flush` only when dropped tail records are an intentional part of
the experiment; their warning remains visible and the workload difference is
reported.

## Development

Install editable development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Run tests:

```bash
python -m pytest
```

The tests can also run with the standard library test runner:

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## Project Layout

```text
src/lba/
  wrapper.py       # public adapter around a DataLoader
  config.py        # user-facing configuration and resolved defaults
  source.py        # source DataLoader construction and length records
  budget.py        # length-budget resolution
  planner.py       # planner state machine
  candidates.py    # candidate batch window search
  distributed.py   # DDP synchronization and final-flush planning
  logging_utils.py # run logging and structured event reporting
  spill.py         # spill-to-disk shards
  metrics.py       # padding and planner metrics
```

## Design Docs

- [Design](docs/design.md)
- [Usage](docs/usage.md)
- [Stable v1 Notes](docs/v1.md)
- [Edge Cases](docs/edge_cases.md)
- [145 Benchmark](docs/benchmark_145.md)

## License

[MIT](LICENSE).
