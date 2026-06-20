# LBA

Length-based Batching Adapter for PyTorch `DataLoader`s with variable-length
samples.

LBA wraps an existing `DataLoader`, measures each raw sample with a user-provided
`len_fn`, and emits dynamic batches that keep
`max_length_in_batch * batch_size` under a length budget. The original
`collate_fn` is still used for the final batch.

```python
from lba import LBA

loader = LBA(
    dataloader,
    len_fn=lambda sample: len(sample["input_ids"]),
)

for batch in loader:
    train_step(batch)
```

## Status

This is an early package intended for experimentation with variable-length
sequence training. The current implementation includes:

- source record collation before the original `collate_fn`
- warmup-based or explicit `max_padded_length` resolution
- sorted-pool dynamic batch planning
- bounded background prefetch
- spill-to-disk support when the planner cache grows too large
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


base_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn,
)

loader = LBA(
    base_loader,
    len_fn=lambda sample: len(sample["input_ids"]),
    max_padded_length=8192,
    log_dir="outputs/lba_logs",
)

for batch in loader:
    train_step(batch)
```

`len_fn` receives a raw dataset sample before the original `collate_fn` runs.
It must return a positive integer.

LBA infers whether the wrapped loader uses a map-style dataset or an
`IterableDataset`. Map-style loaders reuse the original `batch_sampler`;
iterable loaders reuse `batch_size` and `drop_last`. Iterable loaders must be
batched (`batch_size` cannot be `None`), because LBA needs groups of raw samples
before applying the original `collate_fn`.

## Public API

The package exposes both a short alias and a descriptive class name:

```python
from lba import LBA, LengthBatchingAdapter

assert LBA is LengthBatchingAdapter
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
| `spill_dir` | Directory for planner spill shards. If omitted, LBA uses a temporary directory. |
| `log_dir` | Directory for per-run logs. If omitted, logs are written under `~/.lba/logs/`. |

## Distributed Training

When `torch.distributed` is initialized, LBA keeps the steady-state path close
to normal iteration: each rank emits one planned batch after each source
`DataLoader` batch. At the final flush, ranks gather their remaining records into
a shared metadata pool, replan that pool, and distribute the flush batches so
every rank performs the same number of DDP steps. Map-style datasets exchange
only `(sample_index, length)` metadata for this flush path; records without
stable indices fall back to object gathering.

Use source loaders that yield the same number of batches on every rank, such as
map-style datasets with `DistributedSampler`. Explicit `max_padded_length`
values must match on every rank; inferred budgets are synchronized with the
maximum inferred value.

## Logs

Each adapter run creates a log file and emits a warning with its path:

```text
~/.lba/logs/lba-YYYYmmdd-HHMMSS-PID.log
```

The summary includes both global token-weighted padding ratios and the mean of
per-batch padding ratios:

```text
LBA summary padding before_padding_ratio=... before_mean_batch_padding_ratio=... after_padding_ratio=... after_mean_batch_padding_ratio=... padding_ratio_reduction=...
LBA summary lengths before_batches=... before_samples=... before_raw_length_sum=... before_padded_length_sum=... before_padding_length_sum=... after_batches=... after_samples=... after_raw_length_sum=... after_padded_length_sum=... after_padding_length_sum=...
LBA summary planner planned_batches=... oversized_batches=... other_batches=... sort_time_seconds=... sort_calls=... average_sort_time_ms=... pop_ready_time_seconds=... pop_ready_calls=... average_pop_ready_time_ms=... candidate_window_checks=... average_candidate_window_checks=... max_candidate_window_checks=... fast_path_batches=... full_search_batches=... flush_search_batches=... planner_oversized_batches=... no_ready_calls=... records_sorted_total=... max_cache_size_seen=... spill_events=... spilled_records=...
```

Definitions:

- `before_*` describes the original `DataLoader` batches before LBA replans them.
- `after_*` describes the dynamic batches emitted by LBA.
- `padding_ratio` is `padding_length_sum / padded_length_sum`.
- `mean_batch_padding_ratio` is the arithmetic mean of each batch's padding ratio.
- `sort_time_seconds` measures time spent sorting the planner's in-memory pool.
- `pop_ready_*` and `candidate_window_checks` measure planner search work.
- `fast_path_batches`, `full_search_batches`, `flush_search_batches`, and
  `planner_oversized_batches` split batches by the planner path that emitted them.

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
  source.py        # source DataLoader construction and length records
  planner.py       # planner state machine
  candidates.py    # candidate batch window search
  spill.py         # spill-to-disk shards
  metrics.py       # padding and planner metrics
```

## Design Docs

- [Design](docs/design.md)
- [Usage](docs/usage.md)
- [Edge Cases](docs/edge_cases.md)
- [145 Benchmark](docs/benchmark_145.md)

## License

[MIT](LICENSE).
