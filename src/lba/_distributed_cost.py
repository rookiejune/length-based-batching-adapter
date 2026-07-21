"""Pure metadata matching for distributed cost windows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

from ._records import BatchPlan, PlanReason


@dataclass(frozen=True)
class RecordMetadata:
    """Dataset metadata needed to materialize one remote sample."""

    index: Optional[int]
    length: int
    arrival_id: int


@dataclass(frozen=True)
class PlanMetadata:
    """Serializable representation of a planned batch."""

    records: tuple[RecordMetadata, ...]
    reason: PlanReason
    estimated_cost: int


@dataclass(frozen=True)
class PlanRef:
    """One globally matched plan and its source location."""

    source_rank: int
    source_position: int
    metadata: PlanMetadata


def plan_metadata(plan: BatchPlan) -> PlanMetadata:
    """Remove sample objects from a plan before distributed exchange."""

    return PlanMetadata(
        records=tuple(
            RecordMetadata(
                index=record.index,
                length=record.length,
                arrival_id=record.arrival_id,
            )
            for record in plan.records
        ),
        reason=plan.reason,
        estimated_cost=(
            plan.estimated_cost
            if plan.estimated_cost is not None
            else plan.padded_length
        ),
    )


def match_cost_block(
    gathered: Sequence[Sequence[PlanMetadata]],
    *,
    step_offset: int = 0,
) -> tuple[tuple[PlanRef, ...], ...]:
    """Match adjacent global cost quantiles into deterministic DDP steps."""

    if not gathered:
        raise RuntimeError("LBA distributed cost matching requires at least one rank.")
    if step_offset < 0:
        raise ValueError("step_offset must be non-negative.")

    block_size = len(gathered[0])
    if block_size == 0:
        raise RuntimeError("LBA distributed cost matching received an empty block.")
    if any(len(plans) != block_size for plans in gathered):
        raise RuntimeError(
            "LBA distributed cost matching requires the same plan block size "
            "on every rank."
        )

    refs: list[PlanRef] = []
    for source_rank, plans in enumerate(gathered):
        for source_position, metadata in enumerate(plans):
            _validate_metadata(metadata)
            refs.append(
                PlanRef(
                    source_rank=source_rank,
                    source_position=source_position,
                    metadata=metadata,
                )
            )

    refs.sort(
        key=lambda ref: (
            -ref.metadata.estimated_cost,
            ref.source_rank,
            ref.source_position,
        )
    )
    world_size = len(gathered)
    assigned: list[list[PlanRef]] = [[] for _ in range(world_size)]
    for local_step in range(block_size):
        step_refs = refs[local_step * world_size : (local_step + 1) * world_size]
        rotation = (step_offset + local_step) % world_size
        for cost_position, ref in enumerate(step_refs):
            target_rank = (rotation + cost_position) % world_size
            assigned[target_rank].append(ref)

    return tuple(tuple(rank_refs) for rank_refs in assigned)


def _validate_metadata(metadata: PlanMetadata) -> None:
    if not metadata.records:
        raise RuntimeError("LBA distributed cost matching received an empty plan.")
    if metadata.estimated_cost <= 0:
        raise RuntimeError(
            "LBA distributed cost matching requires positive estimated costs."
        )
    if any(record.index is None for record in metadata.records):
        raise RuntimeError(
            "LBA distributed cost matching requires map-style sample indices."
        )
    if any(record.length <= 0 for record in metadata.records):
        raise RuntimeError(
            "LBA distributed cost matching requires positive sample lengths."
        )


__all__ = [
    "PlanMetadata",
    "PlanRef",
    "RecordMetadata",
    "match_cost_block",
    "plan_metadata",
]
