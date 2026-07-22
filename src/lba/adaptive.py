"""Opt-in adaptive controls for LBA planning."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Optional, Union

from ._records import BatchPlan


class _Disabled:
    def __repr__(self) -> str:
        return "DISABLED"


DISABLED = _Disabled()
AutoFloat = Union[float, None, _Disabled]
AutoInt = Union[int, None, _Disabled]


@dataclass(frozen=True)
class CostWindowStats:
    """Cost spread summary for one distributed plan block."""

    block_size: int
    mean_cost: float
    source_mean_step_spread: float
    matched_mean_step_spread: float
    source_spread_ratio: float
    improvement_ratio: float
    remote_plan_count: int
    remote_record_count: int


@dataclass(frozen=True)
class AdaptiveConfig:
    """Opt-in adaptive planner knobs.

    Omitted fields are disabled. Passing None to a field enables LBA's built-in
    automatic policy for that knob. Passing a concrete value keeps the knob
    fixed inside the adaptive path.
    """

    max_padding_ratio: AutoFloat = None
    distributed_cost_window_batches: AutoInt = DISABLED
    max_candidate_windows: AutoInt = DISABLED
    max_padding_ratio_values: tuple[float, ...] = (
        0.025,
        0.05,
        0.075,
        0.10,
        0.15,
    )
    distributed_cost_window_values: tuple[int, ...] = (2, 4, 8)
    max_candidate_window_values: tuple[int, ...] = (64, 128, 256, 512)
    low_padding_ratio: float = 0.50
    high_padding_ratio: float = 0.90
    padding_patience: int = 8
    high_cost_spread_ratio: float = 0.15
    low_cost_spread_ratio: float = 0.05
    min_cost_improvement_ratio: float = 0.25

    def __post_init__(self) -> None:
        max_padding_ratio_values = _float_values(
            self.max_padding_ratio_values,
            field_name="max_padding_ratio_values",
            minimum=0.0,
            maximum=1.0,
        )
        object.__setattr__(
            self,
            "max_padding_ratio_values",
            max_padding_ratio_values,
        )
        distributed_cost_window_values = _int_values(
            self.distributed_cost_window_values,
            field_name="distributed_cost_window_values",
            minimum=2,
        )
        object.__setattr__(
            self,
            "distributed_cost_window_values",
            distributed_cost_window_values,
        )
        max_candidate_window_values = _int_values(
            self.max_candidate_window_values,
            field_name="max_candidate_window_values",
            minimum=1,
        )
        object.__setattr__(
            self,
            "max_candidate_window_values",
            max_candidate_window_values,
        )

        if self.max_padding_ratio is not DISABLED:
            if self.max_padding_ratio is not None:
                value = _float_value(
                    self.max_padding_ratio,
                    field_name="max_padding_ratio",
                    minimum=0.0,
                    maximum=1.0,
                )
                object.__setattr__(self, "max_padding_ratio", value)
                _require_member(
                    value,
                    max_padding_ratio_values,
                    field_name="max_padding_ratio",
                )

        if self.distributed_cost_window_batches is not DISABLED:
            if self.distributed_cost_window_batches is not None:
                value = _int_value(
                    self.distributed_cost_window_batches,
                    field_name="distributed_cost_window_batches",
                    minimum=2,
                )
                object.__setattr__(
                    self,
                    "distributed_cost_window_batches",
                    value,
                )
                _require_member(
                    value,
                    distributed_cost_window_values,
                    field_name="distributed_cost_window_batches",
                )

        if self.max_candidate_windows is not DISABLED:
            if self.max_candidate_windows is not None:
                value = _int_value(
                    self.max_candidate_windows,
                    field_name="max_candidate_windows",
                    minimum=1,
                )
                object.__setattr__(self, "max_candidate_windows", value)
                _require_member(
                    value,
                    max_candidate_window_values,
                    field_name="max_candidate_windows",
                )

        for field_name in (
            "low_padding_ratio",
            "high_padding_ratio",
            "high_cost_spread_ratio",
            "low_cost_spread_ratio",
            "min_cost_improvement_ratio",
        ):
            _float_value(
                getattr(self, field_name),
                field_name=field_name,
                minimum=0.0,
                maximum=1.0,
            )
        if self.low_padding_ratio > self.high_padding_ratio:
            raise ValueError(
                "low_padding_ratio must be less than or equal to "
                "high_padding_ratio."
            )
        if self.low_cost_spread_ratio > self.high_cost_spread_ratio:
            raise ValueError(
                "low_cost_spread_ratio must be less than or equal to "
                "high_cost_spread_ratio."
            )
        _int_value(self.padding_patience, field_name="padding_patience", minimum=1)

    @property
    def adjusts_max_padding_ratio(self) -> bool:
        return self.max_padding_ratio is not DISABLED

    @property
    def adjusts_distributed_cost_window(self) -> bool:
        return self.distributed_cost_window_batches is not DISABLED

    @property
    def adjusts_max_candidate_windows(self) -> bool:
        return self.max_candidate_windows is not DISABLED


class AdaptiveState:
    """Mutable per-iteration adaptive state."""

    def __init__(self, config: AdaptiveConfig) -> None:
        self.config = config
        self.max_padding_ratio = _initial_value(
            config.max_padding_ratio,
            config.max_padding_ratio_values,
        )
        self.distributed_cost_window_batches = _initial_value(
            config.distributed_cost_window_batches,
            config.distributed_cost_window_values,
        )
        self.max_candidate_windows = _initial_value(
            config.max_candidate_windows,
            config.max_candidate_window_values,
        )
        self._low_padding_streak = 0

    def feedback_for_missing_plan(self) -> list[dict[str, object]]:
        """Loosen padding readiness when no batch is ready."""

        updates: list[dict[str, object]] = []
        if self.max_padding_ratio is not None:
            old_value = self.max_padding_ratio
            new_value = self._larger(
                old_value,
                self.config.max_padding_ratio_values,
            )
            self.max_padding_ratio = new_value
            self._low_padding_streak = 0
            if new_value != old_value:
                updates.append(
                    {
                        "knob": "max_padding_ratio",
                        "reason": "no_ready",
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )
        if self.max_candidate_windows is not None:
            old_value = self.max_candidate_windows
            new_value = self._larger(
                old_value,
                self.config.max_candidate_window_values,
            )
            self.max_candidate_windows = new_value
            if new_value != old_value:
                updates.append(
                    {
                        "knob": "max_candidate_windows",
                        "reason": "no_ready",
                        "old_value": old_value,
                        "new_value": new_value,
                    }
                )
        return updates

    def feedback_for_plan(self, plan: BatchPlan) -> list[dict[str, object]]:
        """Adjust padding readiness from the observed planned batch."""

        if self.max_padding_ratio is None:
            return []
        old_value = self.max_padding_ratio
        if plan.padding_ratio > old_value:
            self.max_padding_ratio = self._larger(
                old_value,
                self.config.max_padding_ratio_values,
            )
            self._low_padding_streak = 0
            if self.max_padding_ratio == old_value:
                return []
            return [
                {
                    "knob": "max_padding_ratio",
                    "reason": "fallback_exceeded_threshold",
                    "old_value": old_value,
                    "new_value": self.max_padding_ratio,
                    "plan_padding_ratio": plan.padding_ratio,
                }
            ]

        if plan.padding_ratio <= old_value * self.config.low_padding_ratio:
            self._low_padding_streak += 1
        elif plan.padding_ratio >= old_value * self.config.high_padding_ratio:
            self._low_padding_streak = 0

        if self._low_padding_streak < self.config.padding_patience:
            return []

        self.max_padding_ratio = self._smaller(
            old_value,
            self.config.max_padding_ratio_values,
        )
        self._low_padding_streak = 0
        if self.max_padding_ratio == old_value:
            return []
        return [
            {
                "knob": "max_padding_ratio",
                "reason": "low_padding_streak",
                "old_value": old_value,
                "new_value": self.max_padding_ratio,
                "plan_padding_ratio": plan.padding_ratio,
            }
        ]

    def update_cost_window(self, stats: CostWindowStats) -> Optional[dict[str, object]]:
        """Adjust the next distributed cost-window size."""

        if self.distributed_cost_window_batches is None:
            return None
        old_value = self.distributed_cost_window_batches
        if (
            stats.source_spread_ratio >= self.config.high_cost_spread_ratio
            and stats.improvement_ratio >= self.config.min_cost_improvement_ratio
        ):
            new_value = self._larger(
                old_value,
                self.config.distributed_cost_window_values,
            )
        elif (
            stats.source_spread_ratio <= self.config.low_cost_spread_ratio
            or stats.improvement_ratio < self.config.min_cost_improvement_ratio
        ):
            new_value = self._smaller(
                old_value,
                self.config.distributed_cost_window_values,
            )
        else:
            new_value = old_value

        self.distributed_cost_window_batches = new_value
        if new_value == old_value:
            return None
        return {
            "knob": "distributed_cost_window_batches",
            "reason": "cost_spread",
            "old_value": old_value,
            "new_value": new_value,
            "source_spread_ratio": stats.source_spread_ratio,
            "improvement_ratio": stats.improvement_ratio,
        }

    @staticmethod
    def _larger(value: Union[int, float], values: tuple[Union[int, float], ...]):
        index = values.index(value)
        return values[min(index + 1, len(values) - 1)]

    @staticmethod
    def _smaller(value: Union[int, float], values: tuple[Union[int, float], ...]):
        index = values.index(value)
        return values[max(index - 1, 0)]


def adaptive_config_fields(
    config: Optional[AdaptiveConfig],
) -> Optional[dict[str, object]]:
    """Return a stable representation for logging and distributed validation."""

    if config is None:
        return None
    return {
        "max_padding_ratio": _field_value(config.max_padding_ratio),
        "distributed_cost_window_batches": _field_value(
            config.distributed_cost_window_batches
        ),
        "max_candidate_windows": _field_value(config.max_candidate_windows),
        "max_padding_ratio_values": config.max_padding_ratio_values,
        "distributed_cost_window_values": config.distributed_cost_window_values,
        "max_candidate_window_values": config.max_candidate_window_values,
        "low_padding_ratio": config.low_padding_ratio,
        "high_padding_ratio": config.high_padding_ratio,
        "padding_patience": config.padding_patience,
        "high_cost_spread_ratio": config.high_cost_spread_ratio,
        "low_cost_spread_ratio": config.low_cost_spread_ratio,
        "min_cost_improvement_ratio": config.min_cost_improvement_ratio,
    }


def _initial_value(
    value: Union[int, float, None, _Disabled],
    values: tuple[Union[int, float], ...],
) -> Optional[Union[int, float]]:
    if value is DISABLED:
        return None
    if value is None:
        return values[min(1, len(values) - 1)]
    return value


def _field_value(value: object) -> object:
    if value is DISABLED:
        return "disabled"
    if value is None:
        return "auto"
    return value


def _int_values(
    values: tuple[int, ...],
    *,
    field_name: str,
    minimum: int,
) -> tuple[int, ...]:
    if not values:
        raise ValueError(f"{field_name} must include at least one value.")
    normalized: list[int] = []
    for value in values:
        parsed = _int_value(value, field_name=field_name, minimum=minimum)
        if parsed not in normalized:
            normalized.append(parsed)
    normalized.sort()
    return tuple(normalized)


def _float_values(
    values: tuple[float, ...],
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> tuple[float, ...]:
    if not values:
        raise ValueError(f"{field_name} must include at least one value.")
    normalized: list[float] = []
    for value in values:
        parsed = _float_value(
            value,
            field_name=field_name,
            minimum=minimum,
            maximum=maximum,
        )
        if parsed not in normalized:
            normalized.append(parsed)
    normalized.sort()
    return tuple(normalized)


def _int_value(value: object, *, field_name: str, minimum: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer.")
    try:
        parsed = operator.index(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise TypeError(f"{field_name} must be an integer.") from error
    if parsed < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}.")
    return parsed


def _float_value(
    value: object,
    *,
    field_name: str,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number.")
    parsed = float(value)
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{field_name} must be between {minimum} and {maximum}.")
    return parsed


def _require_member(
    value: Union[int, float],
    values: tuple[Union[int, float], ...],
    *,
    field_name: str,
) -> None:
    if value not in values:
        raise ValueError(f"{field_name} must be present in its adaptive values.")


__all__ = [
    "DISABLED",
    "AdaptiveConfig",
    "AdaptiveState",
    "CostWindowStats",
    "adaptive_config_fields",
]
