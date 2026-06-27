"""Compatibility exports for length-budget resolution."""

from __future__ import annotations

from .budget import BudgetResolver

LengthBudgetResolver = BudgetResolver

__all__ = ["BudgetResolver", "LengthBudgetResolver"]
