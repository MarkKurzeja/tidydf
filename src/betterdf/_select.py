"""Monkey-patch pd.DataFrame with a .select() method."""

import pandas as pd


def _select(self: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Select columns by name, raising if any are missing."""
    if not columns:
        raise ValueError("select() requires at least one column name")

    missing = [c for c in columns if c not in self.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    return self[list(columns)]


def patch():
    """Register the select method on pd.DataFrame."""
    pd.DataFrame.select = _select


# Auto-patch on import so `from betterdf import patch` is enough.
patch()
