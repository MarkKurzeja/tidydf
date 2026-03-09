"""Select and deselect columns on pd.DataFrame."""

import pandas as pd


def _select(self: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Select columns by name, raising if any are missing."""
    if not columns:
        raise ValueError("select() requires at least one column name")
    missing = [c for c in columns if c not in self.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return self[list(columns)]


def _deselect(self: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Drop columns by name, raising if any are missing."""
    if not columns:
        raise ValueError("deselect() requires at least one column name")
    missing = [c for c in columns if c not in self.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return self.drop(columns=list(columns))
