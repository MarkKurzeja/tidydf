"""A Series wrapper that raises on access to non-existent columns."""

import pandas as pd

# Sentinel for dropping rows in vapply/papply.
_DROP = object()


class StrictSeries:
    """Wraps a pandas Series to raise clear errors on bad column access."""

    __slots__ = ("_series",)

    def __init__(self, series: pd.Series):
        object.__setattr__(self, "_series", series)

    # --- dot access ---

    def __getattr__(self, name: str):
        series = object.__getattribute__(self, "_series")
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in series.index:
            raise AttributeError(
                f"Column '{name}' not found. Available columns: {list(series.index)}"
            )
        return series[name]

    # --- bracket access ---

    def __getitem__(self, key):
        series = object.__getattribute__(self, "_series")
        if key not in series.index:
            raise KeyError(f"Column '{key}' not found. Available columns: {list(series.index)}")
        return series[key]

    def __setattr__(self, name, value):
        raise AttributeError("StrictSeries is read-only")

    def __repr__(self):
        return f"StrictSeries({object.__getattribute__(self, '_series').to_dict()})"
