"""Row-level column creation/mutation for pd.DataFrame."""

import pandas as pd

from betterdf._strict_series import StrictSeries


def _mutate(self: pd.DataFrame, **columns) -> pd.DataFrame:
    """Add or overwrite columns using row-level functions. Returns a new DataFrame.

    Example: df.mutate(bmi=lambda r: r.weight / r.height**2)

    Each keyword argument is a column name mapped to a function that
    receives a StrictSeries (row) and returns a scalar value.
    """
    if not columns:
        raise ValueError("mutate() requires at least one column=fn argument")
    df = self.copy()
    for col_name, fn in columns.items():
        if not callable(fn):
            raise TypeError(f"Value for '{col_name}' must be callable, got {type(fn).__name__}")
        df[col_name] = [fn(StrictSeries(row)) for _, row in self.iterrows()]
    return df
