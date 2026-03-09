"""Relabel (recode) values in DataFrame columns."""

import pandas as pd


def _relabel(self: pd.DataFrame, **mappings) -> pd.DataFrame:
    """Recode values in one or more columns. Returns a new DataFrame.

    Example: df.relabel(region={"N": "North", "S": "South"})

    Each keyword argument is a column name mapped to a dict of
    {old_value: new_value}. Values not in the mapping are left unchanged.
    """
    if not mappings:
        raise ValueError("relabel() requires at least one column=mapping argument")
    missing = [col for col in mappings if col not in self.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    df = self.copy()
    for col, mapping in mappings.items():
        if not isinstance(mapping, dict):
            raise TypeError(f"Mapping for '{col}' must be a dict, got {type(mapping).__name__}")
        df[col] = df[col].replace(mapping)
    return df
