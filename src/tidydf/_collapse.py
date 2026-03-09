"""Group-by and collect remaining columns into list-of-dicts."""

import pandas as pd


def _collapse(self: pd.DataFrame, *group_cols: str, name: str = "records") -> pd.DataFrame:
    """Group by group_cols, collect remaining columns as a list of dicts per group."""
    if not group_cols:
        raise ValueError("collapse() requires at least one column name")
    missing = [c for c in group_cols if c not in self.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    other_cols = [c for c in self.columns if c not in group_cols]

    groups = self.groupby(list(group_cols), sort=True)
    rows = []
    for keys, group in groups:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row[name] = group[other_cols].to_dict("records")
        rows.append(row)

    return pd.DataFrame(rows)
