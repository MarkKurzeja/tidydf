"""Column type casting for pd.DataFrame."""

import pandas as pd


def _cast(self: pd.DataFrame, **types) -> pd.DataFrame:
    """Cast columns to specified types. Returns a new DataFrame.

    Example: df.cast(price=float, count=int)
    """
    if not types:
        raise ValueError("cast() requires at least one column=type argument")
    missing = [col for col in types if col not in self.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")
    return self.astype(types)
