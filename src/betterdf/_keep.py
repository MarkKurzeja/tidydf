"""Row-wise filtering for pd.DataFrame."""

import pandas as pd

from betterdf._strict_series import StrictSeries


def _keep(self: pd.DataFrame, fn) -> pd.DataFrame:
    """Keep rows where fn(StrictSeries(row)) is truthy."""
    mask = [bool(fn(StrictSeries(row))) for _, row in self.iterrows()]
    return self.loc[mask]
