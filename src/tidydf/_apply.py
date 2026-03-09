"""Row-wise apply (serial and parallel) for pd.DataFrame."""

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from tqdm import tqdm

from tidydf._strict_series import _DROP, StrictSeries


def _get_func_name(fn) -> str:
    return getattr(fn, "__name__", getattr(fn, "__qualname__", str(fn)))


def _apply_one(fn, row):
    return fn(StrictSeries(row))


def _unwrap(result):
    """Unwrap StrictSeries back to pd.Series."""
    if isinstance(result, StrictSeries):
        return object.__getattribute__(result, "_series")
    return result


def _collect_results(results, original_df):
    """Collect apply results, filtering out DROP sentinels."""
    kept = []
    kept_indices = []
    for idx, result in zip(original_df.index, results):
        if result is _DROP:
            continue
        kept.append(_unwrap(result))
        kept_indices.append(idx)

    if not kept:
        return pd.DataFrame(columns=original_df.columns)

    out = pd.DataFrame(kept, index=kept_indices)
    return out


def _vapply(self: pd.DataFrame, fn) -> pd.DataFrame:
    """Row-wise apply with tqdm progress. fn receives StrictSeries, returns Series.
    Return DROP to drop a row."""
    desc = _get_func_name(fn)
    results = []
    for _, row in tqdm(self.iterrows(), total=len(self), desc=desc):
        results.append(fn(StrictSeries(row)))
    return _collect_results(results, self)


def _papply(self: pd.DataFrame, fn, n_workers: int | None = None) -> pd.DataFrame:
    """Parallel row-wise apply with tqdm. fn receives StrictSeries, returns Series.
    Return DROP to drop a row. Row order is preserved."""
    if n_workers is None:
        n_workers = os.cpu_count() or 1

    desc = _get_func_name(fn)
    rows = [row for _, row in self.iterrows()]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = [pool.submit(_apply_one, fn, row) for row in rows]
        results = []
        for future in tqdm(futures, total=len(futures), desc=desc):
            results.append(future.result())

    return _collect_results(results, self)
