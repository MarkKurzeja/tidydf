"""Peek at a DataFrame: print shape and first rows, return self for chaining."""

import pandas as pd


def _is_ipython():
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except ImportError:
        return False


def _peek(self: pd.DataFrame, name: str | None = None) -> pd.DataFrame:
    """Print optional name, shape, and first 2 rows. Returns self for chaining."""
    header = f"[{name}] " if name else ""
    header += f"shape={self.shape}"

    if _is_ipython():
        from IPython.display import display

        print(header)
        display(self.head(2))
    else:
        print(header)
        print(self.head(2).to_string())

    return self
