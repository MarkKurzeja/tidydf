"""betterdf — Better DataFrame operations for pandas."""

import pandas as pd

from betterdf._apply import _papply, _vapply
from betterdf._assert_types import _assert_types
from betterdf._keep import _keep
from betterdf._peek import _peek
from betterdf._select import _deselect, _select
from betterdf._strict_series import _DROP

DROP = _DROP

_METHODS = {
    "select": _select,
    "deselect": _deselect,
    "keep": _keep,
    "vapply": _vapply,
    "papply": _papply,
    "assert_types": _assert_types,
    "peek": _peek,
}


def patch():
    """Monkey-patch all betterdf methods onto pd.DataFrame."""
    for name, fn in _METHODS.items():
        setattr(pd.DataFrame, name, fn)


def unpatch():
    """Remove all betterdf methods from pd.DataFrame."""
    for name in _METHODS:
        if hasattr(pd.DataFrame, name):
            delattr(pd.DataFrame, name)


__all__ = ["patch", "unpatch", "DROP"]
