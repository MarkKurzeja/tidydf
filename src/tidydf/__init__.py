"""tidydf — Better DataFrame operations for pandas."""

import pandas as pd

from tidydf._apply import _papply, _vapply
from tidydf._assert_types import _assert_types
from tidydf._cast import _cast
from tidydf._collapse import _collapse
from tidydf._keep import _keep
from tidydf._mutate import _mutate
from tidydf._peek import _peek
from tidydf._relabel import _relabel
from tidydf._select import _deselect, _select
from tidydf._strict_series import _DROP

DROP = _DROP

_METHODS = {
    "select": _select,
    "deselect": _deselect,
    "keep": _keep,
    "vapply": _vapply,
    "papply": _papply,
    "assert_types": _assert_types,
    "cast": _cast,
    "collapse": _collapse,
    "mutate": _mutate,
    "relabel": _relabel,
    "peek": _peek,
}


def patch():
    """Monkey-patch all tidydf methods onto pd.DataFrame."""
    for name, fn in _METHODS.items():
        setattr(pd.DataFrame, name, fn)


def unpatch():
    """Remove all tidydf methods from pd.DataFrame."""
    for name in _METHODS:
        if hasattr(pd.DataFrame, name):
            delattr(pd.DataFrame, name)


__all__ = ["patch", "unpatch", "DROP"]
