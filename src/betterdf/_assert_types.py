"""Dtype assertion for pd.DataFrame."""

import numpy as np
import pandas as pd

# Loose mapping: Python type -> set of numpy/pandas dtype kinds
_TYPE_MAP = {
    float: {"f"},  # float16, float32, float64
    int: {"i", "u"},  # signed and unsigned ints
    str: {"O", "U", "S"},  # object, unicode, bytes (string-like)
    bool: {"b"},
}


def _assert_types(self: pd.DataFrame, **expected) -> pd.DataFrame:
    """Assert column dtypes loosely. Returns self for chaining.

    Example: df.assert_types(price=float, name=str)
    """
    missing = [col for col in expected if col not in self.columns]
    if missing:
        raise KeyError(f"Columns not found: {missing}")

    errors = []
    for col, expected_type in expected.items():
        actual_dtype = self[col].dtype
        kinds = _TYPE_MAP.get(expected_type)

        if kinds is not None:
            if actual_dtype.kind not in kinds:
                errors.append(
                    f"Column '{col}': expected {expected_type.__name__}, got {actual_dtype}"
                )
        elif not np.issubdtype(actual_dtype, expected_type):
            errors.append(f"Column '{col}': expected {expected_type}, got {actual_dtype}")

    if errors:
        raise TypeError("Type assertion failed:\n  " + "\n  ".join(errors))

    return self
