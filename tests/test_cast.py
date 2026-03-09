import numpy as np
import pandas as pd
import pytest

import tidydf


@pytest.fixture(autouse=True)
def _patch():
    tidydf.patch()
    yield
    tidydf.unpatch()


class TestCast:
    def test_cast_int_to_float(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = df.cast(x=float)
        assert result["x"].dtype == np.float64

    def test_cast_float_to_int(self):
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        result = df.cast(x=int)
        assert result["x"].dtype in (np.int64, np.int32)

    def test_cast_to_str(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = df.cast(x=str)
        assert result["x"].tolist() == ["1", "2", "3"]

    def test_cast_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["3.0", "4.0"]})
        result = df.cast(a=float, b=float)
        assert result["a"].dtype == np.float64
        assert result["b"].dtype == np.float64

    def test_cast_preserves_other_columns(self):
        df = pd.DataFrame({"a": [1], "b": ["hello"]})
        result = df.cast(a=float)
        assert result["b"].tolist() == ["hello"]

    def test_cast_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="missing"):
            df.cast(missing=float)

    def test_cast_no_args_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            df.cast()

    def test_cast_returns_dataframe(self):
        df = pd.DataFrame({"a": [1]})
        result = df.cast(a=float)
        assert isinstance(result, pd.DataFrame)

    def test_cast_does_not_mutate_original(self):
        df = pd.DataFrame({"a": [1, 2]})
        df.cast(a=float)
        assert df["a"].dtype in (np.int64, np.int32)
