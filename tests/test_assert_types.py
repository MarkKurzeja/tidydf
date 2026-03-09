import numpy as np
import pandas as pd
import pytest

import betterdf


@pytest.fixture(autouse=True)
def _patch():
    betterdf.patch()
    yield
    betterdf.unpatch()


class TestAssertTypes:
    def test_assert_types_passes(self):
        df = pd.DataFrame({"price": [1.0, 2.0], "name": ["a", "b"], "count": [1, 2]})
        # Should not raise
        df.assert_types(price=float, name=str, count=int)

    def test_assert_types_float_matches_float64(self):
        df = pd.DataFrame({"val": np.array([1.0, 2.0], dtype=np.float64)})
        df.assert_types(val=float)

    def test_assert_types_float_matches_float32(self):
        df = pd.DataFrame({"val": np.array([1.0, 2.0], dtype=np.float32)})
        df.assert_types(val=float)

    def test_assert_types_int_matches_int32(self):
        df = pd.DataFrame({"val": np.array([1, 2], dtype=np.int32)})
        df.assert_types(val=int)

    def test_assert_types_int_matches_int64(self):
        df = pd.DataFrame({"val": np.array([1, 2], dtype=np.int64)})
        df.assert_types(val=int)

    def test_assert_types_str_matches_object(self):
        df = pd.DataFrame({"name": ["a", "b"]})
        df.assert_types(name=str)

    def test_assert_types_wrong_type_raises(self):
        df = pd.DataFrame({"price": [1.0, 2.0]})
        with pytest.raises(TypeError):
            df.assert_types(price=int)

    def test_assert_types_missing_column_raises(self):
        df = pd.DataFrame({"price": [1.0]})
        with pytest.raises(KeyError, match="missing_col"):
            df.assert_types(missing_col=float)

    def test_assert_types_multiple_wrong(self):
        df = pd.DataFrame({"a": [1.0], "b": ["x"]})
        with pytest.raises(TypeError):
            df.assert_types(a=str, b=float)

    def test_assert_types_returns_dataframe(self):
        df = pd.DataFrame({"x": [1]})
        result = df.assert_types(x=int)
        assert isinstance(result, pd.DataFrame)
        pd.testing.assert_frame_equal(result, df)
