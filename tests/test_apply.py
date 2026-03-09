import pandas as pd
import pytest

import betterdf
from betterdf import DROP


@pytest.fixture(autouse=True)
def _patch():
    betterdf.patch()
    yield
    betterdf.unpatch()


@pytest.fixture
def df():
    return pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})


class TestVapply:
    def test_vapply_returns_dataframe(self, df):
        result = df.vapply(lambda r: r)
        assert isinstance(result, pd.DataFrame)

    def test_vapply_identity(self, df):
        result = df.vapply(lambda r: r)
        pd.testing.assert_frame_equal(result, df)

    def test_vapply_transform(self, df):
        result = df.vapply(lambda r: pd.Series({"x": r.x * 2, "y": r.y + 1}))
        assert result["x"].tolist() == [2, 4, 6]
        assert result["y"].tolist() == [11, 21, 31]

    def test_vapply_drop_sentinel(self, df):
        result = df.vapply(lambda r: DROP if r.x == 2 else r)
        assert len(result) == 2
        assert result["x"].tolist() == [1, 3]

    def test_vapply_drop_all(self, df):
        result = df.vapply(lambda r: DROP)
        assert len(result) == 0

    def test_vapply_preserves_row_order(self, df):
        result = df.vapply(lambda r: r)
        assert result["x"].tolist() == [1, 2, 3]

    def test_vapply_strict_series_bad_dot_access(self, df):
        with pytest.raises(AttributeError, match="bad_col"):
            df.vapply(lambda r: pd.Series({"out": r.bad_col}))

    def test_vapply_strict_series_bad_bracket_access(self, df):
        with pytest.raises(KeyError, match="bad_col"):
            df.vapply(lambda r: pd.Series({"out": r["bad_col"]}))

    def test_vapply_new_columns(self, df):
        result = df.vapply(lambda r: pd.Series({"sum": r.x + r.y}))
        assert list(result.columns) == ["sum"]
        assert result["sum"].tolist() == [11, 22, 33]


class TestPapply:
    def test_papply_identity(self, df):
        result = df.papply(lambda r: r)
        pd.testing.assert_frame_equal(result, df)

    def test_papply_transform(self, df):
        result = df.papply(lambda r: pd.Series({"x": r.x * 2, "y": r.y + 1}))
        assert result["x"].tolist() == [2, 4, 6]
        assert result["y"].tolist() == [11, 21, 31]

    def test_papply_drop_sentinel(self, df):
        result = df.papply(lambda r: DROP if r.x == 2 else r)
        assert len(result) == 2
        assert result["x"].tolist() == [1, 3]

    def test_papply_preserves_row_order(self, df):
        result = df.papply(lambda r: r)
        assert result["x"].tolist() == [1, 2, 3]

    def test_papply_custom_n_workers(self, df):
        result = df.papply(lambda r: r, n_workers=2)
        pd.testing.assert_frame_equal(result, df)

    def test_papply_strict_series_bad_dot_access(self, df):
        with pytest.raises(AttributeError, match="bad_col"):
            df.papply(lambda r: pd.Series({"out": r.bad_col}))
