import pandas as pd
import pytest

import betterdf


@pytest.fixture(autouse=True)
def _patch():
    betterdf.patch()
    yield
    betterdf.unpatch()


@pytest.fixture
def df():
    return pd.DataFrame({"name": ["alice", "bob", "carol"], "age": [30, 25, 35]})


class TestKeep:
    def test_keep_filters_rows(self, df):
        result = df.keep(lambda r: r.age > 25)
        assert len(result) == 2
        assert result["name"].tolist() == ["alice", "carol"]

    def test_keep_returns_dataframe(self, df):
        result = df.keep(lambda r: r.age > 25)
        assert isinstance(result, pd.DataFrame)

    def test_keep_preserves_columns(self, df):
        result = df.keep(lambda r: r.age > 25)
        assert list(result.columns) == ["name", "age"]

    def test_keep_all_filtered_out(self, df):
        result = df.keep(lambda r: r.age > 100)
        assert len(result) == 0
        assert list(result.columns) == ["name", "age"]

    def test_keep_none_filtered(self, df):
        result = df.keep(lambda r: r.age > 0)
        assert len(result) == 3

    def test_keep_strict_series_dot_access_bad_column(self, df):
        with pytest.raises(AttributeError, match="no_such_col"):
            df.keep(lambda r: r.no_such_col > 5)

    def test_keep_strict_series_bracket_access_bad_column(self, df):
        with pytest.raises(KeyError, match="no_such_col"):
            df.keep(lambda r: r["no_such_col"] > 5)

    def test_keep_preserves_index(self, df):
        df.index = [10, 20, 30]
        result = df.keep(lambda r: r.age >= 30)
        assert result.index.tolist() == [10, 30]
