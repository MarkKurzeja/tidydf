import pandas as pd
import pytest

from betterdf import patch  # noqa: F401 — registers the accessor


@pytest.fixture
def df():
    return pd.DataFrame({"taco": [1, 2], "nacho": [3, 4], "burrito": [5, 6]})


class TestSelect:
    def test_select_single_column(self, df):
        result = df.select("taco")
        assert list(result.columns) == ["taco"]
        assert result["taco"].tolist() == [1, 2]

    def test_select_multiple_columns(self, df):
        result = df.select("taco", "nacho")
        assert list(result.columns) == ["taco", "nacho"]

    def test_select_preserves_data(self, df):
        result = df.select("nacho", "burrito")
        assert result["nacho"].tolist() == [3, 4]
        assert result["burrito"].tolist() == [5, 6]

    def test_select_returns_dataframe(self, df):
        result = df.select("taco")
        assert isinstance(result, pd.DataFrame)

    def test_select_missing_column_raises(self, df):
        with pytest.raises(KeyError, match="salsa"):
            df.select("taco", "salsa")

    def test_select_all_missing_raises(self, df):
        with pytest.raises(KeyError):
            df.select("salsa", "queso")

    def test_select_preserves_index(self, df):
        df.index = [10, 20]
        result = df.select("taco")
        assert result.index.tolist() == [10, 20]

    def test_select_no_args_raises(self, df):
        with pytest.raises(ValueError):
            df.select()

    def test_select_column_order_matches_args(self, df):
        result = df.select("burrito", "taco")
        assert list(result.columns) == ["burrito", "taco"]
