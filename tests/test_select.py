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


class TestDeselect:
    def test_deselect_single_column(self, df):
        result = df.deselect("burrito")
        assert list(result.columns) == ["taco", "nacho"]

    def test_deselect_multiple_columns(self, df):
        result = df.deselect("taco", "burrito")
        assert list(result.columns) == ["nacho"]

    def test_deselect_preserves_data(self, df):
        result = df.deselect("burrito")
        assert result["taco"].tolist() == [1, 2]
        assert result["nacho"].tolist() == [3, 4]

    def test_deselect_missing_column_raises(self, df):
        with pytest.raises(KeyError, match="salsa"):
            df.deselect("salsa")

    def test_deselect_no_args_raises(self, df):
        with pytest.raises(ValueError):
            df.deselect()

    def test_deselect_preserves_index(self, df):
        df.index = [10, 20]
        result = df.deselect("burrito")
        assert result.index.tolist() == [10, 20]


class TestPatchUnpatch:
    def test_unpatch_removes_methods(self):
        betterdf.unpatch()
        assert not hasattr(pd.DataFrame, "select")
        assert not hasattr(pd.DataFrame, "deselect")
        assert not hasattr(pd.DataFrame, "keep")
        assert not hasattr(pd.DataFrame, "vapply")
        assert not hasattr(pd.DataFrame, "papply")
        assert not hasattr(pd.DataFrame, "peek")
        assert not hasattr(pd.DataFrame, "assert_types")
        # re-patch for other tests
        betterdf.patch()

    def test_patch_is_idempotent(self):
        betterdf.patch()
        betterdf.patch()
        df = pd.DataFrame({"a": [1]})
        assert list(df.select("a").columns) == ["a"]
