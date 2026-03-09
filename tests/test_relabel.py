import pandas as pd
import pytest

import tidydf


@pytest.fixture(autouse=True)
def _patch():
    tidydf.patch()
    yield
    tidydf.unpatch()


class TestRelabel:
    def test_relabel_basic(self):
        df = pd.DataFrame({"region": ["N", "S", "E"]})
        result = df.relabel(region={"N": "North", "S": "South"})
        assert result["region"].tolist() == ["North", "South", "E"]

    def test_relabel_multiple_columns(self):
        df = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
        result = df.relabel(a={"x": "X"}, b={1: 100})
        assert result["a"].tolist() == ["X", "y"]
        assert result["b"].tolist() == [100, 2]

    def test_relabel_missing_column_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(KeyError, match="missing"):
            df.relabel(missing={"x": "y"})

    def test_relabel_no_args_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError):
            df.relabel()

    def test_relabel_non_dict_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(TypeError, match="dict"):
            df.relabel(a="not a dict")

    def test_relabel_does_not_mutate_original(self):
        df = pd.DataFrame({"a": ["x", "y"]})
        df.relabel(a={"x": "X"})
        assert df["a"].tolist() == ["x", "y"]

    def test_relabel_returns_dataframe(self):
        df = pd.DataFrame({"a": ["x"]})
        result = df.relabel(a={"x": "X"})
        assert isinstance(result, pd.DataFrame)

    def test_relabel_unmapped_values_unchanged(self):
        df = pd.DataFrame({"a": ["x", "y", "z"]})
        result = df.relabel(a={"x": "X"})
        assert result["a"].tolist() == ["X", "y", "z"]
