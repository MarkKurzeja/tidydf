import pandas as pd
import pytest

import tidydf


@pytest.fixture(autouse=True)
def _patch():
    tidydf.patch()
    yield
    tidydf.unpatch()


class TestMutate:
    def test_mutate_add_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = df.mutate(doubled=lambda r: r.x * 2)
        assert result["doubled"].tolist() == [2, 4, 6]
        assert result["x"].tolist() == [1, 2, 3]

    def test_mutate_overwrite_column(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = df.mutate(x=lambda r: r.x + 10)
        assert result["x"].tolist() == [11, 12, 13]

    def test_mutate_multiple_columns(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = df.mutate(
            sum_ab=lambda r: r.a + r.b,
            prod_ab=lambda r: r.a * r.b,
        )
        assert result["sum_ab"].tolist() == [4, 6]
        assert result["prod_ab"].tolist() == [3, 8]

    def test_mutate_uses_strict_series(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(AttributeError):
            df.mutate(y=lambda r: r.nonexistent)

    def test_mutate_no_args_raises(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError):
            df.mutate()

    def test_mutate_non_callable_raises(self):
        df = pd.DataFrame({"x": [1]})
        with pytest.raises(TypeError, match="callable"):
            df.mutate(y=42)

    def test_mutate_does_not_mutate_original(self):
        df = pd.DataFrame({"x": [1, 2]})
        df.mutate(y=lambda r: r.x * 2)
        assert "y" not in df.columns

    def test_mutate_returns_dataframe(self):
        df = pd.DataFrame({"x": [1]})
        result = df.mutate(y=lambda r: r.x)
        assert isinstance(result, pd.DataFrame)

    def test_mutate_string_manipulation(self):
        df = pd.DataFrame({"name": ["alice", "bob"]})
        result = df.mutate(upper=lambda r: r.name.upper())
        assert result["upper"].tolist() == ["ALICE", "BOB"]
