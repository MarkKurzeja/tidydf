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
    return pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})


class TestPeek:
    def test_peek_returns_same_dataframe(self, df):
        result = df.peek()
        pd.testing.assert_frame_equal(result, df)

    def test_peek_is_chainable(self, df):
        result = df.peek().select("a")
        assert list(result.columns) == ["a"]

    def test_peek_with_name(self, df, capsys):
        df.peek("my_data")
        captured = capsys.readouterr()
        assert "my_data" in captured.out

    def test_peek_shows_shape(self, df, capsys):
        df.peek()
        captured = capsys.readouterr()
        assert "(5, 2)" in captured.out

    def test_peek_shows_first_two_rows(self, df, capsys):
        df.peek()
        captured = capsys.readouterr()
        # Should contain the first two values
        assert "1" in captured.out
        assert "2" in captured.out
