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
    return pd.DataFrame(
        {
            "region": ["US", "US", "EU", "EU"],
            "year": [2024, 2024, 2024, 2025],
            "product": ["A", "B", "C", "D"],
            "revenue": [100, 200, 300, 400],
        }
    )


class TestCollapse:
    def test_collapse_basic(self, df):
        result = df.collapse("region", "year")
        assert "records" in result.columns
        assert list(result.columns) == ["region", "year", "records"]

    def test_collapse_groups_correctly(self, df):
        result = df.collapse("region", "year")
        # US 2024 should have 2 records, EU 2024 has 1, EU 2025 has 1
        assert len(result) == 3
        us_2024 = result[(result["region"] == "US") & (result["year"] == 2024)]
        records = us_2024["records"].iloc[0]
        assert len(records) == 2
        assert {"product": "A", "revenue": 100} in records
        assert {"product": "B", "revenue": 200} in records

    def test_collapse_custom_name(self, df):
        result = df.collapse("region", "year", name="items")
        assert "items" in result.columns
        assert "records" not in result.columns

    def test_collapse_single_group_col(self, df):
        result = df.collapse("region")
        assert list(result.columns) == ["region", "records"]
        us_rows = result.loc[result["region"] == "US", "records"].values[0]
        assert len(us_rows) == 2

    def test_collapse_records_exclude_group_cols(self, df):
        result = df.collapse("region", "year")
        for records in result["records"]:
            for record in records:
                assert "region" not in record
                assert "year" not in record

    def test_collapse_no_args_raises(self, df):
        with pytest.raises(ValueError):
            df.collapse()

    def test_collapse_missing_column_raises(self, df):
        with pytest.raises(KeyError):
            df.collapse("nonexistent")

    def test_collapse_returns_dataframe(self, df):
        result = df.collapse("region")
        assert isinstance(result, pd.DataFrame)
