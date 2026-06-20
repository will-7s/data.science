from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.store import (
    _classify_column,
    _is_binary,
    _is_datetime_col,
    _is_id_column,
    reset,
    schema,
    all_cols,
    num_cols,
    cat_cols,
)


class TestClassifyColumn:
    def test_binary_int_as_numeric(self):
        arr = np.array([0, 1, 0, 1, 0])
        assert _classify_column(arr) == "numeric"

    def test_low_cardinality_int_as_categorical(self):
        arr = np.array([1, 2, 3, 1, 2, 3, 1, 2])
        assert _classify_column(arr) == "categorical"

    def test_high_cardinality_int_as_numeric(self):
        arr = np.arange(100)
        assert _classify_column(arr) == "numeric"

    def test_string_as_categorical(self):
        arr = np.array(["a", "b", "c"])
        assert _classify_column(arr) == "categorical"

    def test_float_with_few_values_as_categorical(self):
        arr = np.array([0.0, 1.0, 0.0, 1.0, np.nan])
        assert _classify_column(arr) == "categorical"

    def test_float_with_many_values_as_numeric(self):
        arr = np.linspace(0, 1, 20)
        assert _classify_column(arr) == "numeric"


class TestIsBinary:
    def test_true_for_two_values(self):
        assert _is_binary(np.array([0, 1, 0, 1, 0])) is True

    def test_false_for_one_value(self):
        assert _is_binary(np.array([0, 0, 0])) is False

    def test_false_for_three_values(self):
        assert _is_binary(np.array([0, 1, 2])) is False

    def test_handles_nan(self):
        assert _is_binary(np.array([0.0, 1.0, np.nan])) is True


class TestIsDatetimeCol:
    def test_native_datetime(self):
        arr = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        assert _is_datetime_col(arr) is True

    def test_string_datetime(self):
        arr = np.array(["2024-01-01", "2024-01-02", "2024-01-03"])
        assert _is_datetime_col(arr) is True

    def test_non_datetime_string(self):
        arr = np.array(["hello", "world", "foo"])
        assert _is_datetime_col(arr) is False

    def test_numeric(self):
        arr = np.array([1, 2, 3])
        assert _is_datetime_col(arr) is False


class TestIsIdColumn:
    def test_keyword_match(self):
        arr = np.array([1, 2, 3])
        assert _is_id_column(arr, "user_id") is True

    def test_high_cardinality(self):
        arr = np.arange(1000)
        assert _is_id_column(arr, "some_field") is True

    def test_low_cardinality_no_keyword(self):
        arr = np.array([1, 2, 3, 1, 2, 3])
        assert _is_id_column(arr, "color") is False

    def test_datetime_excluded(self):
        arr = np.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]")
        assert _is_id_column(arr, "timestamp") is False


class TestReset:
    def test_populates_all_fields(self):
        df = pd.DataFrame({
            "user_id": [1, 2, 3],
            "group": ["A", "B", "A"],
            "converted": [0, 1, 0],
        })
        reset(df)
        assert len(all_cols) > 0
        assert "converted" in num_cols
        assert "group" in cat_cols

    def test_clears_previous_state(self):
        df1 = pd.DataFrame({"a": [1, 2, 3, 4]})
        reset(df1)
        assert len(all_cols) > 0
        df2 = pd.DataFrame({"z": [5, 6, 7, 8]})
        reset(df2)
        assert "z" in all_cols
        assert "a" not in all_cols
