import numpy as np
import store
from store import _classify_column, _is_date_string


def _make_dataset() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 1000
    return {
        "age":       rng.normal(40, 15, n).astype(np.float64),
        "income":    np.concatenate([rng.exponential(50, n - 10), [np.nan] * 10]),
        "city":      rng.choice(["Paris", "Lyon", "Marseille"], n),
        "score":     rng.integers(0, 101, n).astype(np.float64),
        "group":     rng.choice(["A", "B", "C", "D"], n),
        "joined":    rng.choice(["2024-01-01", "2024-06-15", "2025-03-10"], n),
    }


class TestClassification:
    def test_date_string_detection(self):
        assert _is_date_string("2023-01-15") is True
        assert _is_date_string("01/15/2023") is True
        assert _is_date_string("15/Jan/2023") is True
        assert _is_date_string("hello") is False
        assert _is_date_string("") is False
        assert _is_date_string("abc123") is False

    def test_classify_temporal_from_strings(self):
        arr = np.array(["2023-01-01", "2023-06-15", "2024-03-10"], dtype=object)
        assert _classify_column(arr) == "temporal"

    def test_classify_temporal_mixed_with_non_dates(self):
        arr = np.array(["2023-01-01", "hello", "2024-03-10"], dtype=object)
        assert _classify_column(arr) == "temporal"

    def test_classify_normal_string_as_categorical(self):
        arr = np.array(["apple", "banana", "cherry"], dtype=object)
        assert _classify_column(arr) == "categorical"

    def test_classify_numeric_float(self):
        arr = np.array([1.5, 2.7, 3.2, 4.9])
        assert _classify_column(arr) == "numeric"

    def test_classify_few_integers_as_categorical(self):
        arr = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
        assert _classify_column(arr) == "categorical"

    def test_joined_column_is_temporal_in_dataset(self):
        ds = _make_dataset()
        store.reset(ds, source="test.csv")
        assert store.col_meta["joined"] == "temporal"

    def test_temporal_in_col_types_metadata(self):
        ds = _make_dataset()
        store.reset(ds, source="test.csv")
        m = store.get_metadata()
        assert "temporal" in m["col_types"]
        assert m["col_types"]["temporal"] >= 1


class TestMetadataCache:
    def setup_method(self):
        ds = _make_dataset()
        store.reset(ds, source="test.csv")

    def test_metadata_shape(self):
        m = store.get_metadata()
        assert m["n_rows"] == 1000
        assert m["n_cols"] == 6
        assert "numeric" in m["col_types"]
        assert "categorical" in m["col_types"]
        assert isinstance(m["memory_bytes"], int)
        assert m["memory_bytes"] > 0
        assert m["source"] == "test.csv"

    def test_col_types_distribution(self):
        m = store.get_metadata()
        num = sum(1 for v in store.col_meta.values() if v == "numeric")
        cat = sum(1 for v in store.col_meta.values() if v == "categorical")
        assert m["col_types"].get("numeric", 0) == num
        assert m["col_types"].get("categorical", 0) == cat

    def test_metadata_cache_returns_same_object(self):
        m1 = store.get_metadata()
        m2 = store.get_metadata()
        assert m1 is m2

    def test_metadata_cache_cleared_on_reset(self):
        m1 = store.get_metadata()
        store.reset(_make_dataset(), source="other.csv")
        m2 = store.get_metadata()
        assert m1 is not m2
        assert m2["source"] == "other.csv"

    def test_top_missing_contains_income(self):
        m = store.get_metadata()
        cols_with_na = [t["col"] for t in m["top_missing"]]
        assert "income" in cols_with_na

    def test_top_missing_ordered_by_descending(self):
        m = store.get_metadata()
        counts = [t["n_missing"] for t in m["top_missing"]]
        assert counts == sorted(counts, reverse=True)

    def test_empty_dataset_returns_empty_metadata(self):
        store.reset({}, source="empty.csv")
        m = store.get_metadata()
        assert m["n_rows"] == 0
        assert m["n_cols"] == 0
        assert m["top_missing"] == []

    def test_no_missing_values(self):
        ds = {"a": np.array([1.0, 2.0, 3.0]), "b": np.array(["x", "y", "z"])}
        store.reset(ds, source="no_na.csv")
        m = store.get_metadata()
        assert m["top_missing"] == []
