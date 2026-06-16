import numpy as np
from decorators import sample_random, sample_stratified, sample_rows


# ── Sampling helpers ──────────────────────────────────────────────────────────

def test_sample_random_reduces_row_count():
    arrays = {"a": np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
              "b": np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])}
    result = sample_random(arrays, n=3)
    assert len(result["a"]) == 3
    assert len(result["b"]) == 3
    assert all(v in arrays["a"] for v in result["a"])


def test_sample_random_preserves_dict_keys():
    arrays = {"x": np.arange(100), "y": np.arange(100, 200)}
    result = sample_random(arrays, n=10)
    assert set(result.keys()) == {"x", "y"}


def test_sample_random_n_equals_length():
    arrays = {"a": np.array([1, 2, 3])}
    result = sample_random(arrays, n=3)
    assert len(result["a"]) == 3
    assert list(result["a"]) == [1, 2, 3]


def test_sample_random_n_greater_than_length():
    arrays = {"a": np.array([1, 2])}
    result = sample_random(arrays, n=10)
    assert len(result["a"]) == 2


def test_sample_random_zero_rows():
    arrays = {"a": np.array([])}
    result = sample_random(arrays, n=5)
    assert len(result["a"]) == 0


def test_sample_stratified_preserves_all_categories():
    rng = np.random.default_rng(42)
    num = rng.normal(size=100)
    cat = np.repeat(["A", "B", "C", "D"], 25)

    s_num, s_cat = sample_stratified(num, cat, n_per_group=5)

    assert len(s_num) == 20
    assert len(s_cat) == 20
    unique, counts = np.unique(s_cat, return_counts=True)
    assert set(unique) == {"A", "B", "C", "D"}
    assert all(c == 5 for c in counts)


def test_sample_stratified_preserves_order_within_group():
    num = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    cat = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

    s_num, s_cat = sample_stratified(num, cat, n_per_group=2)

    assert len(s_num) == 4
    assert np.all(s_num[:2] <= s_num[:2].max())
    assert np.all(s_num[2:] <= s_num[2:].max())


def test_sample_stratified_n_per_group_exceeds_group_size():
    num = np.array([1.0, 2.0, 3.0])
    cat = np.array(["A", "A", "A"])

    s_num, s_cat = sample_stratified(num, cat, n_per_group=10)

    assert len(s_num) == 3
    assert len(s_cat) == 3


def test_sample_rows_preserves_row_alignment():
    a = np.array([1, 2, 3, 4, 5])
    b = np.array([10, 20, 30, 40, 50])

    s_a, s_b = sample_rows([a, b], n=3)

    assert len(s_a) == 3
    assert len(s_b) == 3
    for i in range(3):
        assert s_b[i] == s_a[i] * 10


def test_sample_rows_n_equals_length():
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    s_a, s_b = sample_rows([a, b], n=3)

    assert list(s_a) == [1, 2, 3]
    assert list(s_b) == [4, 5, 6]


def test_sample_rows_n_greater_than_length():
    a = np.array([1, 2])
    result = sample_rows([a], n=10)
    assert len(result[0]) == 2
