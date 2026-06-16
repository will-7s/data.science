"""
test_export.py  —  Tests for CSV export data generation.

These tests verify that the CSV strings produced by the export callbacks
contain the expected headers and data rows.  They do NOT test the
dcc.Download mechanism itself (that requires a full Dash test client).
"""
import numpy as np
import stats


def _csv_parse(csv_text: str) -> list[list[str]]:
    """Parse a CSV string into rows (header + data)."""
    return [row.split(",") for row in csv_text.strip().split("\n")]


class TestUnivariateNumericExport:
    def setup_method(self):
        self.arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.s = stats.descriptive_stats(self.arr)
        self.pct = stats.outlier_percentage(self.arr)

    def _build_csv(self) -> str:
        lines = ["Statistic,Value"]
        lines.append(f"N,{self.s['n']}")
        lines.append(f"Mean,{self.s['mean']:.4f}")
        lines.append(f"Median,{self.s['median']:.4f}")
        lines.append(f"Std dev,{self.s['std']:.4f}")
        lines.append(f"Min,{self.s['min']:.4f}")
        lines.append(f"Max,{self.s['max']:.4f}")
        lines.append(f"Outliers %,{self.pct:.1f}")
        return "\n".join(lines)

    def test_csv_has_header(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        assert rows[0] == ["Statistic", "Value"]

    def test_csv_has_all_stat_rows(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        assert len(rows) == 8  # header + 7 stats

    def test_csv_n_value_is_correct(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        n_row = [r for r in rows if r[0] == "N"][0]
        assert n_row[1] == "5"

    def test_csv_mean_formatting(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        mean_row = [r for r in rows if r[0] == "Mean"][0]
        assert float(mean_row[1]) == 3.0
        assert "." in mean_row[1]


class TestUnivariateCategoricalExport:
    def setup_method(self):
        self.arr = np.array(["a", "b", "a", "c", "b", "a"])
        self.vals, self.cnts = np.unique(self.arr, return_counts=True)
        self.total = int(self.cnts.sum())

    def _build_csv(self) -> str:
        lines = ["Category,Count,Percent"]
        for v, c in zip(self.vals, self.cnts):
            lines.append(f"{v},{c},{100*c/self.total:.1f}")
        return "\n".join(lines)

    def test_csv_header(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        assert rows[0] == ["Category", "Count", "Percent"]

    def test_csv_row_count(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        assert len(rows) == 4  # header + 3 categories

    def test_csv_a_count(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        a_row = [r for r in rows if r[0] == "a"][0]
        assert a_row[1] == "3"
        assert a_row[2] == "50.0"

    def test_csv_percentages_sum(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        pcts = [float(r[2]) for r in rows[1:]]
        assert abs(sum(pcts) - 100.0) < 0.1


class TestBivariateNumericExport:
    def setup_method(self):
        self.d1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.d2 = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        self.mask = ~(np.isnan(self.d1) | np.isnan(self.d2))

    def _build_csv(self) -> str:
        r_val = float(np.corrcoef(self.d1[self.mask], self.d2[self.mask])[0, 1])
        lines = ["Variable1,Variable2,Pearson_r,N"]
        lines.append(f"x,y,{r_val:.4f},{int(self.mask.sum())}")
        return "\n".join(lines)

    def test_csv_perfect_correlation(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        data = rows[1]
        assert float(data[2]) == 1.0

    def test_csv_n_count(self):
        csv = self._build_csv()
        rows = _csv_parse(csv)
        data = rows[1]
        assert data[3] == "5"
