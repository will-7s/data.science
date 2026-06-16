"""
test_playwright.py  —  Sequential browser integration test.

All scenarios run in a single page to avoid Dash/Playwright navigation
timeouts between tests. Each step depends on the previous one.
"""

from pathlib import Path

import pytest

HERE = Path(__file__).parent
FIXTURES = HERE / "fixtures"


@pytest.fixture(scope="session")
def page(browser, app_server):
    """Single page for the whole session."""
    ctx = browser.new_context(viewport={"width": 1280, "height": 900})
    page = ctx.new_page()
    page.goto(app_server, wait_until="domcontentloaded", timeout=30000)
    yield page
    ctx.close()


def _upload(page, csv_name: str):
    page.locator("#upload-data input[type=file]").set_input_files(
        str(FIXTURES / csv_name)
    )


class TestIntegration:
    """Sequential scenarios on a single shared page."""

    def test_01_upload_shows_status(self, page):
        _upload(page, "iris.csv")
        page.wait_for_function(
            "name => document.getElementById('upload-status')"
            ".innerText.toLowerCase().includes(name)",
            arg="iris",
            timeout=25000,
        )
        assert "iris" in page.locator("#upload-status").inner_text().lower()

    def test_02_chart_renders(self, page):
        page.wait_for_timeout(2000)
        page.locator("#univariate-plot .js-plotly-plot").wait_for(
            state="attached", timeout=20000
        )
        traces = page.evaluate(
            """() => {
                const gd = document.querySelector('#univariate-plot .js-plotly-plot');
                return gd && gd.data ? gd.data.length : -1;
            }"""
        )
        assert traces > 0

    def test_03_stats_appear(self, page):
        page.locator("#univariate-stats .stat-row").first.wait_for(
            state="visible", timeout=10000
        )
        text = page.locator("#univariate-stats").inner_text()
        assert "Mean" in text

    def test_04_csv_export_button_visible(self, page):
        assert page.locator("#uni-export-csv").is_visible()

    def test_05_png_export_downloads(self, page):
        with page.expect_download(timeout=15000) as dl:
            page.locator("#uni-export-png").click()
        assert dl.value.suggested_filename.endswith(".png")

    def test_06_theme_toggle(self, page):
        before = page.evaluate(
            "document.documentElement.getAttribute('data-theme')"
        )
        page.locator("#theme-toggle").click()
        page.wait_for_timeout(500)
        after = page.evaluate(
            "document.documentElement.getAttribute('data-theme')"
        )
        assert after != before

    def test_07_theme_cycles_back(self, page):
        before = page.evaluate(
            "document.documentElement.getAttribute('data-theme')"
        )
        page.locator("#theme-toggle").click()
        page.wait_for_timeout(300)
        page.locator("#theme-toggle").click()
        page.wait_for_timeout(300)
        after = page.evaluate(
            "document.documentElement.getAttribute('data-theme')"
        )
        assert after == before

    def test_08_bivariate_tab(self, page):
        page.locator("text=Bivariate Analysis").click()
        page.wait_for_timeout(2000)
        assert page.locator("#bivariate-var1").is_visible()
        assert page.locator("#bivariate-var2").is_visible()

    def test_09_sidebar_collapse(self, page):
        sidebar = page.locator("#uni-sidebar")
        if not sidebar.is_visible():
            return
        page.locator("#uni-collapse-btn").click()
        page.wait_for_timeout(1000)
        assert not sidebar.is_visible()
