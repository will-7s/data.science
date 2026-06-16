"""
test_integration.py  —  Integration tests for the EDA app.

Starts the Dash server in a subprocess and verifies that key endpoints
respond correctly.  This catches import errors, layout crashes, and
basic connectivity issues.
"""
import time
import sys
import subprocess
import urllib.request
import urllib.error


def _wait_for_server(url: str, timeout: float = 15.0) -> bool:
    """Poll `url` until it returns 200 or `timeout` expires."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=5)
            if resp.status == 200:
                return True
        except (urllib.error.URLError, ConnectionResetError, OSError):
            time.sleep(0.5)
    return False


class TestAppStarts:
    """Start the real server once per test session and verify endpoints."""

    _proc: subprocess.Popen | None = None
    _base = "http://127.0.0.1:8050"

    @classmethod
    def setup_class(cls):
        cls._proc = subprocess.Popen(
            [sys.executable, "-u", "eda_app.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=".",
        )
        if not _wait_for_server(cls._base):
            cls._proc.kill()
            raise RuntimeError("Server did not start within 15 seconds")

    @classmethod
    def teardown_class(cls):
        if cls._proc:
            cls._proc.terminate()
            cls._proc.wait(timeout=5)

    def test_index_returns_200(self):
        resp = urllib.request.urlopen(f"{self._base}/")
        assert resp.status == 200

    def test_index_has_dash_entry_point(self):
        resp = urllib.request.urlopen(f"{self._base}/")
        html = resp.read().decode("utf-8")
        assert "react-entry-point" in html
        assert "_dash-config" in html

    def test_favicon_served(self):
        try:
            resp = urllib.request.urlopen(f"{self._base}/_favicon.ico")
            assert resp.status in (200, 204)
        except urllib.error.HTTPError as e:
            assert e.code in (404,), f"Unexpected favicon error: {e.code}"

    def test_dash_dependencies_route(self):
        resp = urllib.request.urlopen(f"{self._base}/_dash-dependencies")
        assert resp.status == 200
        data = resp.read().decode("utf-8")
        assert len(data) > 100

    def test_dash_layout_route_returns_initial_props(self):
        resp = urllib.request.urlopen(f"{self._base}/_dash-layout")
        assert resp.status == 200
        data = resp.read().decode("utf-8")
        assert "theme-toggle" in data
        assert "upload-data" in data

    def test_theme_toggle_in_initial_layout(self):
        resp = urllib.request.urlopen(f"{self._base}/_dash-layout")
        data = resp.read().decode("utf-8")
        assert "theme-store" in data
        assert "theme-init" in data

    def test_stylesheet_served(self):
        resp = urllib.request.urlopen(f"{self._base}/assets/style.css")
        assert resp.status == 200
        css = resp.read().decode("utf-8")
        assert len(css) > 500
        assert "sidebar-scroll" in css
