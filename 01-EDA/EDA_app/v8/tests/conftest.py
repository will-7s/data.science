import subprocess
import sys
import time
import urllib.request
import urllib.error

import pytest


@pytest.fixture(scope="session")
def _app_context():
    """Start the Dash server once per session."""
    proc = subprocess.Popen(
        [sys.executable, "-u", "eda_app.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=".",
    )
    url = "http://127.0.0.1:8050"
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=3)
            if resp.status == 200:
                break
        except (urllib.error.URLError, OSError):
            time.sleep(0.5)
    else:
        proc.kill()
        pytest.fail("Server did not start within 20 seconds")

    yield url, proc

    proc.terminate()
    proc.wait(timeout=5)


@pytest.fixture(scope="session")
def app_server(_app_context):
    return _app_context[0]
