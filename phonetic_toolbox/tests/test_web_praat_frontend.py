from pathlib import Path
import shutil
import subprocess

import pytest


def test_web_editor_async_regressions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed to run the web editor behavior tests")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("web_praat_async.test.cjs"))],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
