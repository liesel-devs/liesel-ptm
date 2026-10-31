"""Stored guide results must change with their execution inputs and artifacts."""

import json
import os
import subprocess
import sys

import psutil
import pytest

from scripts import docs
from scripts.docs import MANIFEST, check, snapshot


@pytest.mark.parametrize(
    "name",
    [
        "liesel_ptm/model.py",
        "docs/guides/example.qmd",
        "docs/_includes/_setup.qmd",
        "docs/filters/alt-text.lua",
        "demo/data/columb.csv",
        "uv.lock",
        "docs/guides/example.md",
        "docs/guides/example_files/figure-gfm/plot.png",
        "docs/_freeze/guides/example/execute-results/md.json",
    ],
)
def test_changed_or_missing_guide_input_or_output_requires_refresh(tmp_path, name):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original")
    manifest = tmp_path / MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # The source identifies the Markdown and figure directory as generated outputs.
    source = tmp_path / "docs/guides/example.qmd"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.touch(exist_ok=True)
    manifest.write_text(json.dumps(snapshot(tmp_path)))
    check(tmp_path)
    path.write_text("changed")
    with pytest.raises(SystemExit, match="render --fresh"):
        check(tmp_path)
    path.unlink()
    with pytest.raises(SystemExit, match="render --fresh"):
        check(tmp_path)


@pytest.mark.skipif(os.name == "nt", reason="The test executable uses a POSIX shebang.")
def test_render_timeout_stops_kernel_descendants(tmp_path, monkeypatch):
    quarto = tmp_path / "quarto"
    quarto.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', "
        '"import os, pathlib, time; '
        "pathlib.Path('kernel.pid').write_text(str(os.getpid())); time.sleep(30)\"])\n"
        "child.wait()\n"
    )
    quarto.chmod(0o755)
    monkeypatch.setattr(docs, "ROOT", tmp_path)
    env = os.environ | {"PATH": str(tmp_path) + os.pathsep + os.environ["PATH"]}
    with pytest.raises(subprocess.TimeoutExpired):
        docs.run_quarto(env, timeout=2)
    kernel_pid = int((tmp_path / "kernel.pid").read_text())
    assert not psutil.pid_exists(kernel_pid)
