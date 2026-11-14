"""Stored guide results must be refreshed when, and only when, a guide changes."""

import hashlib
import json
import os
import subprocess
import sys

import psutil
import pytest

from scripts import docs
from scripts.docs import check


def write_guide(root, name="example", source="original"):
    """Write a guide source with frozen results that match it, as Quarto does."""
    qmd = root / f"docs/guides/{name}.qmd"
    qmd.parent.mkdir(parents=True, exist_ok=True)
    qmd.write_text(source)
    frozen = root / f"docs/_freeze/guides/{name}/execute-results/md.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.md5(qmd.read_bytes()).hexdigest()
    frozen.write_text(json.dumps({"hash": digest, "result": {}}))
    return qmd, frozen


def test_unchanged_guides_pass(tmp_path):
    write_guide(tmp_path, "first")
    write_guide(tmp_path, "second")
    check(tmp_path)


def test_edited_guide_requires_refresh(tmp_path):
    write_guide(tmp_path, "unchanged")
    qmd, _ = write_guide(tmp_path, "edited")
    qmd.write_text("changed")
    with pytest.raises(SystemExit, match="docs/guides/edited.qmd") as err:
        check(tmp_path)
    assert "unchanged" not in str(err.value)
    assert "scripts/docs.py render" in str(err.value)


def test_guide_without_frozen_results_requires_refresh(tmp_path):
    _, frozen = write_guide(tmp_path)
    frozen.unlink()
    with pytest.raises(SystemExit, match="docs/guides/example.qmd"):
        check(tmp_path)


@pytest.mark.parametrize(
    "name",
    [
        "src/liesel_ptm/model.py",
        "uv.lock",
        "pyproject.toml",
        "docs/_quarto.yml",
        "docs/filters/alt-text.lua",
        "demo/data/columb.csv",
        "scripts/docs.py",
        "docs/guides/example.md",
        "docs/guides/example_files/figure-commonmark/plot.png",
    ],
)
def test_changes_outside_guide_sources_pass(tmp_path, name):
    # Library, dependency and tooling changes rarely change guide results, and a
    # spurious refresh rewrites every figure. Only guide edits require one.
    write_guide(tmp_path)
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("original")
    check(tmp_path)
    path.write_text("changed")
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
