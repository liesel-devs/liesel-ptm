"""Render Quarto guides and reject stale stored outputs before site builds."""

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "docs/guide-outputs.json"


def snapshot(root: Path) -> dict[str, str]:
    """Hash execution inputs and artifacts, including shared guide fragments."""
    paths = {
        root / "uv.lock",
        root / "pyproject.toml",
        root / "docs/_quarto.yml",
        root / "scripts/docs.py",
        root / "demo/data/README.md",
        *root.glob("src/liesel_ptm/**/*.py"),
        *root.glob("demo/reference_*.py"),
        *root.glob("demo/data/*.csv"),
        *root.glob("docs/**/*.qmd"),
        *root.glob("docs/filters/*.lua"),
        *root.glob("docs/_freeze/**/*"),
    }
    for source in root.glob("docs/guides/*.qmd"):
        paths.add(source.with_suffix(".md"))
        paths.update(source.with_name(source.stem + "_files").rglob("*"))
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
        if path.is_file() and ".quarto" not in path.parts
    }


def check(root: Path) -> None:
    """Require exactly the inputs and outputs from the last successful fresh render."""
    path = root / MANIFEST
    if not path.exists() or json.loads(path.read_text()) != snapshot(root):
        raise SystemExit(
            "Guide inputs or outputs changed. Refresh them with:\n"
            "uv run --locked --group docs python scripts/docs.py render --fresh"
        )


def run_quarto(env: dict[str, str], *, timeout: float = 600) -> None:
    """Wait in the foreground; stop this render's descendants if it is interrupted."""
    command = ["quarto", "render", "docs"]
    with subprocess.Popen(command, cwd=ROOT, env=env) as process:
        parent = psutil.Process(process.pid)
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            with contextlib.suppress(psutil.NoSuchProcess):
                children = parent.children(recursive=True)
                for child in reversed(children):
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.terminate()
                _, alive = psutil.wait_procs(children, timeout=10)
                for child in alive:
                    with contextlib.suppress(psutil.NoSuchProcess):
                        child.kill()
                psutil.wait_procs(alive, timeout=10)
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "render"))
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if not args.fresh:
        check(ROOT)
    if args.action == "check":
        if args.fresh:
            parser.error("--fresh requires render")
        return
    if args.fresh:
        (ROOT / MANIFEST).unlink(missing_ok=True)
        shutil.rmtree(ROOT / "docs/_freeze", ignore_errors=True)
        for path in (ROOT / "docs/guides").glob("*_files"):
            shutil.rmtree(path)
    env = os.environ | {"QUARTO_PYTHON": sys.executable, "MPLBACKEND": "Agg"}
    run_quarto(env)
    # Match the repository's text-file hooks before hashing generated JSON.
    for path in (ROOT / "docs/_freeze").rglob("*.json"):
        path.write_text(path.read_text().rstrip() + "\n")
    if args.fresh:
        (ROOT / MANIFEST).write_text(json.dumps(snapshot(ROOT), indent=2) + "\n")


if __name__ == "__main__":
    main()
