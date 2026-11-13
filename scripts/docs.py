"""Render Quarto guides and reject stored results of edited guides."""

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


def stale_guides(root: Path) -> list[str]:
    """Return guides whose source changed since their stored results were rendered.

    Quarto records an MD5 hash of each guide source in its frozen results. Only a
    mismatch, or missing results, counts as stale. Library, dependency and tooling
    changes do not: they rarely change guide results, and a spurious refresh
    rewrites every figure, which grows the repository.
    """
    stale = []
    for source in sorted(root.glob("docs/guides/*.qmd")):
        frozen = root / "docs/_freeze/guides" / source.stem / "execute-results/md.json"
        digest = hashlib.md5(source.read_bytes()).hexdigest()
        if not frozen.exists() or json.loads(frozen.read_text())["hash"] != digest:
            stale.append(source.relative_to(root).as_posix())
    return stale


def check(root: Path) -> None:
    """Require stored results for the current version of every guide."""
    stale = stale_guides(root)
    if stale:
        raise SystemExit(
            "Stored results are out of date for: "
            + ", ".join(stale)
            + "\nRe-render the changed guides with:\n"
            "uv run --locked --group docs python scripts/docs.py render"
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
    parser.add_argument(
        "--fresh", action="store_true", help="re-execute all guides, not only edited"
    )
    args = parser.parse_args()
    if args.action == "check":
        if args.fresh:
            parser.error("--fresh requires render")
        check(ROOT)
        return
    if args.fresh:
        shutil.rmtree(ROOT / "docs/_freeze", ignore_errors=True)
        for path in (ROOT / "docs/guides").glob("*_files"):
            shutil.rmtree(path)
    env = os.environ | {"QUARTO_PYTHON": sys.executable, "MPLBACKEND": "Agg"}
    # freeze: auto re-executes only guides whose source changed.
    run_quarto(env)
    # Match the repository's text-file hooks before committing generated JSON.
    for path in (ROOT / "docs/_freeze").rglob("*.json"):
        path.write_text(path.read_text().rstrip() + "\n")


if __name__ == "__main__":
    main()
