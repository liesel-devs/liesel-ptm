# Build and update the documentation

## Build the documentation

Use Python 3.13, [uv](https://docs.astral.sh/uv/),
[Quarto](https://quarto.org/docs/get-started/) 1.8.27, and Graphviz (`dot`).
CI uses the same Quarto version. On macOS, install Quarto and Graphviz with
`brew install --cask quarto` and `brew install graphviz`; the Quarto download page
also provides versioned installers. On Ubuntu, install Graphviz with
`sudo apt-get install graphviz` and use Quarto's Linux installer.

From the repository root:

```bash
uv sync --locked --group dev --group docs
uv run --locked --group docs python scripts/docs.py render
uv run --locked --group docs zensical build --clean --strict
```

The first command installs the package and the locked development/documentation
tools. The second checks output freshness and renders Quarto sources to
GitHub-flavored Markdown using stored execution results. The third builds the site
in `site/`, failing on warnings. You can also run `uv run zensical build --clean --strict`
after syncing the documentation group.

To preview the generated Markdown with live site rebuilding:

```bash
uv run --locked --group docs zensical serve
```

Stop the preview with Ctrl-C. It does not execute Quarto cells. Refresh the guides
separately when their sources change.

## Edit a page

Edit `docs/index.md` and other handwritten Markdown directly. Edit executable
guides in `docs/guides/*.qmd`, then regenerate their `.md` files and figures.
Keep imports, shared setup, and model choices visible so readers can run the
cells in a fresh session. Keep expensive examples small and use fixed seeds.

API pages in `docs/api/` use mkdocstrings directives to render the implementation's
NumPy-style docstrings. Update those docstrings for argument details and exact
rules. Use `[Name][liesel_ptm.Name]` for API cross-references in handwritten
Markdown and docstrings. In Quarto sources, use page-and-anchor links such as
`../api/models.md#liesel_ptm.PTMDist`; Pandoc does not preserve mkdocstrings labels.
Keep explicit navigation in `zensical.toml` up to date when adding pages.
Add new executable guides to the render list in `docs/_quarto.yml` as well.

## Refresh stored results

Quarto's `freeze: auto` reuses results during a project render. It does not detect
all changes to library code, included fragments, data, or dependencies. After any
such change, or after editing a guide, force fresh execution:

```bash
uv run --locked --group docs python scripts/docs.py render --fresh
uv run --locked --group docs python scripts/docs.py check
uv run --locked --group docs zensical build --clean --strict
```

The fresh render removes the guides' previous frozen results and generated asset
directories, runs all executable guides in the current environment, and records a manifest
only after the render succeeds. The renderer allows ten minutes for all guides,
stops its Quarto and kernel processes on timeout or interruption, and fails on
execution errors. Kernels shut down after rendering. CI also bounds the whole job.

Commit the `.qmd` sources together with their rendered `.md` files, figure
directories, `docs/_freeze/`, and `docs/guide-outputs.json`. Do not edit the manifest
or frozen results by hand. The check compares hashes of the library, reference
scripts, all Quarto sources (including shared fragments), bundled example data,
Quarto configuration, render script, figure filters, environment files, and
generated artifacts. It catches missing or edited outputs too. If you add an
external data file to a guide, include it in
`snapshot()` in `scripts/docs.py` so data changes also require a refresh.

For a quick iteration, `quarto render docs/guides/conditional-shape.qmd` always executes that
document, even with freezing enabled. Use the fresh-render command above before
committing so all stored results and the manifest agree. See Quarto's
[execution and freeze documentation](https://quarto.org/docs/projects/code-execution.html)
for the distinction between project and single-document renders.

## Validate changes

```bash
uv run --locked pytest tests/test_docs.py
uv run --locked pre-commit run -a
```

For changed Python cells, convert the source to a scratch notebook and check the
copy with Ruff and ty; edit the `.qmd`, not the converted notebook:

```bash
quarto convert docs/guides/conditional-shape.qmd --output /tmp/ptm-conditional-shape.ipynb
uv run --locked ruff check /tmp/ptm-conditional-shape.ipynb
uv run --locked ruff format --diff /tmp/ptm-conditional-shape.ipynb
uvx ty check /tmp/ptm-conditional-shape.ipynb
```

Inspect the affected site pages, including graphs, alt text, equations, and links.
A site build alone checks rendering; it does not prove the examples executed.

## CI and GitHub Pages

The documentation workflow validates stored outputs, renders with Quarto, and
builds Zensical in strict mode for pull requests and pushes to `main`. A manual run
also executes both guides freshly, which checks that stored outputs remain
reproducible on Linux. Only a push to `main` uploads and deploys the Pages artifact
to <https://liesel-devs.github.io/liesel-ptm/>. Repository maintainers must select
**GitHub Actions** as the Pages source in the repository settings.
