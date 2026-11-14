# Build and update the documentation

## Build the documentation

Use Python 3.13, [uv](https://docs.astral.sh/uv/),
[Quarto](https://quarto.org/docs/get-started/) 1.11.9, and Graphviz (`dot`).
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
tools. The second renders Quarto sources to GitHub-flavored Markdown, reusing
stored execution results for guides you have not edited. The third builds the site
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
Initialize term builders in separate assignments before adding terms to predictors.
Name them with a `tb_` prefix, such as `tb_loc`, `tb_scale`, or `tb_shape`.
Use `.lin()` without an explicit prior in guides.
Give longer prior specifications their own names so term additions stay on one line
when practical.

API pages in `docs/api/` use mkdocstrings directives to render the implementation's
NumPy-style docstrings. Update those docstrings for argument details and exact
rules. Use `[Name][liesel_ptm.Name]` for API cross-references in handwritten
Markdown and docstrings. In Quarto sources, use page-and-anchor links such as
`../api/models.md#liesel_ptm.PTMDist`; Pandoc does not preserve mkdocstrings labels.
Keep explicit navigation in `zensical.toml` up to date when adding pages.
Add new executable guides to the render list in `docs/_quarto.yml` as well.

## Refresh stored results

Quarto's `freeze: auto` stores each guide's execution results with a hash of its
source and re-executes only guides whose source changed. After editing a guide,
re-render and check:

```bash
uv run --locked --group docs python scripts/docs.py render
uv run --locked --group docs python scripts/docs.py check
uv run --locked --group docs zensical build --clean --strict
```

Commit the `.qmd` sources together with their rendered `.md` files, figure
directories and `docs/_freeze/`. Do not edit frozen results by hand.

The check fails only when a guide's source no longer matches its stored results.
Changes to library code, dependencies, data or tooling don't fail it:
they rarely change guide results, and every refresh rewrites the figures, which
grows the repository. When such a change does affect a guide, refresh all guides
with `render --fresh`. It removes the previous frozen results and figure
directories and re-executes every guide. The renderer allows ten minutes for all
guides, stops its Quarto and kernel processes on timeout or interruption, and
fails on execution errors. Kernels shut down after rendering. CI also bounds the
whole job, and a manual run of the docs workflow executes all guides fresh.

For a quick iteration, `quarto render docs/guides/conditional-shape.qmd` always executes that
document, even with freezing enabled. Run the project render above before
committing so the stored results match the source. See Quarto's
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
also executes all guides freshly, which checks that stored outputs remain
reproducible on Linux. Only a push to `main` uploads and deploys the Pages artifact
to <https://liesel-devs.github.io/liesel-ptm/>. Repository maintainers must select
**GitHub Actions** as the Pages source in the repository settings.
