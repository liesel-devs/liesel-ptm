from typing import Any

import jax
import jax.numpy as jnp
import liesel.goose as gs
import plotnine as p9
from jax import Array
from jax.typing import ArrayLike

from ..util.summary import summarise_by_samples
from .var import Term

KeyArray = Any


def plot_term(
    term: Term,
    samples: dict[str, Array],
    grid: ArrayLike | None = None,
    ci_quantiles: tuple[float, float] | None = (0.05, 0.95),
    hdi_prob: float | None = None,
    show_n_samples: int | None = 50,
    seed: int | KeyArray = 1,
):
    assert isinstance(term, Term)

    if grid is None:
        xgrid = jnp.linspace(term.basis.x.value.min(), term.basis.x.value.max(), 150)
    else:
        xgrid = jnp.asarray(grid)

    term_samples = term.predict(samples, newdata={term.basis.x.name: xgrid})
    ci_quantiles_ = (0.05, 0.95) if ci_quantiles is None else ci_quantiles
    hdi_prob_ = 0.9 if hdi_prob is None else hdi_prob
    term_summary = (
        gs.SamplesSummary.from_array(
            term_samples, name=term.name, quantiles=ci_quantiles_, hdi_prob=hdi_prob_
        )
        .to_dataframe()
        .reset_index()
    )

    term_summary[term.basis.x.name] = xgrid

    p = p9.ggplot() + p9.labs(
        title=f"Posterior summary of {term.name}",
        x=term.basis.x.name,
        y=term.name,
    )

    if ci_quantiles is not None:
        p = p + p9.geom_ribbon(
            p9.aes(
                term.basis.x.name,
                ymin=f"q_{str(ci_quantiles[0])}",
                ymax=f"q_{str(ci_quantiles[1])}",
            ),
            fill="#56B4E9",
            alpha=0.5,
            data=term_summary,
        )

    if hdi_prob is not None:
        p = p + p9.geom_line(
            p9.aes(term.basis.x.name, "hdi_low"),
            linetype="dashed",
            data=term_summary,
        )

        p = p + p9.geom_line(
            p9.aes(term.basis.x.name, "hdi_high"),
            linetype="dashed",
            data=term_summary,
        )

    if show_n_samples is not None and show_n_samples > 0:
        key = jax.random.key(seed) if isinstance(seed, int) else seed

        summary_samples_df = summarise_by_samples(
            key=key, a=term_samples, name=term.name, n=show_n_samples
        )

        summary_samples_df[term.basis.x.name] = jnp.tile(
            jnp.squeeze(xgrid), show_n_samples
        )

        p = p + p9.geom_line(
            p9.aes(term.basis.x.name, term.name, group="sample"),
            color="grey",
            data=summary_samples_df,
            alpha=0.3,
        )

    p = p + p9.geom_line(
        p9.aes(term.basis.x.name, "mean"), data=term_summary, size=1.3, color="blue"
    )

    return p
