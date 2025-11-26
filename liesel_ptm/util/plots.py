import jax
import liesel.goose as gs
import plotnine as p9

Array = jax.Array


def plot_loss(results: gs.OptimResult, legend: bool = True, title: str | None = None):
    history = gs.history_to_df(results.history)
    plot_data = history[["loss_validation", "loss_train", "iteration"]]
    plot_data = plot_data.melt(
        id_vars="iteration", var_name="loss_type", value_name="loss"
    )

    p = (
        p9.ggplot(plot_data)
        + p9.aes(x="iteration", y="loss", color="loss_type")
        + p9.geom_line()
    )

    if title is not None:
        p += p9.ggtitle(title)

    if not legend:
        p += p9.theme(legend_position="none")

    return p


def plot_param_history(
    results: gs.OptimResult | dict[str, Array],
    legend: bool = True,
    title: str | None = None,
):
    position = (
        results.history["position"] if isinstance(results, gs.OptimResult) else results
    )
    history = gs.history_to_df(position)

    plot_data = history.melt(id_vars="iteration")

    p = (
        p9.ggplot(plot_data)
        + p9.aes(x="iteration", y="value", color="variable", group="variable")
        + p9.geom_line()
    )

    if title is not None:
        p += p9.ggtitle(title)

    if not legend:
        p += p9.theme(legend_position="none")

    return p
