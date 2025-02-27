"""Create a LaTeX table from simulation results."""

from typing import Annotated

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pytask
import regex as re
from plotly.subplots import make_subplots
from pytask import task

from missing_data_gmm.config import BLD, DATA_CATALOGS

GRID_SUBNAMES = [
    "gamma20_homoskedastic",
    "gamma20_heteroskedastic_imputation",
    "gamma20_heteroskedastic_regression",
]

for GRID_SUBNAME in GRID_SUBNAMES:

    @task(id=GRID_SUBNAME)
    @pytask.mark.after("task_simulate_grid")
    def task_plot(
        raw_statistics: Annotated[pytask.DataCatalog, DATA_CATALOGS["simulation"]],
        grid_subname: str = GRID_SUBNAME,
    ):
        """Create a figure from simulation results.

        Parameters:
            raw_statistics (pytask.DataCatalog): DataCatalog with simulation results.
            grid_subname (str): Subname of the grid.
        """
        return pio.write_image(
            _create_figure(raw_statistics, grid_subname),
            BLD / "figures" / f"simulation_results_{grid_subname}.png",
            width=3 * 300,
            height=1.25 * 300,
        )


def _create_figure(data_catalog: pytask.DataCatalog, s: str) -> go.Figure:
    sorted_keys = _get_sorted_grid_keys(s + "_GRID_")
    data = _merge_results(data_catalog, sorted_keys)
    return _format_figure(data)


def _get_sorted_grid_keys(grid_name: str) -> list:
    keys = [key for key in DATA_CATALOGS["simulation"]._entries if grid_name in key]  # noqa: SLF001
    return sorted(keys, key=lambda s: int(re.search(r"\d+$", s).group()))


def _merge_results(data_catalog: pytask.DataCatalog, sorted_keys: list) -> pd.DataFrame:
    out = pd.DataFrame()
    for key in sorted_keys:
        data = data_catalog[key].load()
        out = pd.concat([out, data[data.Parameter.str.contains("beta_")]]).reset_index(
            drop=True
        )
    return out


def _format_figure(data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=(r"$\beta_1$", r"$\beta_2$"))

    fig_beta1 = px.line(
        data[data["Parameter"] == "beta_1"],
        x="gamma_20",
        y="MSE",
        color="Method",
        markers=True,
        labels={"gamma_20": r"$\gamma_{20}$", "MSE": r"$MSE(\beta_1)$"},
    )

    fig_beta2 = px.line(
        data[data["Parameter"] == "beta_2"],
        x="gamma_20",
        y="MSE",
        color="Method",
        markers=True,
        labels={"gamma_20": r"$\gamma_{20}$", "MSE": r"$MSE(\beta_2)$"},
    )

    for trace in fig_beta1.data:
        trace.showlegend = True
        fig.add_trace(trace, row=1, col=1)

    for trace in fig_beta2.data:
        trace.showlegend = False
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(
        legend_title_text=r"$\text{Method}$",
        showlegend=True,
        legend={
            "orientation": "h",
            "yanchor": "top",
            "xanchor": "center",
            "x": 0.5,
            "traceorder": "normal",
            "tracegroupgap": 0,
        },
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"l": 0, "r": 0, "t": 50, "b": 50},
    )

    fig.update_xaxes(
        showline=True,
        linecolor="black",
        tickcolor="black",
        mirror=True,
        tickmode="auto",
        title_text=r"$\gamma_{20}$",
        row=1,
        col=1,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        tickcolor="black",
        mirror=True,
        tickmode="auto",
        title_text=r"$MSE$",
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showline=True,
        linecolor="black",
        tickcolor="black",
        mirror=True,
        tickmode="auto",
        title_text=r"$\gamma_{20}$",
        row=1,
        col=2,
    )
    fig.update_yaxes(
        showline=True,
        linecolor="black",
        tickcolor="black",
        mirror=True,
        tickmode="auto",
        row=1,
        col=2,
        matches="y",
    )
    return fig
