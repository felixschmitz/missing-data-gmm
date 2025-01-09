"""Create a LaTeX table from simulation results."""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import task

from missing_data_gmm.config import BLD, DATA_CATALOGS, MC_DESIGNS


def _format_output_table(data, design):
    data["Method"] = data["Method"].mask(data["Method"].duplicated(), "")
    data["Parameter"] = data["Parameter"].apply(lambda x: f"$\\{x}$")
    return (
        data.style.hide(axis=0)
        .relabel_index(
            ["Estimation Method", "Parameter", "Bias", r"n$\times$Var", "MSE"], axis=1
        )
        .format({col: "{:.3f}" for col in data.select_dtypes(include="number").columns})
        .set_caption(f"Monte Carlo Replication Results, Design {design}")
        .to_latex(
            column_format="lcccc",
            label=f"table:MCReplicationResultsDesign{design}",
            position_float="centering",
            hrules=True,
        )
    )


for design in MC_DESIGNS:

    @task(id=str(design))
    def task_output_table(
        raw_statistics: Annotated[
            pd.DataFrame, DATA_CATALOGS["simulation"][f"MC_RESULTS_{design}"]
        ],
        design: Annotated[int, design],
    ) -> Annotated[Path, BLD / "tables" / f"simulation_results_design{design}.tex"]:
        """Create a LaTeX table from simulation results.

        Parameters:
            raw_statistics (DataFrame): Simulation results for methods and parameters.
            design (int): Identifier of Monte Carlo design.
        """
        return _format_output_table(raw_statistics, design)
