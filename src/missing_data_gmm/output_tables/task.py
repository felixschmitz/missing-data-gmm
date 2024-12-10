"""Create a LaTeX table from simulation results."""

from pathlib import Path
from typing import Annotated

import pandas as pd

from missing_data_gmm.config import BLD, DATA_CATALOGS


def task_output_table(
    data: Annotated[pd.DataFrame, DATA_CATALOGS["simulation"]["MC_RESULTS"]],
) -> Annotated[Path, BLD / "tables" / "simulation_results.tex"]:
    """Create a LaTeX table from simulation results.

    Parameters:
        simulation_data (list of dict): Simulation results by method and parameter.
        file_name (str): File name for the output LaTeX table.
    """
    descriptive_statistics = pd.DataFrame(data)
    descriptive_statistics["Method"] = descriptive_statistics["Method"].mask(
        descriptive_statistics["Method"].duplicated(), ""
    )
    descriptive_statistics["Parameter"] = descriptive_statistics["Parameter"].apply(
        lambda x: f"$\\{x}$"
    )

    return descriptive_statistics.to_latex(
        index=False,
        float_format="%.3f",
        column_format="lcccc",
        caption="Monte Carlo Replication Results",
        label="tab:my_simulation_results",
        header=["Estimation Method", "Parameter", "Bias", "n*Var", "MSE"],
        escape=False,
    )
