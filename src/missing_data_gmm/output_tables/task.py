"""Create a LaTeX table from simulation results."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import task

from missing_data_gmm.config import BLD, DATA_CATALOGS, EXPERIMENTS, MC_DESIGNS
from missing_data_gmm.missreg.task import (
    task_complete_data_regression,
    task_dummy_variable_method,
    task_gmm,
)


def _format_simulation_output_table(data: pd.DataFrame, design: int) -> str:
    data["Method"] = data["Method"].mask(data["Method"].duplicated(), "")
    data["Parameter"] = data["Parameter"].apply(lambda x: rf"$\{x}$")
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
    def task_simulation_output_table(
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
        return _format_simulation_output_table(raw_statistics, design)


def _format_panel(content: dict) -> str:
    """Format subpanels into strcutured LaTeX table."""
    columns = list(next(iter(content.values())).keys())
    rows = list(next(iter(next(iter(content.values())).values())).keys())

    out = ""
    for row_name in rows:
        if isinstance(
            next(iter(next(iter(content.values())).values()))[row_name], dict
        ):
            for subpanel in content:
                if "not" in subpanel.sex.name:
                    female_params = [
                        content[subpanel][col_name][row_name]["param"]
                        for col_name in columns
                    ]
                    female_ses = [
                        content[subpanel][col_name][row_name]["HC1_se"]
                        for col_name in columns
                    ]
                else:
                    male_params = [
                        content[subpanel][col_name][row_name]["param"]
                        for col_name in columns
                    ]
                    male_ses = [
                        content[subpanel][col_name][row_name]["HC1_se"]
                        for col_name in columns
                    ]
            out += (
                f"{row_name} & "
                + " & ".join(male_params)
                + " &  & "
                + " & ".join(female_params)
                + "\\\\"
                + "\n"
            )
            out += (
                " & "
                + " & ".join(male_ses)
                + " &  & "
                + " & ".join(female_ses)
                + "\\\\"
                + "\n"
            )
        else:
            for subpanel in content:
                if "not" in subpanel.sex.name:
                    female_values = [
                        content[subpanel][col_name][row_name] for col_name in columns
                    ]
                else:
                    male_values = [
                        content[subpanel][col_name][row_name] for col_name in columns
                    ]
            if row_name in ["test statistic", "observations"]:
                out += "\\\\" + "\n"
            out += (
                f"{row_name} & "
                + " & ".join(male_values)
                + " &  & "
                + " & ".join(female_values)
                + "\\\\"
                + "\n"
            )
    return out


def _initialize_table() -> str:
    out = "\\begin{table}[ht]\n"
    out += "\\centering\n"
    out += "\\caption{Regression examples, Wisconsin Longitudinal Study data}\n"
    out += "\\label{tab:regression_examples}\n"
    out += "\\begin{tabular}{lccccccc}\n"
    return out


def _create_panels_container(experiments: list) -> dict:
    panels = {"A": {}, "B": {}}
    for experiment in experiments:
        if "adultbmi" in experiment.dataset.name:
            panels["B"][experiment] = {method: {} for method in experiment.methods}
        else:
            panels["A"][experiment] = {method: {} for method in experiment.methods}
    return panels


def _add_panel_header(panel_name: str, dependent_variable: str) -> str:
    out = "\\hline\\hline\n"
    out += (
        f"Panel {panel_name} & \\multicolumn{{7}}{{c}}{{Dependent variable = "
        f"{dependent_variable}}} \\\\\n"
    )
    out += "\\cline{2-8}\n"

    out += " & \\multicolumn{3}{c}{Men} & & \\multicolumn{3}{c}{Women} \\\\\n"
    out += "\\cline{2-4}\\cline{6-8}\n"
    out += " & Complete & Dummy & GMM & & Complete & Dummy & GMM \\\\\n"
    out += "\\hline\n"
    return out


def _add_panel_content(
    panels: dict,
    panel_name: str,
    raw_statistics: pytask.DataCatalog,
    independent_variables: list,
) -> str:
    for subpanel in panels[panel_name]:
        for method in subpanel.methods:
            data = raw_statistics[f"{method}-{subpanel.name}"].load()
            for var in independent_variables:
                panels[panel_name][subpanel][method][var] = {}
                if var in data.params.index:
                    panels[panel_name][subpanel][method][var]["param"] = (
                        f"{data.params[var]:.4f}"
                    )
                    panels[panel_name][subpanel][method][var]["HC1_se"] = (
                        f"({data.HC1_se[var]:.4f})"
                    )
                else:
                    panels[panel_name][subpanel][method][var]["param"] = ""
                    panels[panel_name][subpanel][method][var]["HC1_se"] = ""
            if method == "gmm":
                panels[panel_name][subpanel][method]["test statistic"] = (
                    f"{data.test_statistic:.4f}"
                )
                panels[panel_name][subpanel][method]["degrees of freedom"] = (
                    f"{data.degrees_of_freedom:.0f}"
                )
                panels[panel_name][subpanel][method]["p-value"] = f"{data.p_value:.4f}"
            else:
                panels[panel_name][subpanel][method]["test statistic"] = ""
                panels[panel_name][subpanel][method]["degrees of freedom"] = ""
                panels[panel_name][subpanel][method]["p-value"] = ""
            panels[panel_name][subpanel][method]["observations"] = f"{data.nobs:.0f}"
    return _format_panel(panels[panel_name])


def _close_table() -> str:
    out = "\\hline\\hline\n"
    out += "\\end{tabular}\n"
    out += "\\end{table}\n"
    return out


@task(after=[task_complete_data_regression, task_dummy_variable_method, task_gmm])
def task_missreg_output_table(
    experiments: list = EXPERIMENTS,
    raw_statistics: dict = DATA_CATALOGS["regression"]["missreg"],
) -> Annotated[Path, BLD / "tables" / "missreg_results.tex"]:
    """Create a LaTeX table from regression results.

    Args:
        experiments (list): List of experiments.
        raw_statistics (dict): Raw regression results.

    Returns:
        table (str): LaTeX table with regression results.
    """
    table = _initialize_table()
    panels = _create_panels_container(experiments)

    for panel_name, panel_content in panels.items():
        independent_variables = next(
            iter(panel_content.keys())
        ).dataset.independent_variables
        dependent_variable = next(iter(panel_content.keys())).dataset.dependent_variable
        table += _add_panel_header(panel_name, dependent_variable)
        table += _add_panel_content(
            panels, panel_name, raw_statistics, independent_variables
        )

    table += _close_table()
    return table
