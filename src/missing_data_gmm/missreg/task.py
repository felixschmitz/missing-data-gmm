"""Task related to missing data regression."""

from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import task

from missing_data_gmm.config import DATA_CATALOGS, EXPERIMENTS
from missing_data_gmm.missreg.helper import (
    filter_data,
    gmm_descriptive_statistics,
    initialize_gmm_params,
    partition_data,
    regr_robust_summary,
)
from missing_data_gmm.monte_carlo.gmm import gmm_method

for experiment in EXPERIMENTS:

    @task(id=experiment.name)
    def task_store_filtered_data(
        path_to_data: Path = experiment.dataset.path,
        sex_name: str = experiment.sex.name,
    ) -> Annotated[pd.DataFrame, DATA_CATALOGS["regression"]["raw"][experiment.name]]:
        """Store the raw data in the data catalog.

        Parameters:
            path_to_data (Path): Path to the raw data.

        Returns:
            pd.DataFrame: The raw filtered data.
        """
        raw_data = pd.read_stata(path_to_data)
        data = raw_data.copy()
        data["bmimissing"] = pd.to_numeric(data["bmimissing"], downcast="integer")
        return filter_data(data, sex_name)

    @task(id=experiment.name)
    def task_complete_data_regression(
        data: pd.DataFrame = DATA_CATALOGS["regression"]["raw"][f"{experiment.name}"],
        independent_variables: str = experiment.dataset.independent_variables,
        dependent_variable: str = experiment.dataset.dependent_variable,
    ) -> Annotated[
        pytask.PickleNode,
        DATA_CATALOGS["regression"]["missreg"][f"{experiment.complete_model_name}"],
    ]:
        """Estimate the regression model with complete data."""
        complete_data = data[data["bmimissing"] != 1]
        independent_variables_complete = [
            x for x in independent_variables if x not in ["bmimissing"]
        ]
        return regr_robust_summary(
            complete_data,
            independent_variables_complete,
            dependent_variable,
            title="OLS Complete Data Method",
        )

    @task(id=experiment.name)
    def task_dummy_variable_method(
        data: pd.DataFrame = DATA_CATALOGS["regression"]["raw"][f"{experiment.name}"],
        independent_variables: str = experiment.dataset.independent_variables,
        dependent_variable: str = experiment.dataset.dependent_variable,
    ) -> Annotated[
        pytask.PickleNode,
        DATA_CATALOGS["regression"]["missreg"][f"{experiment.dummy_model_name}"],
    ]:
        """Estimate the regression model with the dummy variable method."""
        return regr_robust_summary(
            data,
            independent_variables,
            dependent_variable,
            title="OLS Dummy Variable Method",
        )

    @task(id=experiment.name)
    def task_gmm(
        data: pd.DataFrame = DATA_CATALOGS["regression"]["raw"][f"{experiment.name}"],
        independent_variables: str = experiment.dataset.independent_variables,
        dependent_variable: str = experiment.dataset.dependent_variable,
    ) -> Annotated[
        pytask.PickleNode,
        DATA_CATALOGS["regression"]["missreg"][f"{experiment.gmm_model_name}"],
    ]:
        """Estimate the model with the gmm."""
        independent_variables_gmm = [
            x for x in independent_variables if x not in ["bmimissing"]
        ]
        partitioned_data = partition_data(
            data, independent_variables_gmm, dependent_variable
        )
        params = initialize_gmm_params(data)
        data, theta_final, weight_matrix = gmm_method(
            partitioned_data, params, descriptive_statistics=False
        )
        return gmm_descriptive_statistics(
            data, theta_final, weight_matrix, independent_variables_gmm
        )
