"""Helper functions for the missing data regression."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2

from missing_data_gmm.helper import calculate_moments, calculate_residuals


class GMMResultsObject:
    """Class to store the results of the Generalized Method of Moments."""

    def __init__(self, data, theta_final, weight_matrix):
        """Initialize the GMM results object.

        Args:
            data (dict): Generated data (from `generate_data`).
            theta_final (np.array): Final estimates of the model.
            weight_matrix (np.array): Weight matrix of the model.

        Returns:
            GMMResultsObject: Object containing the results of the GMM estimation.
        """
        self.data = data
        self.params = self.determine_params(theta_final)
        self.HC1_se = self.determine_standard_errors()
        self.moments = calculate_moments(
            self.data,
            self.params,
            theta_final[self.data["w_complete"].shape[1] :],
            theta_final,
        )
        self.test_statistic = self.calculate_hansen_j_statistic(weight_matrix)
        self.degrees_of_freedom = self.calculate_degrees_of_freedom(theta_final)
        self.p_value = self.calculate_p_value()
        self.nobs = self.data["n_observations"]

    def get_converted_statistics(self, index_values):
        """Convert the results to a pandas Series.

        Args:
            index_values (list): Index values for the pandas Series.

        Returns:
            GMMResultsObject: Object with the results converted to a pandas Series.
        """
        self.params = pd.Series(data=self.params, index=index_values)
        self.HC1_se = pd.Series(data=self.HC1_se, index=index_values)
        return self

    def determine_params(self, theta_final) -> np.array:
        """Determine the parameters of the model.

        Args:
            theta_final (np.array): Final estimates of the model.

        Returns:
            np.array: Parameters of the model.
        """
        return theta_final[: self.data["w_complete"].shape[1]]

    def determine_standard_errors(self) -> np.array:
        """Determine the standard errors of the model.

        Returns:
            np.array: Standard errors of the model.
        """
        residuals_complete = calculate_residuals(
            self.data["y_complete"], self.data["w_complete"], self.params
        )
        residuals_missing = calculate_residuals(
            self.data["y_missing"], self.data["z_missing"], self.params[1:]
        )
        residuals_combined = np.concatenate([residuals_complete, residuals_missing])
        sigma_squared = (residuals_combined.T @ residuals_combined) / self.data[
            "n_observations"
        ]
        return np.sqrt(
            np.diag(
                sigma_squared
                * np.linalg.inv(self.data["w_complete"].T @ self.data["w_complete"])
            )
        )

    def determine_gamma(self, theta_final) -> np.array:
        """Determine the gamma parameters of the model.

        Args:
            theta_final (np.array): Final estimates of the model.

        Returns:
            np.array: Gamma parameters of the model.
        """
        return theta_final[self.data["w_complete"].shape[1] :]

    def calculate_hansen_j_statistic(self, weight_matrix) -> float:
        """Calculate the Hansen J statistic.

        Args:
            weight_matrix (np.array): Weight matrix of the model.

        Returns:
            float: Hansen J statistic.
        """
        return self.moments.T @ weight_matrix @ self.moments

    def calculate_degrees_of_freedom(self, theta_final) -> int:
        """Calculate the degrees of freedom for the model.

        Args:
            theta_final (np.array): Final estimates of the model.

        Returns:
            int: Degrees of freedom for the model.
        """
        return len(self.moments) - len(theta_final)

    def calculate_p_value(self) -> float:
        """Calculate the p-value for the model.

        Returns:
            float: P-value for the model.
        """
        return 1 - chi2.cdf(self.test_statistic, self.degrees_of_freedom)


def initialize_gmm_params(data: pd.DataFrame) -> dict:
    """Initialize parameters for the Generalized Method of Moments.

    Args:
        data (pd.DataFrame): Data to be used in the estimation.

    Returns:
        dict: Parameters for the Generalized Method of Moments.
    """
    params = {}
    params["max_iterations"] = 100  # 16000  # number of max iterations of gmm
    params["k_regressors"] = (
        data.shape[1] - 2
    )  # number of regressors including constant
    return params


def filter_data(data: pd.DataFrame, sex_name: str) -> pd.DataFrame:
    """Filter the data based on the name of the sex.

    Args:
        data (pd.DataFrame): Data to be filtered.
        sex_name (str): Name

    Returns:
        pd.DataFrame: Filtered data.
    """
    condition = 1 if sex_name == "male" else 0
    data["constant"] = 1
    return data[data["male"] == condition].drop(columns=["male"])


def partition_data(data, independent_variables, dependent_variable) -> dict:
    """Partition data into complete and incomplete cases.

    Args:
        data (pd.DataFrame): Data to be partitioned.
        independent_variables (list): Independent variables of the model.
        dependent_variable (str): Dependent variable of the model.

    Returns:
        dict: Partitioned data.
    """
    complete = data.query("bmimissing == 0")
    missing = data.query("bmimissing == 1")
    instruments = [x for x in independent_variables if x != "bmirating"]
    return {
        "w_complete": np.column_stack(
            (np.array(complete["bmirating"]), np.array(complete[instruments]))
        ),
        "y_complete": np.array(complete[dependent_variable]),
        "x_complete": np.array(complete["bmirating"]),
        "z_complete": np.array(complete[instruments]),
        "z_missing": np.array(missing[instruments]),
        "y_missing": np.array(missing[dependent_variable]),
        "n_complete": complete.shape[0],
        "n_missing": missing.shape[0],
        "n_observations": data.shape[0],
    }


def regr_robust_summary(
    data: pd.DataFrame,
    x_columns: list[str],
    y_column: list[str],
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Estimate the regression model with robust standard errors (HC1).

    Args:
        data (pd.DataFrame): Data to be used in the estimation.
        x_columns (list[str]): Independent variables of the model.
        y_column (list[str]): Dependent variable of the model.

    Returns:
        sm.regression.linear_model.RegressionResultsWrapper: Results of the regression.
    """
    x = data[x_columns]
    y = data[y_column]
    return sm.OLS(y, x, hasconst=True).fit(cov_type="HC1")


def gmm_descriptive_statistics(
    data: dict,
    theta_final: np.array,
    weight_matrix: np.array,
    independent_variables: list,
) -> GMMResultsObject:
    """Descriptive statistics for GMM.

    Args:
        data (dict): Generated data (from `generate_data`).
        theta_final (np.array): Final estimates of the model.
        weight_matrix (np.array): Weight matrix of the model.
        independent_variables (list): Independent variables of the model.

    Returns:
        GMMResultsObject: Results containing coefficients and standard errors.
    """
    results = GMMResultsObject(data, theta_final, weight_matrix)
    return results.get_converted_statistics(independent_variables)
