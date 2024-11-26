"""Helper functions for the Monte Carlo simulation."""

import numpy as np
import pandas as pd

from missing_data_gmm.monte_carlo.complete import complete_case_method
from missing_data_gmm.monte_carlo.dagenais import dagenais_weighted_method
from missing_data_gmm.monte_carlo.dummy import dummy_variable_method
from missing_data_gmm.monte_carlo.gmm import (
    gmm_method,
    scipy_gmm_method,
    statsmodels_gmm_method,
)


def initialize_params() -> dict:
    """Initialize parameters for the Monte Carlo simulation.

    Returns:
        dict: Parameters for the Monte Carlo simulation.
    """
    params = {}
    params["methods"] = [
        "Complete case method",
        "Dummy case method",
        "Dagenais (FGLS)",
        # "(Full) GMM",
        "Statsmodels GMM",
        # "SciPy GMM"
    ]
    params["n_observations"] = 400  # Number of observations
    params["k_regressors"] = 3  # Number of regressors (including intercept)
    params["lambda_"] = 0.5  # Proportion of observations with missing data
    params["sd_xi"] = 1.0  # Standard deviation for xi
    params["sd_epsilon"] = 1.0  # Standard deviation for epsilon
    params["n_replications"] = 1000  # Number of Monte Carlo replications

    params["n_complete"] = int(
        params["n_observations"] * params["lambda_"]
    )  # Complete cases
    params["n_missing"] = (
        params["n_observations"] - params["n_complete"]
    )  # Missing cases
    params["b0_coefficients"] = np.array([1, 1, 1])  # True coefficients
    params["gamma_coefficients"] = np.array(
        [1] + [1] * (params["k_regressors"] - 2)
    )  # Imputation coefficients
    return params


def _generate_instruments(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate instrument variables z including intercept."""
    # binary_instrument = rng.standard_normal(n) > 0.5  # z1
    continuous_instrument = rng.standard_normal(n)  # z2
    return np.column_stack((np.ones(n), continuous_instrument))  # z


def _generate_x(
    z: np.ndarray, gamma_coefficients: np.ndarray, sd_xi: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate independent variable x with heteroskedasticity."""
    mean_x = z @ gamma_coefficients  # mx
    heteroskedasticity_factor = np.sqrt(z[:, 0] + z[:, 1] ** 2)  # hetx
    heteroskedastic_xi = (
        sd_xi * rng.standard_normal(len(z)) * heteroskedasticity_factor
    )  # xi
    return mean_x + heteroskedastic_xi  # mx + xi


def _generate_y(x, z, b0, se, rng):
    """Generate dependent variable y with heteroskedasticity."""
    mean_y = np.column_stack((x, z)) @ b0  # my
    heteroskedasticity_factor = np.sqrt(z[:, 0] + z[:, 1] ** 2 + x**2)  # hety
    heteroskedastic_e = (
        se * rng.standard_normal(len(z)) * heteroskedasticity_factor
    )  # e
    return mean_y + heteroskedastic_e  # my + e


def _partitiona_data(x, z, y, n_observations, lambda_):
    """Partition data into complete and incomplete cases."""
    n_complete = int(n_observations * lambda_)  # Number of complete cases
    return {
        "w_complete": np.column_stack((x[:n_complete], z[:n_complete])),
        "y_complete": y[:n_complete],
        "x_complete": x[:n_complete],
        "z_complete": z[:n_complete, 1:],
        "z_missing": z[n_complete:, 1:],
        "y_missing": y[n_complete:],
        "n_complete": n_complete,
        "n_missing": n_observations - n_complete,
    }


def generate_data(params: dict, rng: np.random.Generator) -> dict:
    """Data generating process.

    Args:
        params (dict): Parameters of the Monte Carlo simulation.
        rng (np.random.Generator): Random number generator.

    Returns:
        dict: Generated data.
    """
    n_observations, lambda_, b0_coefficients, gamma_coefficients, sd_xi, sd_epsilon = (
        params["n_observations"],
        params["lambda_"],
        params["b0_coefficients"],
        params["gamma_coefficients"],
        params["sd_xi"],
        params["sd_epsilon"],
    )
    z = _generate_instruments(n_observations, rng)
    x = _generate_x(z, gamma_coefficients, sd_xi, rng)
    y = _generate_y(x, z, b0_coefficients, sd_epsilon, rng)
    partitions = _partitiona_data(x, z, y, n_observations, lambda_)
    return {"x": x, "y": y, "z": z, **partitions}


def apply_method(data, method, params):
    """Apply the specified estimation method to the generated data.

    Parameters:
        data (dict): Generated data (from `generate_data`).
        method (str): The name of the estimation method to apply.
        params (dict): Simulation parameters.

    Returns:
        dict: Results containing estimates and standard errors.
    """
    if method == "Complete case method":
        return complete_case_method(data, params)
    if method == "Dummy case method":
        return dummy_variable_method(data, params)
    if method == "Dagenais (FGLS)":
        return dagenais_weighted_method(data, params)
    if method == "(Full) GMM":
        return gmm_method(data, params)
    if method == "Statsmodels GMM":
        return statsmodels_gmm_method(data, params)
    if method == "SciPy GMM":
        return scipy_gmm_method(data, params)
    msg = f"Unknown method: {method}"
    raise ValueError(msg)


def results_statistics(results: dict, params: dict) -> pd.DataFrame:
    """Calculate statistics of the results from the Monte Carlo simulation.

    Args:
        results (dict): Results from the Monte Carlo simulation.
        params (dict): Parameters of the Monte Carlo simulation.

    Returns:
        pd.DataFrame: DataFrame with statistics of the results.
    """
    parameters = [
        f"beta_{i}" if i > 0 else "alpha_0" for i in range(params["k_regressors"])
    ]
    results_df = []
    for method, method_results in results.items():
        coefficients = np.array([entry["coefficients"] for entry in method_results])

        mean_estimates = np.mean(coefficients, axis=0)
        mean_biases = mean_estimates - params["b0_coefficients"]
        n_vars = params["n_observations"] * np.var(coefficients, axis=0, ddof=1)
        mses = mean_biases**2 + (n_vars / params["n_observations"])

        # Create rows for the DataFrame
        for i, parameter in enumerate(parameters):
            results_df.append(
                {
                    "Method": method,
                    "Parameter": parameter,
                    "Bias": mean_biases[i],
                    "n*Var": n_vars[i],
                    "MSE": mses[i],
                }
            )

    return pd.DataFrame(results_df)
