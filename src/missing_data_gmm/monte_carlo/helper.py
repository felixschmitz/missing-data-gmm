"""Helper functions for the Monte Carlo simulation."""

import numpy as np
import pandas as pd

from missing_data_gmm.config import METHODS
from missing_data_gmm.monte_carlo.complete import complete_case_method
from missing_data_gmm.monte_carlo.dagenais import dagenais_weighted_method
from missing_data_gmm.monte_carlo.dummy import dummy_variable_method
from missing_data_gmm.monte_carlo.gmm import gmm_method


def _get_design_parameters(design: int, k_regressors: int) -> list:
    match design:
        case 1:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([10, 0]),
                np.array([1] * (k_regressors - 1)),
                np.array([10, 0, 0]),
                False,
            ]
        case 2:
            return [
                np.array([0.1]),
                np.array([1] * (k_regressors - 1)),
                np.array([10, 0]),
                np.array([1] * (k_regressors - 1)),
                np.array([10, 0, 0]),
                False,
            ]
        case 3:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([10, 0]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 0, 0]),
                False,
            ]
        case 4:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 0, 1]),
                False,
            ]
        case 5:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 1, 1]),
                False,
            ]
        case 6:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 0]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 0, 0]),
                False,
            ]
        case 7:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 0, 1]),
                False,
            ]
        case 8:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([0.1, 0.2]),
                np.array([1] * (k_regressors - 1)),
                np.array([0.1, 0.2, 0.1]),
                True,
            ]
        case 9:
            return [
                np.array([0.1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 1, 1]),
                False,
            ]
        case 10:
            return [
                np.array([1]),
                np.array([1] * (k_regressors - 1)),
                np.array([1, 1]),
                np.array([1, 0.1]),
                np.array([1, 1, 1]),
                False,
            ]


def initialize_replication_params(design: int = 0) -> dict:
    """Initialize parameters for the Monte Carlo simulation.

    Returns:
        dict: Parameters for the Monte Carlo simulation.
    """
    params = {}
    params["design"] = design
    params["methods"] = METHODS
    params["n_observations"] = 400  # Number of observations
    params["k_regressors"] = 3  # Number of regressors (including intercept)
    params["lambda_"] = 0.5  # Proportion of observations with missing data
    params["n_replications"] = 5000  # Number of Monte Carlo replications
    params["n_complete"] = int(
        params["n_observations"] * params["lambda_"]
    )  # Number of complete cases
    params["n_missing"] = (
        params["n_observations"] - params["n_complete"]
    )  # Number of missing cases

    keys = [
        "alpha_coefficients",
        "beta_coefficients",
        "delta_coefficients",
        "gamma_coefficients",  # Imputation coefficients
        "theta_coefficients",
        "exponential",
    ]
    values = _get_design_parameters(design, params["k_regressors"])
    params.update(dict(zip(keys, values, strict=False)))
    params["b0_coefficients"] = np.concatenate(
        [params["alpha_coefficients"], params["beta_coefficients"]]
    )  # True coefficients

    params["max_iterations"] = 200  # number of max iterations of gmm
    params["random_key"] = 123456
    return params


def _generate_instruments(n: int, rng: np.random.Generator) -> np.ndarray:
    """Generate instrument variables z including intercept."""
    # binary_instrument = rng.standard_normal(n) > 0.5  # z1
    continuous_instrument = rng.standard_normal(n)  # z2
    return np.column_stack((np.ones(n), continuous_instrument))  # z with intercept


def _generate_x(
    z: np.ndarray,
    v: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Generate independent variable x with heteroskedasticity."""
    mean_x = z @ params["gamma_coefficients"]  # mx
    regressors = np.column_stack([np.ones_like(z[:, 1]), np.square(z[:, 1])])
    sigma_xi = np.sqrt(np.dot(regressors, params["delta_coefficients"]))

    if params["exponential"]:
        return mean_x + v * np.exp(sigma_xi)
    return mean_x + v * sigma_xi


def _generate_y(
    x: np.ndarray,
    z: np.ndarray,
    u: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Generate dependent variable y with heteroskedasticity."""
    mean_y = np.column_stack((x, z)) @ params["b0_coefficients"]  # my
    regressors = np.column_stack([np.ones_like(x), np.square(x), np.square(z[:, 1])])
    sigma_epsilon = np.sqrt(np.dot(regressors, params["theta_coefficients"]))

    if params["exponential"]:
        return mean_y + u * np.exp(sigma_epsilon)
    return mean_y + u * sigma_epsilon


def _partition_data(x, z, y, n_complete):
    """Partition data into complete and incomplete cases."""
    return {
        "w_complete": np.column_stack((x[:n_complete], z[:n_complete])),
        "y_complete": y[:n_complete],
        "x_complete": x[:n_complete],
        "z_complete": z[:n_complete, :],
        "z_missing": z[n_complete:, :],
        "y_missing": y[n_complete:],
        "n_complete": n_complete,
    }


def generate_data(params: dict, rng: np.random.Generator) -> dict:
    """Data generating process.

    Args:
        params (dict): Parameters of the Monte Carlo simulation.
        rng (np.random.Generator): Random number generator.

    Returns:
        dict: Generated data.
    """
    z = np.column_stack(
        (
            np.ones(params["n_observations"]),
            rng.standard_normal(params["n_observations"]),
        )
    )  # continuous instrument z with intercept

    u = rng.standard_normal(params["n_observations"])
    if params["design"] in [6, 7]:
        v = np.square(u) - 1
    else:
        v = rng.standard_normal(params["n_observations"])
    x = _generate_x(z, v, params)
    y = _generate_y(x, z, u, params)

    partitions = _partition_data(x, z, y, params["n_complete"])
    return {"x": x, "y": y, "z": z, "n_missing": params["n_missing"], **partitions}


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
    if method == "FGLS (Dagenais)":
        return dagenais_weighted_method(data, params)
    if method == "GMM":
        return gmm_method(data, params)
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
