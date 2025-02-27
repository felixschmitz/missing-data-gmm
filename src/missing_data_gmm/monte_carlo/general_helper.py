"""General helper functions for the Monte Carlo simulation."""

import numpy as np

from missing_data_gmm.monte_carlo.complete import complete_case_method
from missing_data_gmm.monte_carlo.dagenais import dagenais_weighted_method
from missing_data_gmm.monte_carlo.dummy import dummy_variable_method
from missing_data_gmm.monte_carlo.gmm import gmm_method


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
    mean_y = np.column_stack((x, z)) @ (
        np.concatenate([params["alpha_coefficients"], params["beta_coefficients"]])
    )  # my
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
        "x_missing": x[n_complete:],
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
