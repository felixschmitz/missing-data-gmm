"""Script running replication of Monte Carlo simulation."""

from typing import Annotated

import numpy as np
import pandas as pd

from missing_data_gmm.config import DATA_CATALOGS
from missing_data_gmm.monte_carlo.helper import (
    apply_method,
    generate_data,
    initialize_params,
    results_statistics,
)


def _error_handling_methods(methods: list):
    for method in methods:
        if method not in [
            "Complete case method",
            "Dummy case method",
            "Dagenais (FGLS)",
            "(Full) GMM",
            "Statsmodels GMM",
            "SciPy GMM",
        ]:
            msg = f"Unknown method: {method}"
            raise ValueError(msg)


def _error_handling_params(params: dict):
    if not all(
        isinstance(params[key], int)
        for key in [
            "n_observations",
            "k_regressors",
            "n_replications",
            "n_complete",
            "n_missing",
        ]
    ):
        msg = """Parameters n_observations, k_regressors, n_replications, n_complete,
        and n_missing must be integers."""
        raise ValueError(msg)
    if not all(
        isinstance(params[key], float) for key in ["lambda_", "sd_xi", "sd_epsilon"]
    ):
        msg = "Parameters lambda_, sd_xi, and sd_epsilon must be floats."
        raise ValueError(msg)
    if not all(
        isinstance(params[key], np.ndarray)
        for key in ["b0_coefficients", "gamma_coefficients"]
    ):
        msg = "Parameters b0_coefficients and gamma_coefficients must be numpy arrays."
        raise ValueError(msg)
    if not isinstance(params["methods"], list):
        msg = "Parameter methods must be a list."
        raise TypeError(msg)
    if not all(isinstance(key, str) for key in params["methods"]):
        msg = "All elements in methods must be strings."
        raise ValueError(msg)


def _error_handling_random_key(random_key: int):
    if not isinstance(random_key, int):
        msg = "Random key must be an integer."
        raise TypeError(msg)


def _error_handling(params: dict, random_key: int):
    _error_handling_methods(params["methods"])
    _error_handling_params(params)
    _error_handling_random_key(random_key)


RANDOM_KEY = 123456
CONSTANT_PARAMS = initialize_params()


def task_simulate(
    params: Annotated[dict, CONSTANT_PARAMS], random_key: Annotated[int, RANDOM_KEY]
) -> Annotated[pd.DataFrame, DATA_CATALOGS["simulation"]["MC_RESULTS"]]:
    """Run Monte Carlo simulation for different methods.

    Parameters:
        params (dict): Simulation parameters.
        random_key (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Formatted simulation results.
    """
    _error_handling(params, random_key)
    rng = np.random.default_rng(random_key)
    results = {method: [] for method in params["methods"]}
    for _replication in range(params["n_replications"]):
        data = generate_data(params, rng)
        for method in params["methods"]:
            results[method].append(apply_method(data, method, params))
    return results_statistics(results, params)
