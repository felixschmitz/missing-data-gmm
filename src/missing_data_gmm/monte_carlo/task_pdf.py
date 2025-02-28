"""Script running replication of Monte Carlo simulation."""

from typing import Annotated

import numpy as np
import pandas as pd
from pytask import task

from missing_data_gmm.config import DATA_CATALOGS, METHODS
from missing_data_gmm.monte_carlo.general_helper import (
    apply_method,
    generate_data,
)
from missing_data_gmm.monte_carlo.pdf_helper import (
    initialize_params,
    results_statistics,
)


def _error_handling_methods(methods: list):
    for method in methods:
        if method not in METHODS:
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
            "random_key",
        ]
    ):
        msg = """Parameters n_observations, k_regressors, n_replications, n_complete,
        n_missing, and random_key must be integers."""
        raise ValueError(msg)
    if not all(isinstance(params[key], (float)) for key in ["lambda_"]):
        msg = "Parameter lambda_ must be a float."
        raise ValueError(msg)
    if not all(
        isinstance(params[key], np.ndarray)
        for key in [
            "alpha_coefficients",
            "beta_coefficients",
            "delta_coefficients",
            "gamma_coefficients",
            "theta_coefficients",
        ]
    ):
        msg = """Parameters alpha_coefficients, beta_coefficients, delta_coefficients,
        gamma_coefficients, and theta_coefficients must be numpy arrays."""
        raise ValueError(msg)
    if not isinstance(params["methods"], list):
        msg = "Parameter methods must be a list."
        raise TypeError(msg)
    if not all(isinstance(key, str) for key in params["methods"]):
        msg = "All elements in methods must be strings."
        raise ValueError(msg)


def _error_handling(params: dict):
    _error_handling_methods(params["methods"])
    _error_handling_params(params)


def _generate_grid_params(deltas: np.ndarray, thetas: np.ndarray) -> dict:
    grid_params = {}

    alpha = np.array([1])
    betas = np.array([1, 1])

    for grid_id, gamma_20 in enumerate(np.arange(-1, 1.1, 0.1)):
        gammas = np.array([1, gamma_20])
        grid_params[grid_id] = initialize_params(
            alphas=alpha, betas=betas, deltas=deltas, gammas=gammas, thetas=thetas
        )
    return grid_params


ERROR_STRUCTURE = {
    "homoskedastic": {"deltas": np.array([10, 0]), "thetas": np.array([10, 0, 0])},
    "heteroskedastic_imputation": {
        "deltas": np.array([1, 1]),
        "thetas": np.array([1, 0, 1]),
    },
    "heteroskedastic_regression": {
        "deltas": np.array([1, 1]),
        "thetas": np.array([1, 1, 1]),
    },
}

for error_name, error_params in ERROR_STRUCTURE.items():
    GRID_PARAMS = _generate_grid_params(error_params["deltas"], error_params["thetas"])

    for grid_id, params in GRID_PARAMS.items():

        @task(id=f"{error_name}_{grid_id}")
        def task_simulate_grid(
            params: Annotated[dict, params],
            error_name: Annotated[str, error_name],  # noqa: ARG001
            grid_id: Annotated[int, grid_id],  # noqa: ARG001
        ) -> Annotated[
            pd.DataFrame,
            DATA_CATALOGS["simulation"][f"MC_gamma20_{error_name}_GRID_{grid_id}"],
        ]:
            """Run Monte Carlo simulation for different methods.

            Parameters:
                params (dict): Simulation parameters.
                error_name (str): Name of the error structure.
                grid_id (int): ID of the grid.

            Returns:
                pd.DataFrame: Formatted simulation results.
            """
            _error_handling(params)
            rng = np.random.default_rng(params["random_key"])
            results = {method: [] for method in params["methods"]}
            for _replication in range(params["n_replications"]):
                data = generate_data(params, rng)
                for method in params["methods"]:
                    results[method].append(apply_method(data, method, params))
            data = results_statistics(results, params)
            data["gamma_20"] = params["gamma_coefficients"][1]
            return data
