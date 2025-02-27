"""Helper functions for the probability density function calculation."""

import numpy as np
import pandas as pd


def initialize_params(
    alphas: np.ndarray,
    betas: np.ndarray,
    deltas: np.ndarray,
    gammas: np.ndarray,
    thetas: np.ndarray,
) -> dict:
    """Initialize parameters for the Monte Carlo simulation.

    Returns:
        dict: Parameters for the Monte Carlo simulation.
    """
    params = {}
    params["design"] = 1
    params["methods"] = ["Dummy case method", "GMM"]
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
    params["exponential"] = False  # Exponential heteroskedasticity

    params["alpha_coefficients"] = alphas
    params["beta_coefficients"] = betas
    params["delta_coefficients"] = deltas
    params["gamma_coefficients"] = gammas
    params["theta_coefficients"] = thetas

    params["max_iterations"] = 200  # number of max iterations of gmm
    params["random_key"] = 123456
    return params


def _calculate_statistics(
    coefficients, true_values, parameter_names, method, n_observations
) -> pd.DataFrame:
    mean_estimates = np.mean(coefficients, axis=0)
    mean_biases = mean_estimates - true_values
    n_vars = n_observations * np.var(coefficients, axis=0, ddof=1)
    mses = mean_biases**2 + (n_vars / n_observations)

    return pd.DataFrame(
        {
            "Method": method,
            "Parameter": parameter_names,
            "Bias": mean_biases,
            "n*Var": n_vars,
            "MSE": mses,
        }
    )


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

    all_results = []
    for method, method_results in results.items():
        coefficients = np.array([entry["coefficients"] for entry in method_results])
        true_coefficients = np.concatenate(
            [params["alpha_coefficients"], params["beta_coefficients"]]
        )
        stats_df = _calculate_statistics(
            coefficients=coefficients,
            true_values=true_coefficients,
            parameter_names=parameters,
            method=method,
            n_observations=params["n_observations"],
        )
        all_results.append(stats_df)
    return pd.concat(all_results, axis=0).reset_index(drop=True)
