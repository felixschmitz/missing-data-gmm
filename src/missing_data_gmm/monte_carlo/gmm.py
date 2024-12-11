"""GMM estimation methods."""

import numpy as np


def gmm_method(data, params):
    """Apply the Full GMM Estimation Method.

    Args:
        data (dict): Generated data (from `generate_data`).
        params (dict): Simulation parameters.

    Returns:
        dict: Results containing coefficients and standard errors.
    """
    # Step 1: Initial OLS estimates using complete cases
    beta_complete = np.linalg.inv(data["w_complete"].T @ data["w_complete"]) @ (
        data["w_complete"].T @ data["y_complete"]
    )  # coefficients from eq 2

    # Step 2: Imputation coefficients
    gamma_complete = np.linalg.inv(data["z_complete"].T @ data["z_complete"]) @ (
        data["z_complete"].T @ data["x_complete"]
    )  # coefficients from eq 3

    # Step 3: Initialize parameters for GMM iteration
    theta_initial = np.concatenate(
        [beta_complete, gamma_complete]
    )  # Initial parameter estimates

    # Compute initial weighting matrix
    residuals_complete = (
        data["y_complete"] - data["w_complete"] @ beta_complete
    )  # epsilon eq 2
    residuals_x_complete = (
        data["x_complete"] - data["z_complete"] @ gamma_complete
    )  # xi eq 3

    # Residuals for missing cases
    delta_complete = (
        gamma_complete[: params["k_regressors"] - 1] * beta_complete[0]
        + beta_complete[1 : params["k_regressors"]]
    )  # coefficient from eq 4 alpha_0 * gamma_0 + beta_0
    residuals_y_missing = (
        data["y_missing"] - data["z_missing"] @ delta_complete
    )  # eta eq 4

    weight_complete = iweight(
        np.hstack(
            [
                data["w_complete"] * residuals_complete[:, None],
                data["z_complete"] * residuals_x_complete[:, None],
            ]
        )
    ) * (data["n_complete"] / params["n_observations"])

    weight_missing = iweight(data["z_missing"] * residuals_y_missing[:, None]) * (
        data["n_missing"] / params["n_observations"]
    )

    weight_matrix = np.block(
        [
            [
                weight_complete,
                np.zeros((weight_complete.shape[0], weight_missing.shape[1])),
            ],
            [
                np.zeros((weight_missing.shape[0], weight_complete.shape[1])),
                weight_missing,
            ],
        ]
    )

    # Iterative GMM estimation
    theta_current = theta_initial.copy()
    for _ in range(params.get("max_iterations", 100)):
        beta_current = theta_current[: params["k_regressors"]]
        gamma_current = theta_current[params["k_regressors"] :]
        delta_current = (
            gamma_current * theta_current[0] + theta_current[1 : params["k_regressors"]]
        )  # coefficient from eq 4
        # Residuals for complete cases
        residuals_complete = data["y_complete"] - data["w_complete"] @ beta_current
        residuals_x_complete = data["x_complete"] - data["z_complete"] @ gamma_current
        residuals_y_missing = data["y_missing"] - data["z_missing"] @ delta_current

        moments = (1 / params["n_observations"]) * np.hstack(
            [
                data["w_complete"].T @ residuals_complete,
                data["z_complete"].T @ residuals_x_complete,
                data["z_missing"].T @ residuals_y_missing,
            ]
        )
        # Gradient matrix
        gradient_matrix = np.vstack(
            [
                np.hstack(
                    [
                        data["w_complete"].T
                        @ data["w_complete"]
                        / params["n_observations"],  # G_{11}
                        np.zeros(
                            (data["w_complete"].shape[1], data["z_complete"].shape[1])
                        ),
                    ]
                ),
                np.hstack(
                    [
                        np.zeros(
                            (data["z_complete"].shape[1], data["w_complete"].shape[1])
                        ),  # G_{22}
                        data["z_complete"].T
                        @ data["z_complete"]
                        / params["n_observations"],
                    ]
                ),
                np.hstack(
                    [
                        (
                            data["z_missing"].T
                            @ data["z_missing"]
                            @ gamma_current
                            / params["n_observations"]
                        ).reshape(-1, 1),  # first part of G_{31}
                        data["z_missing"].T
                        @ data["z_missing"]
                        / params["n_observations"],  # second part of G_{31}
                        data["z_missing"].T
                        @ data["z_missing"]
                        * beta_current[0]
                        / params["n_observations"],  # G_{32}
                    ]
                ),
            ]
        )
        # Update estimates
        theta_new = theta_current + np.linalg.inv(
            gradient_matrix.T @ weight_matrix @ gradient_matrix
        ) @ (gradient_matrix.T @ weight_matrix @ moments)

        # Convergence check
        if np.max(np.abs(theta_new - theta_current)) < params.get("tolerance", 1e-6):
            theta_current = theta_new
            break
        theta_current = theta_new

    # Final estimates
    beta_final = theta_current[: params["k_regressors"]]

    # Compute standard errors
    residuals_combined = np.concatenate(
        [
            data["y_complete"] - data["w_complete"] @ beta_final,
            data["y_missing"] - data["z_missing"] @ beta_final[1:],
        ]
    )
    sigma_squared = (residuals_combined.T @ residuals_combined) / params[
        "n_observations"
    ]
    standard_errors = np.sqrt(
        np.diag(
            sigma_squared * np.linalg.inv(data["w_complete"].T @ data["w_complete"])
        )
    )

    return {
        "coefficients": beta_final,
        "standard_errors": standard_errors,
    }


def iweight(matrix):
    """Compute the weighting matrix."""
    n, k = matrix.shape
    weight_matrix = np.zeros((k, k))
    for j in range(n):
        weight_matrix += np.outer(matrix[j, :], matrix[j, :]) / n
    return np.linalg.inv(weight_matrix)
