"""GMM estimation methods."""

import numpy as np


def _initial_complete_coefficients(data):
    # coefficients from eq 2
    beta = np.linalg.inv(data["w_complete"].T @ data["w_complete"]) @ (
        data["w_complete"].T @ data["y_complete"]
    )
    # coefficients from eq 3
    gamma = np.linalg.inv(data["z_complete"].T @ data["z_complete"]) @ (
        data["z_complete"].T @ data["x_complete"]
    )
    return beta, gamma


def _inverse_weight_covariance_matrix(matrix):
    n, _ = matrix.shape
    cov_matrix = (matrix.T @ matrix) / n
    return np.linalg.inv(cov_matrix)


def _calculate_initial_weight_matrix(data, params, beta_complete, gamma_complete):
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

    weight_complete = _inverse_weight_covariance_matrix(
        np.hstack(
            [
                data["w_complete"] * residuals_complete[:, None],
                data["z_complete"] * residuals_x_complete[:, None],
            ]
        )
    ) * (data["n_complete"] / params["n_observations"])

    weight_missing = _inverse_weight_covariance_matrix(
        data["z_missing"] * residuals_y_missing[:, None]
    ) * (data["n_missing"] / params["n_observations"])

    return np.block(
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


def _calculate_moments(data, params, beta_current, gamma_current, theta_current):
    # coefficient from eq 4
    delta_current = (
        gamma_current * theta_current[0] + theta_current[1 : params["k_regressors"]]
    )

    # Residuals for complete cases
    residuals_complete = data["y_complete"] - data["w_complete"] @ beta_current
    residuals_x_complete = data["x_complete"] - data["z_complete"] @ gamma_current
    residuals_y_missing = data["y_missing"] - data["z_missing"] @ delta_current

    return (1 / params["n_observations"]) * np.hstack(
        [
            data["w_complete"].T @ residuals_complete,
            data["z_complete"].T @ residuals_x_complete,
            data["z_missing"].T @ residuals_y_missing,
        ]
    )


def _calculate_gradient_matrix(
    data,
    params,
    beta_current,
    gamma_current,
):
    return np.vstack(
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
                    ).reshape(-1, 1),  # first element of G_{31}
                    data["z_missing"].T
                    @ data["z_missing"]
                    / params["n_observations"],  # second element of G_{31}
                    data["z_missing"].T
                    @ data["z_missing"]
                    * beta_current[0]
                    / params["n_observations"],  # G_{32}
                ]
            ),
        ]
    )


def _calculate_estimates(data, params, weight_matrix, theta_current):
    beta_current = theta_current[: params["k_regressors"]]
    gamma_current = theta_current[params["k_regressors"] :]

    moments = _calculate_moments(
        data, params, beta_current, gamma_current, theta_current
    )

    gradient_matrix = _calculate_gradient_matrix(
        data, params, beta_current, gamma_current, theta_current
    )

    return theta_current + np.linalg.inv(
        gradient_matrix.T @ weight_matrix @ gradient_matrix
    ) @ (gradient_matrix.T @ weight_matrix @ moments)


def _calculate_final_coefficients_and_standard_errors(data, params, theta_current):
    beta_final = theta_current[: params["k_regressors"]]
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


def gmm_method(data, params):
    """Apply the Full GMM Estimation Method.

    Args:
        data (dict): Generated data (from `generate_data`).
        params (dict): Simulation parameters.

    Returns:
        dict: Results containing coefficients and standard errors.
    """
    # Initial OLS estimates using complete cases and imputation coefficients
    beta_complete, gamma_complete = _initial_complete_coefficients(data)
    # Initial parameter estimates
    theta_initial = np.concatenate([beta_complete, gamma_complete])

    weight_matrix = _calculate_initial_weight_matrix(
        data, params, beta_complete, gamma_complete
    )

    theta_current = theta_initial.copy()
    for _ in range(params["max_iterations"]):
        theta_new = _calculate_estimates(data, params, weight_matrix, theta_current)

        # Convergence check
        if np.max(np.abs(theta_new - theta_current)) < params.get("tolerance", 1e-6):
            theta_current = theta_new
            break
        theta_current = theta_new

    return _calculate_final_coefficients_and_standard_errors(
        data, params, theta_current
    )
