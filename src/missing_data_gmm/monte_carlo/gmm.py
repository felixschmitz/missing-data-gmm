"""GMM estimation methods."""

import numpy as np

from missing_data_gmm.helper import calculate_moments, calculate_residuals


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


def _calculate_initial_weight_matrix(data, beta_complete, gamma_complete):
    # Compute initial weighting matrix
    residuals_complete = calculate_residuals(
        data["y_complete"], data["w_complete"], beta_complete
    )  # epsilon eq 2
    residuals_x_complete = calculate_residuals(
        data["x_complete"], data["z_complete"], gamma_complete
    )  # xi eq 3

    # Residuals for missing cases
    delta_complete = (
        gamma_complete[: data["w_complete"].shape[1] - 1] * beta_complete[0]
        + beta_complete[1 : data["w_complete"].shape[1]]
    )  # coefficient from eq 4 alpha_0 * gamma_0 + beta_0
    residuals_y_missing = calculate_residuals(
        data["y_missing"], data["z_missing"], delta_complete
    )  # eta eq 4

    weight_complete = _inverse_weight_covariance_matrix(
        np.hstack(
            [
                data["w_complete"] * residuals_complete[:, None],
                data["z_complete"] * residuals_x_complete[:, None],
            ]
        )
    ) * (data["n_complete"] / data["n_observations"])

    weight_missing = _inverse_weight_covariance_matrix(
        data["z_missing"] * residuals_y_missing[:, None]
    ) * (data["n_missing"] / data["n_observations"])

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


def _calculate_gradient_matrix(
    data,
    beta_current,
    gamma_current,
):
    return np.vstack(
        [
            np.hstack(
                [
                    data["w_complete"].T
                    @ data["w_complete"]
                    / data["n_observations"],  # G_{11}
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
                    data["z_complete"].T @ data["z_complete"] / data["n_observations"],
                ]
            ),
            np.hstack(
                [
                    (
                        data["z_missing"].T
                        @ data["z_missing"]
                        @ gamma_current
                        / data["n_observations"]
                    ).reshape(-1, 1),  # first element of G_{31}
                    data["z_missing"].T
                    @ data["z_missing"]
                    / data["n_observations"],  # second element of G_{31}
                    data["z_missing"].T
                    @ data["z_missing"]
                    * beta_current[0]
                    / data["n_observations"],  # G_{32}
                ]
            ),
        ]
    )


def _calculate_estimates(data, weight_matrix, theta_current):
    beta_current = theta_current[: data["w_complete"].shape[1]]
    gamma_current = theta_current[data["w_complete"].shape[1] :]

    moments = calculate_moments(data, beta_current, gamma_current, theta_current)

    gradient_matrix = _calculate_gradient_matrix(data, beta_current, gamma_current)

    return theta_current + np.linalg.inv(
        gradient_matrix.T @ weight_matrix @ gradient_matrix
    ) @ (gradient_matrix.T @ weight_matrix @ moments)


def _gmm_descriptive_statistics(data, theta_final):
    beta_final = theta_final[: data["w_complete"].shape[1]]
    residuals_complete = calculate_residuals(
        data["y_complete"], data["w_complete"], beta_final
    )
    residuals_missing = calculate_residuals(
        data["y_missing"],
        data["z_missing"],
        beta_final[1:],
    )
    residuals_combined = np.concatenate([residuals_complete, residuals_missing])
    sigma_squared = (residuals_combined.T @ residuals_combined) / data["n_observations"]
    standard_errors = np.sqrt(
        np.diag(
            sigma_squared * np.linalg.inv(data["w_complete"].T @ data["w_complete"])
        )
    )

    return {
        "coefficients": beta_final,
        "standard_errors": standard_errors,
    }


def gmm_method(data, params, descriptive_statistics=True):
    """Apply the Full GMM Estimation Method.

    Args:
        data (dict): Generated data (from `generate_data`).
        params (dict): Simulation parameters.
        descriptive_statistics (bool): Whether to return descriptive statistics.

    Returns:
        dict: Results containing coefficients and standard errors.
    """
    # Initial OLS estimates using complete cases and imputation coefficients
    beta_complete, gamma_complete = _initial_complete_coefficients(data)
    # Initial parameter estimates
    theta_initial = np.concatenate([beta_complete, gamma_complete])

    weight_matrix = _calculate_initial_weight_matrix(
        data, beta_complete, gamma_complete
    )

    theta_current = theta_initial.copy()
    for _ in range(params["max_iterations"]):
        theta_new = _calculate_estimates(data, weight_matrix, theta_current)

        # Convergence check
        if np.max(np.abs(theta_new - theta_current)) < params.get("tolerance", 1e-6):
            theta_current = theta_new
            break
        theta_current = theta_new

    if descriptive_statistics:
        return _gmm_descriptive_statistics(data, theta_current)
    return data, theta_current, weight_matrix
