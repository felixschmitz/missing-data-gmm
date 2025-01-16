"""General helper functions."""

import numpy as np


def calculate_residuals(dependent_var, independent_var, coefficients):
    """Calculate residuals from dependent and independent variables and coefficients.

    Args:
        dependent_var (np.ndarray): Dependent variable.
        independent_var (np.ndarray): Independent variable.
        coefficients (np.ndarray): Coefficients.

    Returns:
        np.ndarray: Residuals.
    """
    return dependent_var - independent_var @ coefficients


def calculate_moments(data, beta_current, gamma_current, theta_current):
    """Calculate moments for GMM estimation.

    Args:
        data (dict): Data dictionary.
        beta_current (np.ndarray): Current beta coefficients.
        gamma_current (np.ndarray): Current gamma coefficients.
        theta_current (np.ndarray): Current theta coefficients.

    Returns:
        np.ndarray: Moments.
    """
    # coefficient from eq 4
    delta_current = (
        gamma_current * theta_current[0]
        + theta_current[1 : data["w_complete"].shape[1]]
    )

    # Residuals for complete cases
    residuals_complete = calculate_residuals(
        data["y_complete"], data["w_complete"], beta_current
    )
    residuals_x_complete = calculate_residuals(
        data["x_complete"], data["z_complete"], gamma_current
    )
    residuals_y_missing = calculate_residuals(
        data["y_missing"], data["z_missing"], delta_current
    )

    return (1 / data["n_observations"]) * np.hstack(
        [
            data["w_complete"].T @ residuals_complete,
            data["z_complete"].T @ residuals_x_complete,
            data["z_missing"].T @ residuals_y_missing,
        ]
    )
