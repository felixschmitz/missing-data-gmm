"""Dummy variable estimation method."""

import numpy as np


def dummy_variable_method(data):
    """Dummy variable method estimation.

    Args:
        data (dict): Dictionary containing the complete and missing data.

    Returns:
        dict: Dictionary containing the estimated coefficients and standard errors.
    """
    # Construct the design matrix with dummy variables
    w_dummy = np.column_stack(
        (
            np.concatenate([data["x_complete"], np.zeros(data["n_missing"])]),
            np.concatenate([data["z_complete"], data["z_missing"]]),
        )
    )

    # Estimate coefficients using OLS
    beta_hat = np.linalg.inv(w_dummy.T @ w_dummy) @ (w_dummy.T @ data["y"])

    # Calculate standard errors
    residuals = data["y"] - w_dummy @ beta_hat
    sigma_squared = residuals.T @ residuals / data["n_observations"]
    se_beta_hat = np.sqrt(np.diag(sigma_squared * np.linalg.inv(w_dummy.T @ w_dummy)))

    return {"coefficients": beta_hat, "standard_errors": se_beta_hat}
