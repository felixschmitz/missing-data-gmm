"""Complete data estimation method."""

import numpy as np


def complete_case_method(data, params):
    """Complete data method estimation.

    Args:
        data (dict): Dictionary containing the complete data.
        params (dict): Dictionary containing the parameters.

    Returns:
        dict: Dictionary containing the estimated coefficients and standard errors.
    """
    # Estimate coefficients using OLS
    beta_hat = np.linalg.inv(data["w_complete"].T @ data["w_complete"]) @ (
        data["w_complete"].T @ data["y_complete"]
    )

    # Calculate residuals and standard errors
    residuals = data["y_complete"] - data["w_complete"] @ beta_hat
    sigma_squared = (residuals.T @ residuals) / params["n_complete"]
    se_beta_hat = np.sqrt(
        np.diag(
            sigma_squared * np.linalg.inv(data["w_complete"].T @ data["w_complete"])
        )
    )

    return {"coefficients": beta_hat, "standard_errors": se_beta_hat}
