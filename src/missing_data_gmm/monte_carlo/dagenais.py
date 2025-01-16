"""Dagenais estimation method."""

import numpy as np


def dagenais_weighted_method(data):
    """Apply the weighted Dagenais (FGLS) estimation method.

    Parameters:
        data (dict): Generated data from `_generate_data`.

    Returns:
        dict: Results containing coefficients and standard errors.
    """
    # Step 1: Initial OLS estimates using complete cases
    beta_ols = np.linalg.inv(data["w_complete"].T @ data["w_complete"]) @ (
        data["w_complete"].T @ data["y_complete"]
    )

    # Step 2: Impute x for missing cases
    gamma_hat = np.linalg.inv(data["z_complete"].T @ data["z_complete"]) @ (
        data["z_complete"].T @ data["x_complete"]
    )
    x_missing_hat = data["z_missing"] @ gamma_hat

    # Step 3: Create W_missing by combining x_missing_hat and z_missing
    w_missing = np.column_stack([x_missing_hat, data["z_missing"]])

    # Step 4: Variances for complete and missing cases
    residuals_complete = data["y_complete"] - data["w_complete"] @ beta_ols
    sigma_squared_complete = (residuals_complete.T @ residuals_complete) / data[
        "n_complete"
    ]

    residuals_missing = data["y_missing"] - w_missing @ beta_ols
    sigma_squared_missing = (residuals_missing.T @ residuals_missing) / data[
        "n_missing"
    ]

    # Step 5: Combine complete and missing cases
    w_combined = np.vstack([data["w_complete"], w_missing])
    y_combined = np.concatenate([data["y_complete"], data["y_missing"]])

    # Step 6: Weighted least squares (WLS) estimates
    weight_combined = np.block(
        [
            [
                np.eye(data["n_complete"]) / sigma_squared_complete,
                np.zeros((data["n_complete"], data["n_missing"])),
            ],
            [
                np.zeros((data["n_missing"], data["n_complete"])),
                np.eye(data["n_missing"]) / sigma_squared_missing,
            ],
        ]
    )
    beta_weighted = np.linalg.inv(w_combined.T @ weight_combined @ w_combined) @ (
        w_combined.T @ weight_combined @ y_combined
    )

    # Step 7: Compute standard errors
    residuals_combined = y_combined - w_combined @ beta_weighted
    sigma_squared_combined = (
        residuals_combined.T @ weight_combined @ residuals_combined
    ) / data["n_observations"]
    standard_errors = np.sqrt(
        np.diag(
            sigma_squared_combined
            * np.linalg.inv(w_combined.T @ weight_combined @ w_combined)
        )
    )

    return {"coefficients": beta_weighted, "standard_errors": standard_errors}
