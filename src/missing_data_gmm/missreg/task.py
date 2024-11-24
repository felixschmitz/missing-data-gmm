from pathlib import Path
from typing import Annotated

import jax.numpy as jnp
import pandas as pd
import statsmodels.api as sm
from jax import jit
from jax.scipy.optimize import minimize

from missing_data_gmm.config import DATA, DATA_CATALOGS, DATA_NAMES, SEX_NAMES


def _data_loading_preprocessing(raw_path, sex_name):
    raw_data = pd.read_stata(raw_path)
    raw_data[["male", "bmimissing"]] = raw_data[["male", "bmimissing"]].astype(
        "uint8[pyarrow]"
    )
    male_dummy = 0 if "not" in sex_name else 1
    return raw_data[raw_data["male"] == male_dummy].reset_index(drop=True)


def _complete_regression(data):
    X_complete = sm.add_constant(data[["bmirating", "iq"]])
    y_complete = data["educ"]
    return sm.OLS(y_complete, X_complete).fit(cov_type="HC0")


def _dummy_regression(data):
    X_dummy = sm.add_constant(data[["bmirating", "iq", "bmimissing"]]).astype(
        {"bmimissing": "float64"}
    )
    y_dummy = data["educ"]
    return sm.OLS(y_dummy, X_dummy).fit(cov_type="HC0")


def gmm_moment_conditions(params, X, y, missing):
    """Moment conditions for GMM."""
    beta_0, alpha, beta_iq, gamma_0, gamma_iq = params

    # Define residuals
    residual1 = (1 - missing) * (y - (beta_0 + alpha * X[:, 0] + beta_iq * X[:, 1]))
    residual2 = (1 - missing) * (X[:, 0] - (gamma_0 + gamma_iq * X[:, 1]))
    residual3 = missing * (
        y - ((gamma_0 * alpha + beta_0) + (gamma_iq * alpha + beta_iq) * X[:, 1])
    )

    # Define instruments for each residual
    instrument1 = X  # Instruments for residual1: [bmirating, iq]
    instrument2 = X[:, 1:2]  # Instruments for residual2: [iq]
    instrument3 = X[:, 1:2]  # Instruments for residual3: [iq]

    # Compute moments for each residual
    moment1 = residual1[:, None] * instrument1  # Shape: (n_obs, n_instruments1)
    moment2 = residual2[:, None] * instrument2  # Shape: (n_obs, n_instruments2)
    moment3 = residual3[:, None] * instrument3  # Shape: (n_obs, n_instruments3)

    # Stack all moments into a single array
    moments = jnp.hstack([moment1, moment2, moment3])  # Shape: (n_obs, n_moments)

    return moments


@jit
def gmm_objective(params, X, y, missing, weight_matrix):
    """Objective function for GMM."""
    moments = gmm_moment_conditions(params, X, y, missing)  # Shape: (n_obs, n_moments)
    mean_moments = jnp.mean(moments, axis=0)  # Shape: (n_moments,)
    return mean_moments.T @ weight_matrix @ mean_moments


def gmm_estimation(data, max_iter=5, tol=1e-6):
    """Estimate parameters using iterated GMM."""
    # Extract data
    X = data[["bmirating", "iq"]].to_numpy()
    y = data["educ"].to_numpy()
    missing = data["bmimissing"].to_numpy()

    # Instruments: bmirating + iq (group 1), iq (groups 2 and 3)
    instruments = jnp.column_stack([X[:, 0], X[:, 1], X[:, 1], X[:, 1]])

    # Initial parameter guesses (e.g., zeros)
    initial_params = jnp.zeros(5)  # [beta_0, alpha, beta_iq, gamma_0, gamma_iq]

    # Initial weight matrix: Identity matrix
    weight_matrix = jnp.eye(instruments.shape[1])

    # Iterative GMM
    for i in range(max_iter):
        # Minimize the GMM objective function
        result = minimize(
            lambda params: gmm_objective(params, X, y, missing, weight_matrix),
            initial_params,
            method="BFGS",
        )

        if not result.success:
            raise ValueError(f"GMM estimation failed: {result.message}")

        # Update parameters
        params = result.x

        # Compute moments with current parameters
        moments = gmm_moment_conditions(params, X, y, missing)

        # Update weight matrix (covariance of moments)
        weight_matrix_new = inv(jnp.cov(moments, rowvar=False))

        # Check for convergence
        if jnp.linalg.norm(weight_matrix - weight_matrix_new) < tol:
            break

        weight_matrix = weight_matrix_new
    else:
        raise ValueError(
            "Iterative GMM did not converge within the maximum number of iterations."
        )

    return params, result.fun, weight_matrix


for data_name in DATA_NAMES:
    raw_path = Path(DATA / "missreg" / f"{data_name}.dta")
    for sex_name in SEX_NAMES:

        def task_missing_data_methods(
            raw_path: Annotated[Path, raw_path], sex_name: Annotated[str, sex_name]
        ) -> Annotated[
            pd.DataFrame, DATA_CATALOGS[data_name][sex_name]["methods-outputs"]
        ]:
            """Real world data example for missing data regression.

            Args:
                raw_path: Path to the raw data file.
                sex_name: Gender-specific subset name.

            Returns:
                A DataFrame containing results for the three regression methods.
            """
            data = _data_loading_preprocessing(raw_path, sex_name)
            results = []

            # Complete-data regression
            model_complete = _complete_regression(data[data["bmimissing"] == 0])
            results.append(
                {
                    "method": "complete-data",
                    "coefficients": model_complete.params,
                    "summary": model_complete.summary(),
                }
            )

            # Dummy-variable method
            model_dummy = _dummy_regression(data)
            results.append(
                {
                    "method": "dummy-variable",
                    "coefficients": model_dummy.params,
                    "summary": model_dummy.summary(),
                }
            )

            # GMM method
            gmm_params, gmm_obj_value, weight_matrix = gmm_estimation(data)
            results.append(
                {
                    "method": "gmm",
                    "coefficients": gmm_params,
                    "objective_value": gmm_obj_value,
                }
            )

            return pd.DataFrame(results)
