
import numpy as np
import scipy.sparse as sp

def generate_synthetic_data(n: int, m: int | None, factor_model: bool) -> tuple[np.ndarray, ...] | tuple[np.ndarray, np.ndarray]:
    """Generates synthetic data for portfolio optimization.

    Args:
        n: Number of assets.
        m: Number of factors (for factor model).
        factor_model: Boolean indicating whether to generate data for a factor model.

    Returns:
        mu: Mean returns.
        Sigma_tilde: Factor covariance matrix (if factor_model=True).
        D: Diagonal matrix representing idiosyncratic risk (if factor_model=True).
        F: Factor loading matrix (if factor_model=True).
        Sigma: Covariance matrix (if factor_model=False).

    Raises:
        Exception: If n or m are non-positive.
    """
    if n <= 0:
        raise Exception("n must be positive")
    if factor_model and (m is None or m <= 0):
         raise Exception("m must be positive")


    np.random.seed(42)  # Set seed for reproducibility
    mu = np.random.rand(n, 1)

    if factor_model:
        F = np.random.rand(n, m)
        Sigma_tilde = np.random.rand(m, m)
        Sigma_tilde = Sigma_tilde.T @ Sigma_tilde + np.eye(m) * 0.01#ensure PSD
        D_values = np.random.rand(n)
        D = sp.diags(D_values, 0)
        return mu, Sigma_tilde, D, F
    else:
        Sigma = np.random.rand(n, n)
        Sigma = Sigma.T @ Sigma + np.eye(n) * 0.01#ensure PSD
        return mu, Sigma
