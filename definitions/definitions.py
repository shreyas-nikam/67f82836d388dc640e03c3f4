
import numpy as np
from typing import Tuple, Optional, Union

def generate_synthetic_data(
    n: int,
    m: Optional[int] = None,
    factor_model: bool = False
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates synthetic data for portfolio optimization including mean returns
    and covariance matrices or factor model components.
    
    Args:
        n (int): Number of assets.
        m (Optional[int]): Number of factors for factor model. Defaults to None.
        factor_model (bool): Whether to generate factor model data. Defaults to False.
        
    Returns:
        Tuple:
            - If factor_model is False:
                mu (np.ndarray): Mean returns shape (n, 1)
                Sigma (np.ndarray): Covariance matrix shape (n, n)
            - If factor_model is True:
                mu (np.ndarray): Mean returns shape (n, 1)
                Sigma_tilde (np.ndarray): Factor covariance shape (m, m)
                D (np.ndarray): Diagonal risk matrix shape (n, n)
                F (np.ndarray): Factor loadings shape (n, m)
    Raises:
        ValueError: If inputs are invalid.
    """
    # Validate inputs
    if not isinstance(n, int):
        raise TypeError("n must be an integer.")
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if factor_model:
        if m is None:
            m = max(1, n // 2)
        if not isinstance(m, int):
            raise TypeError("m must be an integer.")
        if m <= 0:
            raise ValueError("m must be a positive integer.")
    else:
        m = None

    # Generate mean returns
    mu = np.random.uniform(-0.05, 0.05, size=(n, 1))
    
    if not factor_model:
        # Generate a random positive semi-definite covariance matrix
        A = np.random.randn(n, n)
        Sigma = A @ A.T
        # Add small epsilon for numerical stability
        Sigma += np.eye(n) * 1e-6
        return mu, Sigma
    else:
        # Generate factor covariance matrix
        A_m = np.random.randn(m, m)
        Sigma_tilde = A_m @ A_m.T
        Sigma_tilde += np.eye(m) * 1e-6

        # Generate diagonal risk matrix D with positive entries
        D_diag = np.random.uniform(0.01, 0.1, size=n)
        D = np.diag(D_diag)

        # Generate factor loadings F
        F = np.random.uniform(-1, 1, size=(n, m))
        return mu, Sigma_tilde, D, F
