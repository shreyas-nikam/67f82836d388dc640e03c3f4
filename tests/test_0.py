import pytest
from definition_edce63b965c94345a0309e469e68ee87 import generate_synthetic_data
import numpy as np
import scipy.sparse as sp


def is_positive_semi_definite(matrix):
    """Check if a matrix is positive semi-definite."""
    try:
        np.linalg.cholesky(matrix)
        return True
    except np.linalg.LinAlgError:
        return False

@pytest.mark.parametrize("n, m, factor_model, expected_type", [
    (5, 2, True, tuple),
    (10, 3, True, tuple),
    (5, None, False, tuple),
    (10, None, False, tuple),
    (1, 1, True, tuple),
    (1, None, False, tuple),
    (5, 5, True, tuple)
])
def test_generate_synthetic_data_types(n, m, factor_model, expected_type):
    result = generate_synthetic_data(n, m, factor_model)
    assert isinstance(result, expected_type)

@pytest.mark.parametrize("n, m, factor_model", [
    (5, 2, True),
    (10, 3, True),
    (5, None, False),
    (10, None, False),
    (1, 1, True),
    (1, None, False)
])
def test_generate_synthetic_data_output(n, m, factor_model):
    result = generate_synthetic_data(n, m, factor_model)
    if factor_model:
        mu, Sigma_tilde, D, F = result
        assert mu.shape == (n, 1)
        assert Sigma_tilde.shape == (m, m)
        assert D.shape == (n, n)
        assert F.shape == (n, m)
        assert sp.issparse(D)
    else:
        mu, Sigma = result
        assert mu.shape == (n, 1)
        assert Sigma.shape == (n, n)

@pytest.mark.parametrize("n, m", [
    (5, 2),
    (10, 3),
    (1, 1)
])
def test_generate_synthetic_data_factor_model_covariance(n, m):
    mu, Sigma_tilde, D, F = generate_synthetic_data(n, m, True)
    Sigma = F.dot(Sigma_tilde).dot(F.T) + D.toarray()
    assert Sigma.shape == (n, n)
    assert is_positive_semi_definite(Sigma)

@pytest.mark.parametrize("n", [
    (5),
    (10),
    (1)
])
def test_generate_synthetic_data_covariance(n):
    mu, Sigma = generate_synthetic_data(n, None, False)
    assert Sigma.shape == (n, n)
    assert is_positive_semi_definite(Sigma)

def test_generate_synthetic_data_seed():
    n = 5
    m = 2
    factor_model = True
    mu1, Sigma_tilde1, D1, F1 = generate_synthetic_data(n, m, factor_model)
    mu2, Sigma_tilde2, D2, F2 = generate_synthetic_data(n, m, factor_model)
    assert np.allclose(mu1, mu2)
    assert np.allclose(Sigma_tilde1, Sigma_tilde2)
    assert np.allclose(D1.toarray(), D2.toarray())
    assert np.allclose(F1, F2)

    factor_model = False
    mu1, Sigma1 = generate_synthetic_data(n, None, factor_model)
    mu2, Sigma2 = generate_synthetic_data(n, None, factor_model)
    assert np.allclose(mu1, mu2)
    assert np.allclose(Sigma1, Sigma2)

@pytest.mark.parametrize("n, m", [
    (2, 3),  # m > n in factor model
    (5, 1)
])
def test_generate_synthetic_data_dimensions_factor_model(n, m):
     mu, Sigma_tilde, D, F = generate_synthetic_data(n, m, True)
     assert F.shape == (n, m)
     assert Sigma_tilde.shape == (m,m)

@pytest.mark.parametrize("input_n", [0, -1])
def test_generate_synthetic_data_invalid_n(input_n):
    with pytest.raises(Exception):
        generate_synthetic_data(input_n, 2, True)
    with pytest.raises(Exception):
        generate_synthetic_data(input_n, None, False)

@pytest.mark.parametrize("input_m", [0, -1])
def test_generate_synthetic_data_invalid_m(n=2, input_m=0):
     with pytest.raises(Exception):
        generate_synthetic_data(n, input_m, True)
