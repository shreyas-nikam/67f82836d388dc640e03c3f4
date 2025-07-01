import pytest
from definition_847a705cd3bf4b9b9d80b1158deb588a import generate_synthetic_data

@pytest.mark.parametrize("n, m, factor_model, expected_type", [
    (5, None, False, tuple),
    (10, 3, True, tuple),
    (1, 1, True, tuple),
    (0, None, False, TypeError),
    (-5, None, False, TypeError),
    (10, -2, True, TypeError),
])
def test_generate_synthetic_data(n, m, factor_model, expected_type):
    if n <= 0:
        with pytest.raises(Exception):
            generate_synthetic_data(n, m, factor_model)
    elif factor_model:
        if m is not None and m <= 0:
            with pytest.raises(Exception):
                generate_synthetic_data(n, m, factor_model)
        else:
            result = generate_synthetic_data(n, m, factor_model)
            assert isinstance(result, tuple)
            assert len(result) >= 4
            mu, Sigma_tilde, D, F = result
            assert mu.shape == (n, 1)
            assert Sigma_tilde.shape == (m, m)
            assert D.shape == (n, n)
            assert F.shape == (n, m)
    else:
        result = generate_synthetic_data(n, m, factor_model)
        assert isinstance(result, tuple)
        assert len(result) == 2
        mu, Sigma = result
        assert mu.shape == (n, 1)
        assert Sigma.shape == (n, n)
        # Check positive semi-definiteness
        eigvals = np.linalg.eigvals(Sigma)
        assert np.all(eigvals >= -1e-8)
