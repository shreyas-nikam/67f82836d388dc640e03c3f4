import streamlit as st
import numpy as np
import scipy.stats as stats
import cvxpy as cp
import matplotlib.pyplot as plt

# Helper functions
def generate_synthetic_data(n, m=3, factor_model=True):
    np.random.seed(42)
    mu = np.abs(np.random.randn(n, 1))
    if factor_model:
        # Generate synthetic factor model data
        F = np.random.randn(n, m)
        Sigma_tilde = np.random.randn(m, m)
        Sigma_tilde = Sigma_tilde @ Sigma_tilde.T  # positive semi-definite
        D = np.diag(np.random.uniform(0, 0.1, size=n))
        return mu, F, Sigma_tilde, D
    else:
        Sigma = np.random.randn(n, n)
        Sigma = Sigma @ Sigma.T
        return mu, Sigma

def solve_markowitz(mu, Sigma, gamma, long_only=True):
    n = len(mu)
    w = cp.Variable(n)
    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), constraints)
    prob.solve()
    return w.value

# Main App
def run_page1():
    st.title("Portfolio Optimization Lab")
    st.markdown("This interactive application allows you to explore classical portfolio optimization concepts based on Markowitz's model.")

    n_assets = st.slider("Number of Assets", 5, 50, 10)
    gamma = st.slider("Risk Aversion Parameter (γ)", 0.01, 10.0, 1.0, step=0.01)

    # Generate synthetic data
    mu, F, Sigma_tilde, D = generate_synthetic_data(n_assets)

    # Calculate covariance matrix
    Sigma = F @ Sigma_tilde @ F.T + D

    # Solve for optimal portfolio
    w_opt = solve_markowitz(mu, Sigma, gamma, long_only=True)

    # Calculate risk and return
    port_return = float(mu.T @ w_opt)
    port_risk = float(np.sqrt(w_opt.T @ Sigma @ w_opt))

    # Display results
    st.subheader("Optimal Portfolio")
    st.write("Expected Return:", round(port_return, 4))
    st.write("Risk (Std Dev):", round(port_risk, 4))
    st.write("Asset Weights:")
    for i, weight in enumerate(w_opt):
        st.write(f"Asset {i+1}: {weight:.3f}")

    # Plot Risk-Return trade-off curve
    risk_vals = []
    return_vals = []
    gamma_vals = np.logspace(-2, 2, 50)
    for g in gamma_vals:
        w = solve_markowitz(mu, Sigma, g, long_only=True)
        r = mu.T @ w
        s = np.sqrt(w.T @ Sigma @ w)
        risk_vals.append(float(s))
        return_vals.append(float(r))
    fig, ax = plt.subplots()
    ax.plot(risk_vals, return_vals, marker='o')
    ax.set_xlabel("Risk (Std Dev)")
    ax.set_ylabel("Expected Return")
    ax.set_title("Risk vs Return Trade-off Curve")
    st.pyplot(fig)

    # Return nothing
