import streamlit as st
import numpy as np
import cvxpy as cp
from scipy.stats import norm
import plotly.graph_objects as go

def generate_synthetic_data(n):
    np.random.seed(1)
    # Generate positive mean returns.
    mu = np.abs(np.random.randn(n, 1))
    # Generate a random covariance matrix (positive semi-definite).
    Sigma = np.random.randn(n, n)
    Sigma = Sigma.T.dot(Sigma)
    return mu, Sigma

def solve_portfolio_optimization(mu, Sigma, gamma_value, long_only=True):
    n = mu.shape[0]
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret_expr = mu.T @ w
    risk_expr = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(cp.Maximize(ret_expr - gamma * risk_expr), constraints)
    gamma.value = gamma_value
    prob.solve(solver=cp.SCS, verbose=True)

    # Evaluate risk from the value of quadratic form.
    return w.value, np.sqrt(risk_expr.value), ret_expr.value

def app():
    st.header("Portfolio Optimization")
    st.markdown("""
    This section demonstrates a classical Markowitz portfolio optimization.
    Experiment with the parameters below to observe the risk-return trade-off.
    """)
    
    # User inputs.
    num_assets = st.slider("Number of Assets (n)", min_value=5, max_value=50, value=10, step=1)
    risk_aversion = st.slider("Risk Aversion (γ)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    long_only = st.checkbox("Long-Only Portfolio", value=True)
    
    # Generate synthetic data.
    mu, Sigma = generate_synthetic_data(num_assets)
    
    # Compute the optimal portfolio for the selected risk aversion.
    w_opt, opt_risk, opt_return = solve_portfolio_optimization(mu, Sigma, risk_aversion, long_only)
    st.subheader("Optimal Portfolio")
    st.write("Weights:", np.round(w_opt, 3))
    st.write("Risk (Std Dev):", np.round(opt_risk, 3))
    st.write("Expected Return:", np.round(opt_return, 3))
    
    # Create risk-return trade-off curve.
    st.subheader("Risk-Return Trade-off Curve")
    num_samples = 100
    gamma_vals = np.logspace(-2, 3, num=num_samples)
    risk_data = []
    ret_data = []
    for g in gamma_vals:
        _, r_val, ret_val = solve_portfolio_optimization(mu, Sigma, g, long_only)
        risk_data.append(r_val)
        ret_data.append(ret_val)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=risk_data, y=ret_data, mode='lines', name='Trade-off Curve'))
    # Annotate two sample points.
    marker_indices = [29, 40]
    for idx in marker_indices:
        fig.add_trace(go.Scatter(x=[risk_data[idx]], y=[ret_data[idx]],
                                 mode='markers+text',
                                 marker=dict(size=10, color='red'),
                                 text=[f"γ={gamma_vals[idx]:.2f}"],
                                 textposition="top center"))
    fig.update_layout(xaxis_title="Risk (Std Dev)",
                      yaxis_title="Expected Return",
                      title="Risk-Return Trade-off Curve")
    st.plotly_chart(fig, use_container_width=True)
    
    # Plot return distributions for the two marked risk aversion values.
    st.subheader("Return Distribution")
    x_vals = np.linspace(min(ret_data)-2, max(ret_data)+2, 1000)
    fig2 = go.Figure()
    for idx in marker_indices:
        mu_val = ret_data[idx]
        sigma_val = risk_data[idx]
        pdf_vals = norm.pdf(x_vals, loc=mu_val, scale=sigma_val)
        fig2.add_trace(go.Scatter(x=x_vals, y=pdf_vals,
                                  mode='lines', name=f"γ={gamma_vals[idx]:.2f}"))
    fig2.update_layout(xaxis_title="Return", yaxis_title="Density",
                       title="Return Distributions for Selected γ Values")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Portfolio optimization with leverage limit demonstration.
    st.subheader("Portfolio Optimization with Leverage Limit")
    selected_leverage = st.slider("Select Leverage Limit", min_value=1, max_value=4, value=1, step=1)
    risk_data_leverage = []
    ret_data_leverage = []
    for g in gamma_vals:
        n = num_assets
        w = cp.Variable(n)
        gamma_param = cp.Parameter(nonneg=True)
        Lmax = cp.Parameter(nonneg=True)
        ret_expr = mu.T @ w
        risk_expr = cp.quad_form(w, Sigma)
        constraints = [cp.sum(w) == 1, cp.norm(w, 1) <= Lmax]
        prob = cp.Problem(cp.Maximize(ret_expr - gamma_param * risk_expr), constraints)
        gamma_param.value = g
        Lmax.value = selected_leverage
        prob.solve(solver=cp.SCS, verbose=True)
        risk_data_leverage.append(np.sqrt(risk_expr.value))
        ret_data_leverage.append(ret_expr.value)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=risk_data_leverage, y=ret_data_leverage,
                              mode='lines', name=f'Leverage Limit = {selected_leverage}'))
    fig3.update_layout(xaxis_title="Risk (Std Dev)", yaxis_title="Expected Return",
                       title="Risk-Return Trade-off with Leverage Limit")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Asset Allocation Bar Graph.
    st.subheader("Asset Allocation")
    assets = [f"Asset {i+1}" for i in range(num_assets)]
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=assets, y=w_opt, name="Asset Weights"))
    fig4.update_layout(xaxis_title="Assets", yaxis_title="Allocation",
                      title="Asset Allocation in Optimal Portfolio")
    st.plotly_chart(fig4, use_container_width=True)

    
    
if __name__ == '__main__':
    app()
