import streamlit as st
import numpy as np
import cvxpy as cp
import plotly.express as px
import pandas as pd

def generate_synthetic_data(n):
    np.random.seed(1)
    mu = np.abs(np.random.randn(n, 1))
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
    return w.value, np.sqrt(risk_expr.value), ret_expr.value

def app():
    st.header("Portfolio Optimization")
    st.markdown("""
    This section demonstrates a classical Markowitz portfolio optimization.
    Experiment with the parameters below to observe the trade-offs.
    """)

    num_assets = st.slider("Number of Assets (n)", min_value=5, max_value=50, value=10, step=1)
    risk_aversion = st.slider("Risk Aversion (γ)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    long_only = st.checkbox("Long-Only Portfolio", value=True)

    mu, Sigma = generate_synthetic_data(num_assets)

    w_opt, opt_risk, opt_return = solve_portfolio_optimization(mu, Sigma, risk_aversion, long_only)

    st.subheader("Optimal Portfolio")
    st.write("Weights:", np.round(w_opt, 3))
    st.write("Risk (Std Dev):", np.round(opt_risk, 3))
    st.write("Expected Return:", np.round(opt_return, 3))

    st.subheader("Asset Allocation")
    assets = [f"Asset {i+1}" for i in range(num_assets)]
    fig4 = px.bar(x=assets, y=w_opt, labels={'x': 'Assets', 'y': 'Allocation'}, title="Asset Allocation in Optimal Portfolio")
    st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Trade-off Curves for Different Leverage Limits")
    L_vals = [1, 2, 4]
    num_samples = 100
    gamma_vals = np.logspace(-2, 3, num=num_samples)
    risk_data_multi = np.zeros((len(L_vals), num_samples))
    ret_data_multi = np.zeros((len(L_vals), num_samples))

    for k, L_val in enumerate(L_vals):
        for i in range(num_samples):
            w = cp.Variable(num_assets)
            gamma_param = cp.Parameter(nonneg=True)
            Lmax = cp.Parameter(nonneg=True)
            ret_expr = mu.T @ w
            risk_expr = cp.quad_form(w, Sigma)
            constraints = [cp.sum(w) == 1, cp.norm(w, 1) <= Lmax]
            prob = cp.Problem(cp.Maximize(ret_expr - gamma_param * risk_expr), constraints)
            gamma_param.value = gamma_vals[i]
            Lmax.value = L_val
            prob.solve(solver=cp.SCS)
            risk_data_multi[k, i] = np.sqrt(risk_expr.value)
            ret_data_multi[k, i] = ret_expr.value

    df_multi = pd.DataFrame({
        'Risk': np.concatenate(risk_data_multi),
        'Return': np.concatenate(ret_data_multi),
        'Leverage Limit': np.repeat(L_vals, num_samples)
    })

    fig5 = px.line(df_multi, x='Risk', y='Return', color='Leverage Limit', markers=True, title='Risk-Return Trade-off for Different Leverage Limits')
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Asset Allocation for Different Leverage Limits (with Risk Bound)")
    w = cp.Variable(num_assets)
    Lmax = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    prob = cp.Problem(cp.Maximize(ret), [cp.sum(w) == 1, cp.norm(w, 1) <= Lmax, risk <= 2])

    w_vals = []
    for L_val in L_vals:
        Lmax.value = L_val
        prob.solve()
        w_vals.append(w.value)

    indices = np.argsort(mu.flatten())
    assets = np.arange(1, num_assets + 1)

    df_weights = pd.DataFrame({
        'Asset': np.tile(assets[indices], len(L_vals)),
        'Weight': np.concatenate([w_vals[idx][indices] for idx in range(len(L_vals))]),
        'Leverage Limit': np.repeat(L_vals, num_assets)
    })

    fig6 = px.bar(df_weights, x='Asset', y='Weight', color='Leverage Limit', barmode='group', title='Portfolio Weights for Different Leverage Limits (with Risk Bound)')
    st.plotly_chart(fig6, use_container_width=True)

if __name__ == '__main__':
    app()