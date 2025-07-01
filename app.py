import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.stats as spstats
import cvxpy as cp
import plotly.express as px
# Set page configuration with a dark theme
st.set_page_config(page_title="QuLab", layout="wide", initial_sidebar_state="expanded")

st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
page = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Factor Covariance Model", "Portfolio Optimization"],
    index=0
)
st.sidebar.divider()
st.title("QuLab")
st.divider()


# --- Helper Functions ---
def generate_synthetic_data(n, m=None, factor_model=False):
    """Generates synthetic data for portfolio optimization."""
    np.random.seed(1)
    mu = np.abs(np.random.randn(n, 1))
    if factor_model:
        Sigma_tilde = np.random.randn(m, m)
        Sigma_tilde = Sigma_tilde.T.dot(Sigma_tilde)
        D = sp.diags(np.random.uniform(0, 0.9, size=n))
        F = np.random.randn(n, m)
        return mu, Sigma_tilde, D, F
    else:
        Sigma = np.random.randn(n, n)
        Sigma = Sigma.T.dot(Sigma)
        return mu, Sigma

def solve_portfolio_optimization(mu, Sigma, gamma_value, long_only=True, leverage_limit=None):
    """Solves the Markowitz portfolio optimization problem."""
    n = mu.shape[0]
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)

    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    if leverage_limit is not None:
        constraints.append(cp.norm(w, 1) <= leverage_limit)

    prob = cp.Problem(cp.Maximize(ret - gamma * risk), constraints)
    gamma.value = gamma_value
    prob.solve(solver=cp.SCS, verbose=True)
    risk_value = np.sqrt(risk.value)
    return w.value, risk_value, ret.value

if page == "Overview":
    # --- Streamlit Application ---
    st.title("Portfolio Optimization Lab")

    st.markdown(r"""
    ## Overview

    This application provides an interactive environment for exploring portfolio optimization concepts based on classical Markowitz optimization.

    ## Portfolio Allocation Vector

    The *portfolio allocation vector* defines how an investment budget is distributed across different assets. Let's define:

    *   $w$: the portfolio allocation vector, where $w \in \mathbb{R}^n$.
    *   $w_i$: the fraction of money invested in asset $i$.

    The constraint $1^T w = 1$ ensures that the sum of all allocation fractions equals 1, meaning the entire investment budget is allocated.

    **Short Position:** Borrowing shares to sell now, with the intention of buying them back later. This is denoted by a negative allocation $w_i < 0$.

    **Long-Only Portfolio:** Only investing in assets, without borrowing or shorting. This is denoted by a non-negative allocation $w \geq 0$.

    **Leverage:**

    The leverage is defined as the sum of absolute values of the portfolio weights:

    Where $w_i$ represents the weight of the $i$-th asset in the portfolio.

    ## Asset Returns

    *Asset returns* describe the percentage change in the price of an asset over a period. Let's define:

    *   $p_i > 0$: initial price of asset $i$.
    *   $p_i' > 0$: end-of-period price of asset $i$.
    *   $r_i = (p_i' - p_i) / p_i$: the asset (fractional) return.
    *   $R = r^T w$: the portfolio (fractional) return.

    A common model assumes $r$ is a random variable with mean $E[r] = \mu$ and covariance $E[(r - \mu)(r - \mu)^T] = \Sigma$.

    **Mean Return ($\mu$):** The average expected return of an asset or portfolio.

    **Covariance Matrix ($\Sigma$):** A matrix describing the relationships between the returns of different assets.

    *   $E[R] = \mu^T w$: Expected Portfolio Return.
    *   $var(R) = w^T \Sigma w$: Variance of Portfolio Return.
    *   $std(R) = \sqrt{var(R)}$: Standard Deviation of Portfolio Return.

    ## Classical (Markowitz) Portfolio Optimization

    Classical (Markowitz) portfolio optimization solves the following problem:

    **Objective Function:** Maximizes the risk-adjusted return, balancing expected return ($\mu^T w$) and risk ($w^T \Sigma w$).

    **Risk Aversion ($\gamma$):** A parameter that controls the trade-off between risk and return. Higher values of $\gamma$ indicate greater risk aversion.

    **Constraint:** The sum of the allocation fractions must equal 1.

    **Feasible Set ($W$):** The set of allowed portfolios (e.g., long-only portfolios).
    """)

    # --- User Inputs ---
    num_assets = st.slider("Number of Assets (n)", min_value=5, max_value=50, value=10, step=1)
    risk_aversion = st.slider("Risk Aversion (γ)", min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    long_only = st.checkbox("Long-Only Portfolio", value=True)

    # --- Data Generation ---
    mu, Sigma = generate_synthetic_data(num_assets)

    # --- Portfolio Optimization ---
    w_optimal, risk_optimal, return_optimal = solve_portfolio_optimization(mu, Sigma, risk_aversion, long_only)

    # --- Visualization 1: Risk-Return Trade-off Curve ---
    st.subheader("Risk-Return Trade-off Curve")

    num_samples = 100
    risk_data = np.zeros(num_samples)
    ret_data = np.zeros(num_samples)
    gamma_vals = np.logspace(-2, 3, num=num_samples)

    for i in range(num_samples):
        w_temp, risk_data[i], ret_data[i] = solve_portfolio_optimization(mu, Sigma, gamma_vals[i], long_only)

    # Create the plot using Plotly
    fig1 = px.scatter(x=risk_data, y=ret_data, labels={'x': 'Risk (Standard Deviation)', 'y': 'Return'}, title='Risk-Return Trade-off Curve')

    # Annotate two points on the trade-off curve
    markers_on = [29, 40]  # Example marker indices
    for marker in markers_on:
        fig1.add_annotation(x=risk_data[marker], y=ret_data[marker], text=f'γ = {gamma_vals[marker]:.2f}', showarrow=True, arrowhead=1)

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(r"""
    This chart illustrates the trade-off between risk and return.
    Each point on the curve represents an optimal portfolio for a given level of risk aversion ($\gamma$).
    """)

    # --- Visualization 2: Return Distribution ---
    st.subheader("Return Distribution")

    # Generate return distributions for the marked gamma values
    return_distributions = []
    gamma_values_dist = []
    for midx, idx in enumerate(markers_on):
        x = np.linspace(-2, 5, 1000)
        return_distributions.append(spstats.norm.pdf(x, ret_data[idx], risk_data[idx]))
        gamma_values_dist.append(gamma_vals[idx])

    # Create a single plot with both distributions
    fig2 = px.line(x=x, y=return_distributions, title='Return Distribution for Different Risk Aversion Values')
    fig2.update_layout(xaxis_title="Return", yaxis_title="Density")
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(r"""
    This plot shows the return distributions for two different risk aversion values.
    The shape of the distribution represents the probability of different return outcomes.
    """)

    # --- Visualization 3: Portfolio Optimization with Leverage Limit ---
    st.subheader("Portfolio Optimization with Leverage Limit")

    # Create a Streamlit slider to select the leverage limit
    selected_leverage_limit = st.slider("Select Leverage Limit", min_value=1.0, max_value=5.0, value=2.0, step=0.5)

    num_samples = 100
    gamma_vals = np.logspace(-2, 3, num=num_samples)
    risk_data_leverage = np.zeros(num_samples)
    ret_data_leverage = np.zeros(num_samples)

    for i in range(num_samples):
        w_temp, risk_data_leverage[i], ret_data_leverage[i] = solve_portfolio_optimization(mu, Sigma, gamma_vals[i], long_only, selected_leverage_limit)

    # Plot risk-return trade-off curve with the selected leverage limit
    fig3 = px.scatter(x=risk_data_leverage, y=ret_data_leverage, labels={'x': 'Risk', 'y': 'Return'}, title=f'Risk-Return Trade-off with Leverage Limit = {selected_leverage_limit}')
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(r"""
    This chart shows the risk-return trade-off curve with a leverage limit.
    The leverage limit constrains the sum of the absolute values of the portfolio weights.
    """)

    # --- Visualization 4: Asset Allocation Bar Graph ---
    st.subheader("Asset Allocation Bar Graph")

    # Solve portfolio optimization with the selected leverage limit and risk aversion
    w_optimal, risk_optimal, return_optimal = solve_portfolio_optimization(mu, Sigma, risk_aversion, long_only, selected_leverage_limit)

    # Create a bar chart of the asset allocation
    fig4 = px.bar(x=np.arange(num_assets), y=w_optimal, labels={'x': 'Asset', 'y': 'Weight'}, title='Asset Allocation')
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(r"""
    This bar graph shows the amount of each asset held in the portfolio.
    Negative holdings indicate a short position.
    """)

    # --- Factor Covariance Model Explanation ---
    st.subheader("Factor Covariance Model")

    st.markdown(r"""
    A particularly common and useful variation is to model the covariance matrix (Σ) as a factor model:

    Where:

    *   $F$: the factor loading matrix.
    *   $\Sigma_a$: the factor covariance matrix.
    *   $D$: a diagonal matrix representing idiosyncratic risk.

    **Factor Exposures:**

    A portfolio is *factor j neutral* if $(F^T w)_j = 0$.
    """)


elif page == "Factor Covariance Model":
    from application_pages.factor_model import app
    app()
elif page == "Portfolio Optimization":
    from application_pages.portfolio_optimization import app
    app()


st.divider()

# Disclaimer and copyright at the bottom
st.caption(
    """
    <div class="disclaimer">
        © 2025 QuantUniversity, LLC. All Rights Reserved.<br>
        The purpose of this demonstration is solely for educational use and illustration. Any reproduction of this demonstration requires prior written consent from QuantUniversity, LLC.
    </div>
    """,
    unsafe_allow_html=True
)
