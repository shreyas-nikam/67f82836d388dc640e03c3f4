# Technical Specifications for a Streamlit Application: Custom Portfolio Optimization Lab

## Overview

This Streamlit application will provide an interactive environment for users to explore portfolio optimization concepts based on the provided document. It will guide users through the process of defining portfolio allocation vectors, understanding asset returns and risk, and applying classical Markowitz optimization techniques. The application incorporates user interaction through widgets to allow experimentation with portfolio parameters, and visualizes the results using dynamic charts and graphs.  All concepts, code, and visualizations are explained in detail using markdown, formulae, and examples.

## Step-by-Step Generation Process

1.  **Layout Initialization:**  The application will begin with a main title, followed by an introduction to portfolio optimization.  This introduction will define the goals of portfolio optimization.

2.  **Portfolio Allocation Vector Definition:**

    *   Display a markdown section explaining the concept of a *portfolio allocation vector*.
    *   Define `w` as the portfolio allocation vector, where `w ∈ R^n`.
    *   Explain that `wi` represents the fraction of money invested in asset `i`.
    *   Explain the constraint `1^T w = 1`.  
        ```markdown
        The constraint ensures that the sum of all allocation fractions equals 1, meaning the entire investment budget is allocated.
        ```
    *   Explain short positions (`wi < 0`) and long-only portfolios (`w >= 0`).
        ```markdown
        **Short Position:** Borrowing shares to sell now, with the intention of buying them back later. This is denoted by a negative allocation `w_i < 0`.
        **Long-Only Portfolio:** Only investing in assets, without borrowing or shorting. This is denoted by a non-negative allocation `w >= 0`.
        ```
    *   Define and explain leverage using the formula:
        ```markdown
        ||w||_1 = 1^T w_+ + 1^T w_
        ```
        ```markdown
        Where w_+ represents the long positions and w_- represents the short positions. This is known as leverage.
        ```

3.  **Asset Returns and Risk Explanation:**

    *   Display a markdown section explaining *asset returns*.
    *   Define initial prices `pi > 0` and end-of-period prices `pi' > 0`.
    *   Define the asset (fractional) return `ri = (pi' - pi) / pi`.
    *   Define the portfolio (fractional) return `R = r^T w`.
    *   Explain the common model where `r` is a random variable with mean `E[r] = μ` and covariance `E[(r - μ)(r - μ)^T] = Σ`.
        ```markdown
        **Mean Return (μ):** The average expected return of an asset or portfolio.
        **Covariance Matrix (Σ):** A matrix describing the relationships between the returns of different assets.
        ```
    *   Explain `E[R] = μ^T w` and `var(R) = w^T Σ w`.
        ```markdown
        **Expected Portfolio Return (E[R]):** The weighted average of the expected returns of the assets in the portfolio.
        **Variance of Portfolio Return (var(R)):** A measure of the risk or uncertainty associated with the portfolio's return.
        ```
    *   Define standard deviation `std(R) = sqrt(var(R))` as a measure of risk.

4.  **Classical (Markowitz) Portfolio Optimization Section:**

    *   Display a markdown section explaining *Classical (Markowitz) portfolio optimization*.
    *   Explain that it solves the following optimization problem:
        ```markdown
        maximize  μ^T w - γ w^T Σ w
        subject to 1^T w = 1,  w ∈ W
        ```
        ```markdown
        **Objective Function:** Maximizes the risk-adjusted return, balancing expected return (μ^T w) and risk (w^T Σ w).
        **Risk Aversion (γ):** A parameter that controls the trade-off between risk and return. Higher values of γ indicate greater risk aversion.
        **Constraint:** The sum of the allocation fractions must equal 1.
        **Feasible Set (W):** The set of allowed portfolios (e.g., long-only portfolios).
        ```
    *   Explain that `w` is the optimization variable, `W` is the set of allowed portfolios, and `γ > 0` is the risk aversion parameter.
    *   Explain the risk-adjusted return and how varying `γ` affects the risk-return trade-off.

5.  **Example Implementation (Interactive Trade-off Curve):**

    *   Use Streamlit widgets to allow the user to specify the number of assets (`n`). Use a default value of 10.
    *   Use `numpy` to generate synthetic data for `mu` (mean returns) and `Sigma` (covariance matrix). Ensure `Sigma` is positive semi-definite.
    *   Use `cvxpy` to define the optimization problem.  Include a risk aversion parameter (`gamma`) as a `cp.Parameter()` that the user can control with a Streamlit slider.  Set the range of the slider from 0.01 to 10.0 (log scale).
    *   Solve the optimization problem for a range of `gamma` values to generate the risk-return trade-off curve.
    *   Use `matplotlib` to plot the risk-return trade-off curve as a scatter plot.  Risk (standard deviation) will be on the x-axis, and return will be on the y-axis. Annotate two points on the curve corresponding to different `gamma` values for visual representation. Use interactive tooltips to display the specific risk and return values at any point on the curve. This corresponds to Graph 1 from the document.
    *  Explain the meaning of the trade-off curve and the effect of the risk aversion parameter.

6.  **Return Distribution Plot:**

    *   Plot the return distributions for the two risk aversion values marked on the trade-off curve (Graph 2).
    *   Use `scipy.stats.norm.pdf` to plot the normal distributions.  Annotate each distribution with its corresponding gamma value.
    *   Explain the significance of the return distributions, particularly the probability of loss for different risk aversion values.

7. **Portfolio Constraints Exploration**
    *   Implement and demonstrate portfolio optimization with a leverage limit (Graph 3). Provide users with a slider to adjust the leverage limit.
    *   Plot the risk-return trade-off curves for different leverage limits on the same graph for comparison.

8. **Asset Allocation Bar Graph**
    *   Display a bar graph showing the amount of each asset held in each portfolio (Graph 4).
    *   The bar graph will allow exploration and demonstration of variations in short positions in different portfolios.

9. **Factor Covariance Model Explanation:**

    *   Display a markdown section explaining the *Factor Covariance Model*.
    *   Explain that the covariance matrix is modeled as `Σ = F Σ~ F^T + D`.
    *   Define `F` as the factor loading matrix, `Σ~` as the factor covariance matrix, and `D` as a diagonal matrix representing idiosyncratic risk.
    *   Explain the concept of factor exposures (`F^T w`) and factor neutrality.

10. **Portfolio Optimization with Factor Covariance Model:**

    *   Present the optimization problem using the factor covariance model:
        ```markdown
        maximize  μ^T w - γ (f^T Σ~ f + w^T D w)
        subject to 1^T w = 1,  f = F^T w
        w ∈ W,  f ∈ F
        ```
    *   Explain that `w` represents the allocations, `f` the factor exposures, and `F` the factor exposure constraints.
    *   Discuss the computational advantages of using the factor covariance model.

11. **Factor Model Example:**

    *   Implement and display code for generating synthetic data for the factor model.
    *   Implement and solve the portfolio optimization problem using both the standard covariance model and the factor covariance model.
    *   Compare the solve times for both methods.

## Important Definitions, Examples, and Formulae

This application will include markdown explanations for the following key concepts, using formulae and examples where appropriate:

*   **Portfolio Allocation Vector (w):**  A vector representing the proportion of investment allocated to each asset.  `w ∈ R^n` where `n` is the number of assets.
*   **Short Position:**  Borrowing an asset to sell, expecting its price to decrease. Represented by `wi < 0`.
*   **Long-Only Portfolio:**  Investing only in assets, without shorting. Represented by `w >= 0`.
*   **Leverage:** `||w||_1 = 1^T w_+ + 1^T w_`, where w_+ represents the long positions and w_- represents the short positions.
*   **Asset Return (ri):**  The percentage change in the price of an asset over a period. `ri = (pi' - pi) / pi`.
*   **Portfolio Return (R):**  The weighted average of the returns of assets in a portfolio. `R = r^T w`.
*   **Expected Return (E[R]):**  The average return expected from a portfolio. `E[R] = μ^T w`.
*   **Variance of Return (var(R)):**  A measure of the risk associated with a portfolio's return. `var(R) = w^T Σ w`.
*   **Standard Deviation of Return (std(R)):**  The square root of the variance, providing a more interpretable measure of risk. `std(R) = sqrt(var(R))`.
*   **Risk Aversion Parameter (γ):** A parameter that controls the trade-off between risk and return in portfolio optimization. Higher values indicate greater risk aversion.
*   **Factor Covariance Model:**  A method for modeling the covariance matrix using a smaller number of factors. `Σ = F Σ~ F^T + D`.

## Libraries and Tools

*   **Streamlit:** Used for building the interactive web application.  Streamlit provides the framework for creating the user interface, managing user inputs, and displaying visualizations.
*   **NumPy:** Used for numerical computations, including generating synthetic data, performing matrix operations, and calculating statistical measures.
*   **SciPy:** Specifically `scipy.sparse` and `scipy.stats`, used for sparse matrix operations and statistical calculations, respectively.
*   **cvxpy:** Used for defining and solving the convex optimization problem for portfolio allocation.  cvxpy allows the user to express the optimization problem in a natural mathematical syntax.
*   **Matplotlib:** Used for creating visualizations, including the risk-return trade-off curve and the return distribution plots.

## Appendix Code

```python
import streamlit as st
import numpy as np
import scipy.sparse as sp
import scipy.stats as spstats
import cvxpy as cp
import matplotlib.pyplot as plt

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

def solve_portfolio_optimization(mu, Sigma, gamma_value, long_only=True):
    """Solves the Markowitz portfolio optimization problem."""
    n = mu.shape[0]
    w = cp.Variable(n)
    gamma = cp.Parameter(nonneg=True)
    ret = mu.T @ w
    risk = cp.quad_form(w, Sigma)
    constraints = [cp.sum(w) == 1]
    if long_only:
        constraints.append(w >= 0)
    prob = cp.Problem(cp.Maximize(ret - gamma * risk), constraints)
    gamma.value = gamma_value
    prob.solve()
    return w.value, np.sqrt(risk).value, ret.value


# --- Streamlit Application ---
st.title("Interactive Portfolio Optimization Lab")

st.markdown("""
## Overview

This application provides an interactive environment for exploring portfolio optimization concepts based on classical Markowitz optimization.

## Portfolio Allocation Vector

The *portfolio allocation vector* defines how an investment budget is distributed across different assets. Let's define:

*   `w`: the portfolio allocation vector, where `w ∈ R^n`.
*   `wi`: the fraction of money invested in asset `i`.

The constraint `1^T w = 1` ensures that the sum of all allocation fractions equals 1, meaning the entire investment budget is allocated.

**Short Position:** Borrowing shares to sell now, with the intention of buying them back later. This is denoted by a negative allocation `w_i < 0`.
**Long-Only Portfolio:** Only investing in assets, without borrowing or shorting. This is denoted by a non-negative allocation `w >= 0`.

**Leverage:**
```
||w||_1 = 1^T w_+ + 1^T w_
```
Where w_+ represents the long positions and w_- represents the short positions. This is known as leverage.

## Asset Returns

*Asset returns* describe the percentage change in the price of an asset over a period. Let's define:

*   `pi > 0`: initial price of asset `i`.
*   `pi' > 0`: end-of-period price of asset `i`.
*   `ri = (pi' - pi) / pi`: the asset (fractional) return.
*   `R = r^T w`: the portfolio (fractional) return.

A common model assumes `r` is a random variable with mean `E[r] = μ` and covariance `E[(r - μ)(r - μ)^T] = Σ`.

**Mean Return (μ):** The average expected return of an asset or portfolio.
**Covariance Matrix (Σ):** A matrix describing the relationships between the returns of different assets.

*   `E[R] = μ^T w`: Expected Portfolio Return.
*   `var(R) = w^T Σ w`: Variance of Portfolio Return.
*   `std(R) = sqrt(var(R))`: Standard Deviation of Portfolio Return.

## Classical (Markowitz) Portfolio Optimization

Classical (Markowitz) portfolio optimization solves the following problem:

```
maximize  μ^T w - γ w^T Σ w
subject to 1^T w = 1,  w ∈ W
```

**Objective Function:** Maximizes the risk-adjusted return, balancing expected return (μ^T w) and risk (w^T Σ w).
**Risk Aversion (γ):** A parameter that controls the trade-off between risk and return. Higher values of γ indicate greater risk aversion.
**Constraint:** The sum of the allocation fractions must equal 1.
**Feasible Set (W):** The set of allowed portfolios (e.g., long-only portfolios).
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

fig1, ax1 = plt.subplots()
ax1.plot(risk_data, ret_data, "g-", label="Trade-off Curve")

# Annotate two points on the trade-off curve
markers_on = [29, 40]  # Example marker indices
for marker in markers_on:
    ax1.plot(risk_data[marker], ret_data[marker], "bs")
    ax1.annotate(
        r"$\gamma = %.2f$" % gamma_vals[marker],
        xy=(risk_data[marker] + 0.08, ret_data[marker] - 0.03),
    )

# Plot individual assets (randomly placed)
for i in range(num_assets):
    ax1.plot(np.sqrt(Sigma[i, i]), mu[i], "ro", alpha=0.5)  # Added alpha for clarity

ax1.set_xlabel("Standard Deviation")
ax1.set_ylabel("Return")
ax1.set_title("Risk-Return Trade-off Curve")
ax1.legend()
st.pyplot(fig1)

st.markdown("""
This chart illustrates the trade-off between risk and return.
Each point on the curve represents an optimal portfolio for a given level of risk aversion (γ).
""")

# --- Visualization 2: Return Distribution ---
st.subheader("Return Distribution")
fig2, ax2 = plt.subplots()
for midx, idx in enumerate(markers_on):
    x = np.linspace(-2, 5, 1000)
    ax2.plot(
        x,
        spstats.norm.pdf(x, ret_data[idx], risk_data[idx]),
        label=r"$\gamma = %.2f$" % gamma_vals[idx],
    )
ax2.set_xlabel("Return")
ax2.set_ylabel("Density")
ax2.set_title("Return Distribution for Different Risk Aversion Values")
ax2.legend(loc="upper right")
st.pyplot(fig2)

st.markdown("""
This plot shows the return distributions for two different risk aversion values.
The shape of the distribution represents the probability of different return outcomes.
""")

# --- Visualization 3: Portfolio Optimization with Leverage Limit ---
st.subheader("Portfolio Optimization with Leverage Limit")

leverage_limits = [1, 2, 4]
num_samples = 100
gamma_vals = np.logspace(-2, 3, num=num_samples)
risk_data_leverage = np.zeros((len(leverage_limits), num_samples))
ret_data_leverage = np.zeros((len(leverage_limits), num_samples))

# Create a Streamlit slider to select the leverage limit
selected_leverage_limit = st.slider("Select Leverage Limit", min_value=1, max_value=4, value=1, step=1)

fig3, ax3 = plt.subplots()
for idx, leverage_limit in enumerate(leverage_limits):
    # Calculate risk and return data for each leverage limit
    for i in range(num_samples):
        # Define variables
        w_leverage = cp.Variable(num_assets)
        gamma = cp.Parameter(nonneg=True)
        Lmax = cp.Parameter(nonneg=True)

        # Define portfolio return
        ret = mu.T @ w_leverage

        # Define portfolio risk
        risk = cp.quad_form(w_leverage, Sigma)

        # Define constraints
        constraints = [cp.sum(w_leverage) == 1, cp.norm(w_leverage, 1) <= Lmax]

        # Define objective function
        objective = cp.Maximize(ret - gamma * risk)

        # Define problem
        prob = cp.Problem(objective, constraints)

        # Assign parameter values
        gamma.value = gamma_vals[i]
        Lmax.value = leverage_limit

        # Solve problem
        prob.solve()

        # Store values
        risk_data_leverage[idx, i] = cp.sqrt(risk).value
        ret_data_leverage[idx, i] = ret.value

    # Plot data with label
    ax3.plot(risk_data_leverage[idx, :], ret_data_leverage[idx, :], label=f'Leverage Limit = {leverage_limit}')

# Customize plot
ax3.set_xlabel('Risk')
ax3.set_ylabel('Return')
ax3.set_title('Risk-Return Trade-off with Different Leverage Limits')
ax3.legend()
ax3.grid(True)

# Show plot in Streamlit
st.pyplot(fig3)

# --- Visualization 4: Asset Allocation Bar Graph ---
st.subheader("Asset Allocation Bar Graph")

# Compute solution for different leverage limits.
L_vals = [1, 2, 4]

colors = ["b", "g", "r"]
indices = np.argsort(mu.flatten())

fig4, ax4 = plt.subplots()
for idx, L_val in enumerate(L_vals):
    # Define variables
    w_leverage = cp.Variable(num_assets)
    gamma = cp.Parameter(nonneg=True)
    Lmax = cp.Parameter(nonneg=True)

    # Define portfolio return
    ret = mu.T @ w_leverage

    # Define portfolio risk
    risk = cp.quad_form(w_leverage, Sigma)

    # Define constraints
    constraints = [cp.sum(w_leverage) == 1, cp.norm(w_leverage, 1) <= Lmax]

    # Define objective function
    objective = cp.Maximize(ret - gamma * risk)

    # Define problem
    prob = cp.Problem(objective, constraints)

    # Assign parameter values
    gamma.value = risk_aversion
    Lmax.value = L_val

    # Solve problem
    prob.solve()
    w_vals_leverage = w_leverage.value

    ax4.bar(np.arange(1, num_assets + 1) + 0.25 * idx - 0.375,
            w_vals_leverage[indices],
            color=colors[idx],
            label=f'Leverage Limit = {L_val}',
            width=0.25)

ax4.set_xlabel(r"$i$", fontsize=16)
ax4.set_ylabel(r"$w_i$", fontsize=16)
ax4.set_title("Asset Allocation for Different Leverage Limits")
ax4.set_xlim([1 - 0.375, num_assets + 0.375])
ax4.set_xticks(np.arange(1, num_assets + 1))
ax4.legend()
st.pyplot(fig4)

st.markdown("""
This bar graph shows the amount of each asset held in the portfolio for different leverage limits.
Negative holdings indicate a short position.
""")

# --- Factor Covariance Model Explanation ---
st.subheader("Factor Covariance Model")

st.markdown("""
A particularly common and useful variation is to model the covariance matrix (Σ) as a factor model:

```
Σ = F Σ~ F^T + D
```

Where:
*   `F`: the factor loading matrix.
*   `Σ~`: the factor covariance matrix.
*   `D`: a diagonal matrix representing idiosyncratic risk.

**Factor Exposures:**
```
F^T w
```
A portfolio is *factor j neutral* if `(F^T w)j = 0`.
""")
```

**Note:**  This specification provides a detailed blueprint for the Streamlit application. The code within the Appendix is intended as a starting point and needs to be integrated with the Streamlit framework.