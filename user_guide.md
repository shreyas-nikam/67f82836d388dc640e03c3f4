id: 67f82836d388dc640e03c3f4_user_guide
summary: Portfolio Optimization Lab User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Portfolio Optimization with QuLab

## Introduction to Portfolio Optimization
Duration: 0:05:00

<aside class="positive">
Welcome to the Portfolio Optimization Lab! This lab is designed to help you understand the fundamental concepts of building investment portfolios using the principles of modern portfolio theory, pioneered by Harry Markowitz.
</aside>

In the world of finance, deciding how to allocate your investments across different assets is a crucial task. You want to maximize the potential returns while minimizing the risk. This is where **Portfolio Optimization** comes in.

This application demonstrates **Markowitz's Mean-Variance Optimization** model. The core idea is that investors are **risk-averse**, meaning they prefer higher returns for the same level of risk, or lower risk for the same level of return.

The model helps you find an **optimal portfolio** – a combination of assets that offers the best possible expected return for a given level of risk, or the lowest possible risk for a given expected return.

Key concepts you will explore:

*   **Assets:** Individual investments (like stocks, bonds, etc.) that make up the portfolio.
*   **Portfolio Allocation (Weights):** The proportion of the total investment value assigned to each asset. The sum of these weights must be 1 (or 100%).
*   **Expected Return:** The average return you anticipate earning from an asset or portfolio over a period. For a portfolio with weights $w_i$ for asset $i$ and expected individual asset returns $\mu_i$, the portfolio expected return is $E[R_p] = \sum_i w_i \mu_i$.
*   **Risk:** Measured by the **standard deviation** ($\sigma$) or **variance** ($\sigma^2$) of returns. It quantifies how much the actual return is likely to deviate from the expected return. A key aspect is the **covariance** between asset returns, which measures how asset returns move together. The portfolio variance depends on the weights, individual variances, and all pairwise covariances: $\sigma_p^2 = \mathbf{w}^T \Sigma \mathbf{w}$, where $\mathbf{w}$ is the vector of weights and $\Sigma$ is the covariance matrix.
*   **Risk Aversion:** An investor's preference for less risk. The Markowitz model often incorporates risk aversion by trying to maximize a utility function that balances return and risk, such as $U(\mathbf{w}) = E[R_p] - \gamma \sigma_p^2$, where $\gamma$ is the **risk aversion parameter**. A higher $\gamma$ means the investor is more sensitive to risk and will prefer portfolios with lower risk, even if it means accepting a lower expected return.

This application uses **synthetic data** to simulate asset returns and their relationships, allowing you to experiment with these concepts without needing real market data.

## Accessing the Lab
Duration: 0:01:00

The application is structured with a sidebar for navigation and the main area displaying the interactive components and results.

Once the application loads, you will see the main title "QuLab" and a brief description.

In the sidebar, under "Navigation", you will find the available lab pages. Currently, only "Portfolio Optimization" is available. Select this option if it's not already selected.

The main content area will then display the "Portfolio Optimization Lab" page, which is where all the interactive elements and results for this lab are located.

<aside class="positive">
The sidebar is a common feature in Streamlit applications for navigation, especially as more functionalities are added.
</aside>

## Configuring Your Portfolio Parameters
Duration: 0:03:00

This lab allows you to interactively change two key parameters that influence the optimal portfolio calculation: the number of assets and your level of risk aversion.

Locate the sliders in the application's main area:

1.  **Number of Assets:**
    *   This slider lets you choose how many different assets ($n$) are included in the portfolio analysis.
    *   You can select a number between 5 and 50.
    *   The application generates synthetic data for the expected returns and the covariance matrix for this number of assets.

2.  **Risk Aversion Parameter (γ):**
    *   This slider allows you to set your level of risk aversion ($\gamma$).
    *   You can select a value between 0.01 (very low risk aversion, potentially seeking high returns even with high risk) and 10.0 (very high risk aversion, prioritizing risk reduction even if it means lower returns).
    *   This parameter is crucial because it dictates the trade-off the optimization algorithm makes between expected portfolio return and portfolio risk. A higher $\gamma$ pushes the optimization towards lower-risk portfolios.

Experiment by moving these sliders and observe how the outputs change.

## Analyzing the Optimal Portfolio Results
Duration: 0:04:00

After you set the "Number of Assets" and "Risk Aversion Parameter (γ)", the application automatically performs the Markowitz optimization and displays the results for the calculated optimal portfolio under those conditions.

You will see the following outputs:

1.  **Expected Return:**
    *   This is the calculated expected return for the optimal portfolio based on the synthetic data and the weights determined by the optimization.
    *   It represents the average return you could expect from this portfolio over the long run.

2.  **Risk (Std Dev):**
    *   This is the calculated standard deviation of the returns for the optimal portfolio.
    *   It is a measure of the portfolio's volatility or risk. A higher standard deviation indicates higher risk.

3.  **Asset Weights:**
    *   This section lists the calculated optimal weight for each individual asset (Asset 1, Asset 2, etc.).
    *   These weights tell you what proportion of your total investment should be allocated to each asset.
    *   Notice that the weights are displayed as decimal values. These are proportions, and they sum up to 1 (or close to 1 due to rounding).
    *   This specific optimization also includes a **long-only constraint**, meaning that the weights for each asset must be non-negative ($w_i \ge 0$). This is common in practice as it prevents short-selling.

<aside class="positive">
Try changing the Risk Aversion Parameter (γ). How do the Expected Return, Risk, and Asset Weights change? Do you see a pattern related to higher or lower gamma?
</aside>

## Exploring the Risk-Return Trade-off
Duration: 0:05:00

Below the optimal portfolio details, you will find a plot titled "Risk vs Return Trade-off Curve". This is a visual representation of a key concept in Markowitz theory: the **Efficient Frontier**.

The plot shows a curve where:
*   The **x-axis** represents the **Risk (Std Dev)** of a portfolio.
*   The **y-axis** represents the **Expected Return** of a portfolio.

Each point on the curve represents a hypothetical portfolio that is "efficient" – it offers the highest possible expected return for that specific level of risk, or the lowest possible risk for that specific expected return.

<aside class="positive">
The upper-left part of the curve represents portfolios with lower risk and lower expected returns, while the lower-right part represents portfolios with higher risk and potentially higher expected returns.
</aside>

The Markowitz optimization, guided by your "Risk Aversion Parameter (γ)", finds the single point on this efficient frontier that best matches your risk preference.

*   If you set a **low γ** (low risk aversion), the optimal portfolio found will be a point further towards the **lower-right** on the curve, indicating higher risk and higher expected return.
*   If you set a **high γ** (high risk aversion), the optimal portfolio found will be a point further towards the **upper-left** on the curve, indicating lower risk and lower expected return.

This plot helps visualize the fundamental trade-off investors face: to potentially achieve higher returns, you typically must accept higher risk. The efficient frontier shows the best possible combinations of risk and return achievable with the given set of assets and their statistical properties (expected returns and covariance).

By interacting with the "Risk Aversion Parameter (γ)" slider and observing the changes in the displayed Optimal Portfolio's Risk and Return, you can see how the chosen point on this theoretical curve shifts based on your preference.

