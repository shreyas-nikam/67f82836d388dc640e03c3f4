import streamlit as st

def app():
    st.header("Factor Covariance Model")
    st.markdown("""
    ## Factor Covariance Model Explanation

    In this section, we explain the factor covariance model which is used to model the covariance matrix as:

    ```
    Σ = F Σ̃ F^T + D
    ```

    Where:
    - **F** is the factor loading matrix.
    - **Σ̃** is the factor covariance matrix.
    - **D** is a diagonal matrix capturing idiosyncratic risk.

    **Factor Exposures:**  
    The exposures of the portfolio to these factors can be computed as:  
    ```
    f = F^T w
    ```
    A portfolio is considered *factor neutral* with respect to factor j if:
    ```
    (F^T w)_j = 0
    ```

    ### Advantages of the Factor Model
    - Reduction in the number of parameters to estimate.
    - Possible improved stability of the covariance estimates.
    - Helps capture systematic risk factors driving asset returns.

    ### Example
    In practical applications, you might generate synthetic factor data, solve the corresponding portfolio optimization problem
    under the factor model, and compare the results with the classical model.

    This demonstration focuses on explaining the concept. For full interactivity, visit the Portfolio Optimization page.
    """)
    
if __name__ == '__main__':
    app()
