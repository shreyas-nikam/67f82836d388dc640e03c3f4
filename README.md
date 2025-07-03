# QuLab: Interactive Portfolio Optimization Lab

This repository contains a Streamlit application developed as a laboratory project to explore fundamental concepts in classical portfolio optimization, specifically using the Markowitz model.

## Project Description

The "QuLab: Portfolio Optimization Lab" application provides an interactive environment for users to understand and experiment with portfolio allocation. It allows users to define parameters such as the number of assets and their risk aversion level, and then visualizes the resulting optimal portfolio and the risk-return trade-off curve (an approximation of the efficient frontier) based on synthetically generated asset data.

The lab focuses on demonstrating:
*   Defining portfolio allocation vectors (weights).
*   Understanding asset returns and risk (covariance).
*   Applying optimization techniques (specifically, solving the mean-variance optimization problem) to find an optimal portfolio.
*   Visualizing the relationship between risk and return for different levels of risk aversion.

This application is designed for educational purposes to provide a hands-on experience with portfolio optimization concepts.

## Features

*   **Interactive Parameter Control:** Adjust the number of assets and your risk aversion (`gamma`) using sliders.
*   **Synthetic Data Generation:** The application generates synthetic asset data (expected returns and covariance matrix via a factor model) for demonstration.
*   **Markowitz Optimization:** Solves the mean-variance optimization problem to find the optimal portfolio weights for the given risk aversion.
*   **Results Display:** Shows the expected return, risk (standard deviation), and individual asset weights for the optimal portfolio.
*   **Risk-Return Visualization:** Plots a curve illustrating the risk-return trade-off for various risk aversion levels, showcasing points along the efficient frontier.
*   **Modular Design:** Application logic separated into main app file and a dedicated page file.

## Getting Started

Follow these steps to get the application up and running on your local machine.

### Prerequisites

*   Python 3.7+
*   `pip` package manager

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url> # Replace <repository_url> with the actual URL
    cd <repository_name>       # Replace <repository_name> with the cloned directory name
    ```
2.  **Install required libraries:**
    It's recommended to use a virtual environment.
    ```bash
    # Create a virtual environment (optional but recommended)
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

    # Install dependencies (Assuming you have a requirements.txt or install them manually)
    pip install streamlit numpy cvxpy matplotlib
    ```
    *(Note: A `requirements.txt` file containing `streamlit`, `numpy`, `cvxpy`, and `matplotlib` is recommended for easier setup)*

## Usage

1.  **Run the Streamlit application:**
    Navigate to the project root directory in your terminal (where `app.py` is located) and run:
    ```bash
    streamlit run app.py
    ```
2.  **Interact with the App:**
    *   The application will open in your web browser.
    *   Use the sidebar navigation (currently only "Portfolio Optimization").
    *   Adjust the "Number of Assets" slider (between 5 and 50).
    *   Adjust the "Risk Aversion Parameter (γ)" slider (between 0.01 and 10.0).
    *   Observe the calculated optimal portfolio details (Expected Return, Risk, Asset Weights) and the updated Risk vs Return Trade-off curve in the main area.

## Project Structure

```
.
├── app.py
└── application_pages/
    └── page1.py
```

*   `app.py`: The main entry point for the Streamlit application. Sets up the basic page configuration, sidebar navigation, and routes to different application pages.
*   `application_pages/`: Directory containing individual page modules for the application.
*   `application_pages/page1.py`: Implements the "Portfolio Optimization" page logic, including synthetic data generation, optimization solving using CVXPY, result display, and plotting.

## Technology Stack

*   **Frontend & Application Framework:** Streamlit
*   **Numerical Computation:** NumPy
*   **Optimization Solver:** CVXPY
*   **Plotting:** Matplotlib
*   **Programming Language:** Python

## Contributing

This project is primarily developed as a laboratory exercise. Specific contribution guidelines are not defined for this lab project.

## License

This application is developed by QuantUniversity.

© 2025 QuantUniversity. All Rights Reserved.

The purpose of this demonstration is solely for educational use and illustration. Any reproduction of this demonstration requires prior written consent from QuantUniversity.

## Notes & Disclaimer

This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, which may contain inaccuracies or errors. The synthetic data used is for illustrative purposes and does not represent real-world market conditions. The optimization results are based on simplified assumptions of the Markowitz model and the synthetic data generated.

## Contact

For inquiries regarding QuantUniversity or QuCreate, please visit [https://www.quantuniversity.com](https://www.quantuniversity.com).

