# Portfolio Optimization Lab - QuLab

This repository contains a Streamlit application for the Portfolio Optimization Lab as demonstrated by QuantUniversity.

## Overview

The application provides an interactive environment to explore portfolio optimization techniques based on the classical Markowitz model. The app is structured into multiple pages and includes interactive visualizations, portfolio constraints exploration, and explanations of the factor covariance model.

### Key Features
- **Interactive Dashboard:** Multi-page structure with separate sections for Portfolio Optimization and Factor Covariance Model.
- **Risk-Return Trade-off Visualization:** Explore how risk aversion impacts portfolio risk and return.
- **Return Distribution Analysis:** Display distribution of returns for different risk aversion levels.
- **Portfolio Constraints Exploration:** Adjust leverage limits and view optimal portfolio allocations.
- **Factor Covariance Model Explanation:** Overview and benefits of using a factor model to model asset risks.

## Project Structure

## How to Run the Application

### Locally
1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
2. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

### Using Docker
1. Build the Docker image:
   ```
   docker build -t qulab .
   ```
2. Run the Docker container:
   ```
   docker run -p 8501:8501 qulab
   ```

## Acknowledgments
© 2025 QuantUniversity. All Rights Reserved.

*The purpose of this demonstration is solely for educational use and illustration. Any reproduction of this work requires prior written consent from QuantUniversity.*
