import streamlit as st
from pages import portfolio_optimization, factor_model

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

# Sidebar menu for navigation among pages
page = st.sidebar.selectbox("Select Page", ["Overview", "Portfolio Optimization", "Factor Covariance Model"])

if page == "Overview":
    st.markdown("""
    # Overview
    Welcome to the Portfolio Optimization Lab. This application demonstrates key concepts in portfolio allocation,
    risk-return trade-offs, and classical Markowitz optimization techniques. Use the sidebar to navigate to different
    sections of the lab.
    """)
elif page == "Portfolio Optimization":
    portfolio_optimization.app()
elif page == "Factor Covariance Model":
    factor_model.app()

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "Any reproduction of this demonstration requires prior written consent from QuantUniversity.")
