import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, we explore the concepts of portfolio optimization using classical Markowitz theory. We will delve into defining portfolio allocation vectors, understanding asset returns and risk, and applying optimization techniques to find the optimal portfolio based on user-defined risk aversion.
""")

# Main navigation
page = st.sidebar.selectbox(label="Navigation", options=["Portfolio Optimization"])

if page == "Portfolio Optimization":
    from application_pages.page1 import run_page1
    run_page1()

st.divider()
st.write("© 2025 QuantUniversity. All Rights Reserved.")
st.caption("The purpose of this demonstration is solely for educational use and illustration. "
           "Any reproduction of this demonstration "
           "requires prior written consent from QuantUniversity. "
           "This lab was generated using the QuCreate platform. QuCreate relies on AI models for generating code, which may contain inaccuracies or errors")
