import streamlit as st
from state_utils import show_current_selection

st.title("Reservoir Fill Fraction Forecasting")

show_current_selection()

st.write(
    "Forecast reservoir filling using SARIMAX and "
    "machine learning models with selectable exogenous variables."
)

st.info("Reservoir forecasting will be implemented here.")
