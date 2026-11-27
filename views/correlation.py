import streamlit as st
from state_utils import show_current_selection

st.title("Weather & Reservoir Correlation")

show_current_selection()

st.write(
    "Explore relationships between meteorological properties "
    "and reservoir filling using sliding window correlation."
)

st.info("Sliding window correlation will be implemented here.")