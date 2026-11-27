import streamlit as st
from state_utils import initialize_session_state

# Application configuration
st.set_page_config(
    page_title="IND320 - Energy Data",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize shared geographical and temporal selections.
initialize_session_state()

# Define individual pages
home = st.Page(
    "views/home.py",
    title="Home & Map",
    icon=":material/map:",
    default=True
)

reservoir = st.Page(
    "views/reservoir.py",
    title="Reservoir Data",
    icon=":material/database:"
)

electricity = st.Page(
    "views/electricity.py",
    title="Electricity",
    icon=":material/bolt:"
)

stl = st.Page(
    "views/stl_spectrogram.py",
    title="STL & Spectrogram",
    icon=":material/show_chart:"
)

anomalies = st.Page(
    "views/outlier_anomalies.py",
    title="Outliers & Anomalies",
    icon=":material/warning:"
)

correlation = st.Page(
    "views/correlation.py",
    title="Weather & Reservoir Correlation",
    icon=":material/scatter_plot:"
)

forecasting = st.Page(
    "views/forecasting.py",
    title="Reservoir Fill Fraction Forecasting",
    icon=":material/trending_up:"
)

# Group pages by analytical task
pages = {
    "": [home],

    "Data Exploration": [
        reservoir,
        electricity
    ],

    "Statistical Analysis": [
        stl,
        anomalies,
        correlation
    ],

    "Predictive Analysis": [
        forecasting
    ]
}

# Display navigation and run the selected page
navigation = st.navigation(
    pages,
    position="sidebar"
)

navigation.run()
