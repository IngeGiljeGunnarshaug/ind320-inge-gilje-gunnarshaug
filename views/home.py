import streamlit as st

st.title("Norwegian Energy & Reservoir Dashboard")

st.markdown(
    """
    Explore Norwegian reservoir statistics, electricity transfers
    and meteorological data.

    Use the application to investigate historical patterns,
    identify anomalies, analyse relationships between weather
    and reservoir filling, and forecast future filling levels.
    """
)

st.divider()

# Placeholder for the interactive map
st.subheader("Geographical Selection")

st.info(
    "An interactive map with geographical area and time "
    "selection will be implemented here."
)

st.divider()

# Navigation overview
st.subheader("Explore the application")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Data Exploration")
    st.write(
        "Explore historical reservoir statistics "
        "and electricity transfers."
    )

    st.page_link(
        "views/reservoir.py",
        label="Reservoir Data",
        icon=":material/database:"
    )

    st.page_link(
        "views/electricity.py",
        label="Electricity",
        icon=":material/bolt:"
    )

with col2:
    st.markdown("### Statistical Analysis")
    st.write(
        "Investigate seasonality, anomalies and "
        "relationships between variables."
    )

    st.page_link(
        "views/stl_spectrogram.py",
        label="STL & Spectrogram",
        icon=":material/show_chart:"
    )

    st.page_link(
        "views/outlier_anomalies.py",
        label="Outliers & Anomalies",
        icon=":material/warning:"
    )

    st.page_link(
        "views/correlation.py",
        label="Weather & Reservoir Correlation",
        icon=":material/scatter_plot:"
    )

with col3:
    st.markdown("### Predictive Analysis")
    st.write(
        "Forecast reservoir filling using statistical "
        "and machine learning models."
    )

    st.page_link(
        "views/forecasting.py",
        label="Reservoir Fill Fraction Forecasting",
        icon=":material/trending_up:"
    )

st.divider()

st.caption("Developed as part of the IND320 course at NMBU.")
