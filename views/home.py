import streamlit as st
from datetime import date
from data_utils import load_data
from state_utils import (
    update_area_type,
    update_selected_area,
    update_date_range,
    show_current_selection
)

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

# Geographical and temporal selection
st.subheader("Geographical Selection")

# Available geographical area types
AREA_TYPES = {
    "EL": "Price areas",
    "VO": "Water areas",
    "NO": "Norway"
}

# Restore persistent values to temporary widget keys
st.session_state["_area_type"] = st.session_state.area_type

st.session_state["_date_range"] = (
    st.session_state.start_date,
    st.session_state.end_date
)

col1, col2, col3 = st.columns(3)

with col1:
    st.selectbox(
        "Area type",
        options=list(AREA_TYPES),
        format_func=lambda x: AREA_TYPES[x],
        key="_area_type",
        on_change=update_area_type
    )

with col2:
    st.date_input(
        "Historical period",
        min_value=date(1995, 1, 1),
        max_value=date.today(),
        key="_date_range",
        on_change=update_date_range
    )

# Load reservoir data for the selected area type and period
df = load_data(
    st.session_state.start_date,
    st.session_state.end_date,
    st.session_state.area_type
)

# Identify available geographical areas
area_numbers = sorted(
    df["area_number"].dropna().astype(int).unique().tolist()
)

# Validate the current area selection
if st.session_state.selected_area not in area_numbers:
    st.session_state.selected_area = (
        area_numbers[0] if area_numbers else None
    )

st.session_state["_selected_area"] = (
    st.session_state.selected_area
)

with col3:
    st.selectbox(
        "Select area",
        options=area_numbers,
        format_func=lambda x: (
            f"NO{x}"
            if st.session_state.area_type == "EL"
            else f"Area {x}"
        ),
        key="_selected_area",
        on_change=update_selected_area
    )

show_current_selection()

if df.empty:
    st.warning("No reservoir data for the selected period.")

# Placeholder for the future map
st.info("Interactive GeoJSON map will be implemented here.")


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
