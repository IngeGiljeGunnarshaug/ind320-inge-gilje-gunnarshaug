import streamlit as st
import pandas as pd

from state_utils import show_current_selection

from data_utils import (
    load_weather,
    HOURLY_WEATHER_MEASUREMENTS
)

from map_utils import get_weather_locations

from analysis_utils import (
    plot_weather_outliers,
    plot_weather_anomalies
)


st.title("Outliers & Anomalies")

show_current_selection()

st.write(
    "Detect unusual weather observations using "
    "DCT-based Statistical Process Control (SPC) "
    "and Local Outlier Factor (LOF)."
)


# Retrieve the shared geographical selection.
area_type = st.session_state.area_type
area_number = st.session_state.selected_area

if area_number is None:
    st.info("Select a geographical area on the home page.")
    st.stop()


# Use a local date interval to limit hourly weather downloads.
global_start = pd.Timestamp(
    st.session_state.start_date
)

global_end = pd.Timestamp(
    st.session_state.end_date
)

default_start = max(
    global_start,
    global_end - pd.Timedelta(days=364)
)

dates = st.date_input(
    "Weather analysis period",
    value=(
        default_start.date(),
        global_end.date()
    ),
    min_value=global_start.date(),
    max_value=global_end.date(),
    key="outlier_analysis_dates"
)

if len(dates) != 2:
    st.info("Select a start and end date.")
    st.stop()

start_date, end_date = dates

if start_date > end_date:
    st.error("Start date must precede end date.")
    st.stop()


# Retrieve representative weather locations.
try:
    points = get_weather_locations(
        area_type,
        area_number,
        n_locations=5
    )

except Exception as error:
    st.error(f"Unable to select weather locations: {error}")
    st.stop()


# Create independent analysis tabs.
tab_spc, tab_lof = st.tabs([
    "Outliers (SPC)",
    "Anomalies (LOF)"
])


# --------------------------------------------------
# TAB 1: STATISTICAL PROCESS CONTROL
# --------------------------------------------------

with tab_spc:

    st.subheader("Weather Outliers — DCT & SPC")

    st.write(
        "A Discrete Cosine Transform removes low-frequency "
        "variation from the selected weather series. "
        "Robust control limits are calculated using "
        "the median and Median Absolute Deviation (MAD)."
    )

    spc_variable = st.selectbox(
        "Weather property",
        options=list(HOURLY_WEATHER_MEASUREMENTS),
        format_func=lambda x: HOURLY_WEATHER_MEASUREMENTS[x],
        key="spc_weather_variable"
    )

    cutoff = st.slider(
        "DCT cutoff (number of coefficients)",
        min_value=10,
        max_value=500,
        value=100,
        step=10
    )

    n_std = st.slider(
        "Control limit multiplier (k)",
        min_value=1.0,
        max_value=6.0,
        value=3.0,
        step=0.5
    )

    try:

        with st.spinner("Loading hourly weather..."):

            hourly_weather = load_weather(
                points,
                start_date,
                end_date,
                [spc_variable],
                resolution="hourly"
            )

        fig, outlier_df = plot_weather_outliers(
            hourly_weather,
            variable=spc_variable,
            variable_label=HOURLY_WEATHER_MEASUREMENTS[
                spc_variable
            ],
            cutoff=cutoff,
            n_std=n_std
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme="streamlit",
            key="weather_outliers_chart"
        )

        total_observations = len(hourly_weather)

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Observations",
            total_observations
        )

        col2.metric(
            "Outliers detected",
            len(outlier_df)
        )

        col3.metric(
            "Outlier percentage",
            f"{len(outlier_df) / total_observations:.2%}"
        )

        with st.expander("View detected outliers"):

            st.dataframe(
                outlier_df,
                hide_index=True,
                use_container_width=True
            )

    except Exception as error:

        st.error(f"Unable to perform SPC analysis: {error}")


# --------------------------------------------------
# TAB 2: LOCAL OUTLIER FACTOR
# --------------------------------------------------

with tab_lof:

    st.subheader("Weather Anomalies — LOF")

    st.write(
        "Local Outlier Factor identifies observations "
        "with unusually low local density in the "
        "selected weather variable."
    )

    lof_variable = st.selectbox(
        "Weather property",
        options=list(HOURLY_WEATHER_MEASUREMENTS),
        format_func=lambda x: HOURLY_WEATHER_MEASUREMENTS[x],
        key="lof_weather_variable"
    )

    contamination = st.slider(
        "Contamination",
        min_value=0.005,
        max_value=0.100,
        value=0.010,
        step=0.005,
        format="%.3f"
    )

    n_neighbors = st.slider(
        "Number of neighbors",
        min_value=5,
        max_value=100,
        value=20,
        step=5
    )

    try:

        with st.spinner("Loading hourly weather..."):

            hourly_weather = load_weather(
                points,
                start_date,
                end_date,
                [lof_variable],
                resolution="hourly"
            )

        fig, anomaly_df = plot_weather_anomalies(
            hourly_weather,
            variable=lof_variable,
            variable_label=HOURLY_WEATHER_MEASUREMENTS[
                lof_variable
            ],
            contamination=contamination,
            n_neighbors=n_neighbors
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
            theme="streamlit",
            key="weather_anomalies_chart"
        )

        total_observations = (
            hourly_weather[lof_variable].notna().sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Observations",
            total_observations
        )

        col2.metric(
            "Anomalies detected",
            len(anomaly_df)
        )

        col3.metric(
            "Anomaly percentage",
            f"{len(anomaly_df) / total_observations:.2%}"
        )

        with st.expander("View detected anomalies"):

            st.dataframe(
                anomaly_df,
                hide_index=True,
                use_container_width=True
            )

    except Exception as error:

        st.error(f"Unable to perform LOF analysis: {error}")