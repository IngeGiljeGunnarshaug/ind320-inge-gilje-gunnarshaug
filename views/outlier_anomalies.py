
import matplotlib.pyplot as plt
import streamlit as st

from weather_utils import (
    load_weather,
    get_weather_summary,
    REGION,
    START_DATE,
    END_DATE
)

from analysis_utils import (
    plot_precipitation_outliers,
    plot_temperature_anomalies
)


st.title("Outliers & Anomalies")

st.write(
    "Detect unusual precipitation and temperature observations "
    "using DCT-based Statistical Process Control (SPC) "
    "and Local Outlier Factor (LOF)."
)

# Load the fixed Open-Meteo dataset.
try:
    weather = load_weather()
    hourly_weather = get_weather_summary(weather)

    # Match the column names expected by the notebook functions.
    hourly_weather = hourly_weather.rename(
        columns={"temperature_2m": "temperature"}
    )

    # Ensure the required variables contain valid observations.
    if hourly_weather[["temperature", "precipitation"]].isna().any().any():
        raise ValueError(
            "Temperature or precipitation contains missing values."
        )

except Exception as error:
    st.error(f"Unable to load Open-Meteo data: {error}")
    st.stop()


st.caption(
    f"Open-Meteo ERA5 | {REGION} | "
    f"{START_DATE} to {END_DATE} | "
    "Hourly spatial averages"
)


# Create the two analysis tabs.
tab_spc, tab_lof = st.tabs([
    "Outliers (SPC)",
    "Anomalies (LOF)"
])


# TAB 1: PRECIPITATION OUTLIERS
with tab_spc:

    st.subheader("Precipitation Outliers — DCT & SPC")

    st.write(
        "A Discrete Cosine Transform removes low-frequency "
        "variation from the precipitation series. "
        "Robust control limits are then calculated using "
        "the median and Median Absolute Deviation (MAD)."
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

    fig, outlier_df = plot_precipitation_outliers(
        hourly_weather,
        cutoff=cutoff,
        n_std=n_std
    )

    st.pyplot(fig)
    plt.close(fig)

    # Summary statistics.
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
        f"{len(outlier_df) / total_observations * 100:.2f}%"
    )

    # Optional table containing only detected outliers.
    with st.expander("View detected outliers"):
        st.dataframe(
            outlier_df,
            hide_index=True,
            use_container_width=True
        )


# TAB 2: TEMPERATURE ANOMALIES
with tab_lof:

    st.subheader("Temperature Anomalies — LOF")

    st.write(
        "Local Outlier Factor identifies temperature "
        "observations with unusually low local density "
        "compared with their neighboring observations."
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

    fig, anomaly_df = plot_temperature_anomalies(
        hourly_weather,
        contamination=contamination,
        n_neighbors=n_neighbors
    )

    st.pyplot(fig)
    plt.close(fig)

    # Summary statistics.
    total_observations = len(hourly_weather)

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
        f"{len(anomaly_df) / total_observations * 100:.2f}%"
    )

    # Optional table containing only detected anomalies.
    with st.expander("View detected anomalies"):
        st.dataframe(
            anomaly_df,
            hide_index=True,
            use_container_width=True
        )
