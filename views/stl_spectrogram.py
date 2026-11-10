
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_utils import load_data
from analysis_utils import (
    prepare_weekly_series,
    plot_stl,
    plot_spectrogram
)


st.title("STL & Spectrogram")

st.write(
    "Analyze long-term seasonal patterns and frequency "
    "components in Norwegian reservoir filling."
)

# Load historical NVE data for Norway as a whole.
try:
    df = load_data(
        pd.Timestamp("1995-01-01").date(),
        pd.Timestamp.today().date(),
        "NO"
    )

    series = prepare_weekly_series(df)

except Exception as error:
    st.error(f"Unable to load NVE data: {error}")
    st.stop()

st.caption(
    f"Source: NVE | Norway (area 0) | "
    f"{series.index.min():%Y-%m-%d} to "
    f"{series.index.max():%Y-%m-%d}"
)

tab_stl, tab_spectrogram = st.tabs([
    "STL Analysis",
    "Spectrogram"
])


with tab_stl:
    st.subheader("STL Decomposition")

    seasonal = st.slider(
        "Seasonal window",
        min_value=7,
        max_value=51,
        value=13,
        step=2
    )

    trend = st.slider(
        "Trend window",
        min_value=55,
        max_value=209,
        value=105,
        step=2
    )

    robust = st.checkbox(
        "Robust decomposition",
        value=True
    )

    fig, result = plot_stl(
        series,
        period=52,
        seasonal=seasonal,
        trend=trend,
        robust=robust
    )

    st.pyplot(fig)
    plt.close(fig)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Trend range",
        f"{result.trend.max() - result.trend.min():.3f}"
    )

    col2.metric(
        "Seasonal std.",
        f"{result.seasonal.std():.3f}"
    )

    col3.metric(
        "Residual std.",
        f"{result.resid.std():.3f}"
    )


with tab_spectrogram:
    st.subheader("Reservoir Spectrogram")

    window_length = st.slider(
        "Window length (weeks)",
        min_value=52,
        max_value=260,
        value=156,
        step=26
    )

    overlap = st.slider(
        "Window overlap (weeks)",
        min_value=0,
        max_value=window_length - 1,
        value=min(130, window_length - 1)
    )

    fig = plot_spectrogram(
        series,
        window_length=window_length,
        overlap=overlap
    )

    st.pyplot(fig)
    plt.close(fig)

    st.info(
        "A strong signal near 1 cycle/year indicates "
        "annual seasonality in reservoir filling."
    )
