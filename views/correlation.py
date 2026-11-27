import streamlit as st
import pandas as pd

from state_utils import show_current_selection

from data_utils import (
    load_data,
    load_weather,
    aggregate_weekly_weather,
    MEASUREMENTS,
    WEATHER_MEASUREMENTS
)

from map_utils import get_weather_locations

from analysis_utils import (
    calculate_rolling_correlation,
    plot_rolling_correlation
)


st.title("Sliding-window Correlation")

show_current_selection()

st.write(
    "Investigate how the relationship between weather "
    "and reservoir measurements changes over time. "
    "A positive lag compares earlier weather observations "
    "with later reservoir observations."
)


# Retrieve shared geographical and temporal selections.
area_type = st.session_state.area_type
area_number = st.session_state.selected_area

start_date = pd.Timestamp(
    st.session_state.start_date
)

end_date = pd.Timestamp(
    st.session_state.end_date
)

if area_number is None:
    st.info("Select a geographical area on the home page.")
    st.stop()


# Measurement selectors.
col1, col2 = st.columns(2)

with col1:

    reservoir_variable = st.selectbox(
        "Reservoir measurement",
        options=list(MEASUREMENTS),
        format_func=lambda x: MEASUREMENTS[x],
        index=list(MEASUREMENTS).index(
            "fill_fraction_change"
        )
    )

with col2:

    weather_variable = st.selectbox(
        "Weather property",
        options=list(WEATHER_MEASUREMENTS),
        format_func=lambda x: WEATHER_MEASUREMENTS[x]
    )


# Correlation parameters.
col1, col2 = st.columns(2)

with col1:

    window = st.slider(
        "Sliding-window length (weeks)",
        min_value=4,
        max_value=104,
        value=26,
        step=1
    )

with col2:

    lag = st.slider(
        "Weather lag (weeks)",
        min_value=0,
        max_value=26,
        value=0,
        step=1,
        help=(
            "A positive lag compares earlier weather "
            "with later reservoir observations."
        )
    )


try:

    # Download reservoir observations.
    df = load_data(
        start_date,
        end_date,
        area_type
    )

    df = df.loc[
        df["area_number"] == area_number
    ].copy()

    if df.empty:
        st.warning(
            "No reservoir observations found."
        )
        st.stop()

    reservoir = (
        df.groupby("date")[reservoir_variable]
        .mean()
        .sort_index()
    )

    reservoir.index = pd.to_datetime(
        reservoir.index
    )

    # Establish a regular weekly observation grid.
    reservoir = reservoir.asfreq("7D")

    if len(reservoir) < window:
        st.warning(
            "The selected period is shorter than "
            "the sliding-window length."
        )
        st.stop()

    # Retrieve representative weather locations.
    points = get_weather_locations(
        area_type,
        area_number,
        n_locations=5
    )

    # Include earlier weather to support positive lags.
    # Download the maximum supported lag once so changing
    # the lag slider can reuse the cached weather data.
    max_lag = 26

    weather_dates = pd.date_range(
        start=(
            reservoir.index.min()
            - pd.Timedelta(weeks=max_lag)
        ),
        end=reservoir.index.max(),
        freq="7D"
    )

    weather_start = (
        weather_dates.min()
        - pd.Timedelta(days=6)
    )

    weather_end = weather_dates.max()

    # Download daily weather directly from Open-Meteo.
    with st.spinner("Loading weather observations..."):

        daily_weather = load_weather(
            points,
            weather_start,
            weather_end,
            [weather_variable],
            resolution="daily"
        )

    # Aggregate weather to the NVE observation weeks.
    weekly_weather = aggregate_weekly_weather(
        daily_weather,
        weather_dates,
        weather_variable
    )

    # Calculate sliding-window correlation.
    correlation_df = calculate_rolling_correlation(
        reservoir,
        weekly_weather,
        window=window,
        lag=lag
    )

    # Restrict results to the selected reservoir period.
    correlation_df = correlation_df.loc[
        reservoir.index.min():
        reservoir.index.max()
    ]

    # Create the interactive figure.
    fig = plot_rolling_correlation(
        correlation_df,
        reservoir_label=MEASUREMENTS[
            reservoir_variable
        ],
        weather_label=WEATHER_MEASUREMENTS[
            weather_variable
        ],
        window=window,
        lag=lag
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        key="rolling_correlation_chart"
    )

    # Summary statistics.
    correlations = (
        correlation_df["correlation"].dropna()
    )

    if correlations.empty:

        st.warning(
            "No valid correlation windows. "
            "Try a shorter window or another measurement."
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Mean correlation",
            f"{correlations.mean():.3f}"
        )

        col2.metric(
            "Minimum correlation",
            f"{correlations.min():.3f}"
        )

        col3.metric(
            "Maximum correlation",
            f"{correlations.max():.3f}"
        )

        col4.metric(
            "Latest correlation",
            f"{correlations.iloc[-1]:.3f}"
        )

        st.caption(
            f"{len(correlations)} valid correlation windows | "
            f"{len(points)} sampled weather locations"
        )

    # Optional table.
    with st.expander("View correlation data"):

        st.dataframe(
            correlation_df.reset_index().rename(
                columns={"index": "date"}
            ),
            hide_index=True,
            use_container_width=True,
            column_config={
                "correlation": st.column_config.NumberColumn(
                    "Correlation",
                    format="%.3f"
                )
            }
        )

except Exception as error:

    st.error(
        f"Unable to calculate correlation: {error}"
    )