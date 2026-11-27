import streamlit as st
import pandas as pd

from state_utils import show_current_selection

from forecast_utils import (
    DEFAULT_FORECAST_PARAMS,
    prepare_forecast_data,
    forecast_reservoir,
    plot_reservoir_forecast,
    load_actual_observations
)


# --------------------------------------------------
# PAGE HEADER
# --------------------------------------------------

st.title("Reservoir Filling Forecast")

show_current_selection()

st.write(
    "Forecast future reservoir filling using River's "
    "SNARIMAX model with optional external variables."
)


# --------------------------------------------------
# SHARED SESSION STATE
# --------------------------------------------------

area_type = st.session_state.area_type
area_number = st.session_state.selected_area

if area_number is None:

    st.info(
        "Select a geographical area on the home page."
    )

    st.stop()


# Use the selected map date as the forecast cutoff.
forecast_date = pd.Timestamp(
    st.session_state.map_date
)

global_start = pd.Timestamp(
    st.session_state.start_date
)

defaults = DEFAULT_FORECAST_PARAMS


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

with st.form("forecast_configuration"):

    st.subheader("Forecast configuration")

    col1, col2 = st.columns(2)

    with col1:

        history = st.number_input(
            "Training history (weeks)",
            min_value=104,
            max_value=520,
            value=defaults["history"],
            step=13
        )

    with col2:

        horizon = st.slider(
            "Forecast horizon (weeks)",
            min_value=1,
            max_value=52,
            value=int(
                st.session_state.forecast_horizon
            )
        )

    # Electricity data is available for electricity
    # price areas and Norway as a whole.
    available_sources = [
        "weather",
        "currency"
    ]

    if area_type in ["EL", "NO"]:

        available_sources.append("electricity")

    default_sources = [
        source
        for source in defaults["sources"]
        if source in available_sources
    ]

    sources = st.multiselect(
        "External data sources",
        options=available_sources,
        default=default_sources,
        format_func=lambda x: x.capitalize(),
        key=f"forecast_sources_{area_type}"
    )

    if area_type == "VASS":

        st.caption(
            "Electricity transfers are unavailable for "
            "VASS regions because their geographical "
            "boundaries differ from electricity price areas."
        )

    st.caption(
        "Reservoir observations are always included."
    )

    st.subheader("Model parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        p = st.number_input(
            "Autoregressive order (p)",
            min_value=0,
            max_value=10,
            value=defaults["p"]
        )

    with col2:

        d = st.number_input(
            "Differencing order (d)",
            min_value=0,
            max_value=2,
            value=defaults["d"]
        )

    with col3:

        q = st.number_input(
            "Moving average order (q)",
            min_value=0,
            max_value=10,
            value=defaults["q"]
        )

    st.subheader("Seasonal parameters")

    col1, col2, col3 = st.columns(3)

    with col1:

        sp = st.number_input(
            "Seasonal AR order",
            min_value=0,
            max_value=2,
            value=defaults["sp"]
        )

    with col2:

        sd = st.number_input(
            "Seasonal differencing",
            min_value=0,
            max_value=1,
            value=defaults["sd"]
        )

    with col3:

        sq = st.number_input(
            "Seasonal MA order",
            min_value=0,
            max_value=2,
            value=defaults["sq"]
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        m = st.number_input(
            "Seasonal period (weeks)",
            min_value=2,
            max_value=104,
            value=defaults["m"]
        )

    with col2:

        learning_rate = st.selectbox(
            "Learning rate",
            options=[0.001, 0.01, 0.05],
            index=[0.001, 0.01, 0.05].index(
                defaults["learning_rate"]
            )
        )

    with col3:

        confidence = st.selectbox(
            "Prediction interval",
            options=[0.80, 0.90, 0.95, 0.99],
            index=2,
            format_func=lambda x: f"{x:.0%}"
        )

    submitted = st.form_submit_button(
        "Generate forecast",
        type="primary"
    )


# --------------------------------------------------
# MODEL TRAINING
# --------------------------------------------------

selection = (
    area_type,
    area_number,
    global_start,
    forecast_date
)

if submitted:

    # Remove results from previous model runs.
    st.session_state.pop(
        "forecast_result",
        None
    )

    st.session_state.forecast_horizon = horizon

    try:

        # Ensure sufficient seasonal warmup.
        warmup = (
            d + m * sd
            + max(p, q, m * sp, m * sq)
        )

        min_train = max(
            104,
            warmup + 1
        )

        if history < min_train:

            raise ValueError(
                f"Training history must be at least "
                f"{min_train} weeks for these parameters."
            )

        # Download forecasting data.
        with st.spinner(
            "Downloading forecasting data..."
        ):

            df = prepare_forecast_data(
                area_type=area_type,
                area_number=area_number,
                start_date=global_start,
                end_date=forecast_date,
                sources=tuple(sources)
            )

        # Train and evaluate forecasting model.
        with st.spinner(
            "Training and evaluating SNARIMAX..."
        ):

            forecast, metrics = forecast_reservoir(
                df,
                sources=tuple(sources),
                history=history,
                horizon=horizon,
                p=p,
                d=d,
                q=q,
                sp=sp,
                sd=sd,
                sq=sq,
                m=m,
                confidence=confidence,
                learning_rate=learning_rate,
                min_train=min_train
            )

        # Persist results across Streamlit reruns.
        st.session_state.forecast_result = {
            "selection": selection,
            "historical": df,
            "forecast": forecast,
            "metrics": metrics,
            "confidence": confidence
        }

    except Exception as error:

        st.error(
            f"Forecast failed: {error}"
        )


# --------------------------------------------------
# DISPLAY FORECAST
# --------------------------------------------------

result = st.session_state.get(
    "forecast_result"
)

if result is None:

    st.info(
        "Configure the model and select Generate forecast."
    )

elif result["selection"] != selection:

    st.info(
        "The geographical area or historical period has changed. "
        "Generate a new forecast."
    )

else:

    df = result["historical"]

    forecast = result["forecast"]

    metrics = result["metrics"]

    # Retrieve actual observations independently of training.
    try:

        actual = load_actual_observations(
            area_type=area_type,
            area_number=area_number,
            start_date=forecast.index.min(),
            end_date=forecast.index.max()
        )

    except Exception as error:

        st.warning(
            f"Unable to retrieve actual observations: {error}"
        )

        actual = pd.Series(dtype=float, name="actual")

    # Align actual observations with prediction dates.
    comparison = forecast.copy()

    comparison["actual"] = actual.reindex(
        forecast.index
    )

    comparison["error_pp"] = (
        comparison["prediction"]
        - comparison["actual"]
    ) * 100

    st.subheader("Forecast results")

    st.caption(
        f"Forecast origin: "
        f"{metrics['last_observation']:%d.%m.%Y} | "
        f"Training observations: "
        f"{metrics['training_observations']} weeks | "
        f"Forecast horizon: {len(forecast)} weeks"
    )

    # --------------------------------------------------
    # FORECAST METRICS
    # --------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Backtest MAE",
        f"{metrics['MAE'] * 100:.2f} pp"
    )

    col2.metric(
        "Backtest RMSE",
        f"{metrics['RMSE'] * 100:.2f} pp"
    )

    col3.metric(
        "Final forecast",
        f"{forecast['prediction'].iloc[-1]:.1%}"
    )

    col4.metric(
        "Mean interval width",
        f"{metrics['mean_interval_width'] * 100:.1f} pp"
    )

    if "baseline_MAE" in metrics:

        baseline_mae = metrics["baseline_MAE"]

        if baseline_mae > 0:

            skill = (
                1 - metrics["MAE"] / baseline_mae
            )

            st.caption(
                f"Seasonal naive MAE: "
                f"{baseline_mae * 100:.2f} pp | "
                f"Relative improvement: {skill:.1%}"
            )

    # --------------------------------------------------
    # INTERACTIVE FORECAST PLOT
    # --------------------------------------------------

    display_weeks = st.slider(
        "Historical observations displayed (weeks)",
        min_value=26,
        max_value=260,
        value=104,
        step=13
    )

    fig = plot_reservoir_forecast(
        df,
        forecast,
        confidence=result["confidence"],
        display_weeks=display_weeks,
        actual=comparison["actual"]
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit",
        key="reservoir_forecast_chart"
    )

    st.caption(
        "Press Play below the plot to reveal the predicted "
        "reservoir filling one week at a time."
    )

    # Evaluate the forecast against available actual observations.
    st.subheader("Actual forecast performance")

    evaluated = comparison.dropna(
        subset=["actual", "prediction"]
    )

    if evaluated.empty:

        st.info(
            "No actual observations are available "
            "for the forecast period yet."
        )

    else:

        mae = evaluated["error_pp"].abs().mean()

        rmse = (
            evaluated["error_pp"].pow(2).mean()
        ) ** 0.5

        coverage = (
            (evaluated["actual"] >= evaluated["lower"])
            & (evaluated["actual"] <= evaluated["upper"])
        ).mean()

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Actual MAE",
            f"{mae:.2f} pp"
        )

        col2.metric(
            "Actual RMSE",
            f"{rmse:.2f} pp"
        )

        col3.metric(
            "Interval coverage",
            f"{coverage:.1%}"
        )

        col4.metric(
            "Evaluated weeks",
            f"{len(evaluated)}/{len(forecast)}"
        )

    # --------------------------------------------------
    # FORECAST DATA
    # --------------------------------------------------

    with st.expander("View forecast data"):

        st.dataframe(
            comparison.reset_index(),
            hide_index=True,
            use_container_width=True,
            column_config={
                "prediction": st.column_config.NumberColumn(
                    "Prediction",
                    format="percent"
                ),
                "lower": st.column_config.NumberColumn(
                    "Lower bound",
                    format="percent"
                ),
                "upper": st.column_config.NumberColumn(
                    "Upper bound",
                    format="percent"
                ),
                "actual": st.column_config.NumberColumn(
                    "Actual",
                    format="percent"
                ),
                "error_pp": st.column_config.NumberColumn(
                    "Error (pp)",
                    format="%.2f"
                )
            }
        )

    # --------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------

    with st.expander("Model evaluation"):

        st.write(metrics)

    # --------------------------------------------------
    # TRAINING DATA
    # --------------------------------------------------

    with st.expander("Training data"):

        st.dataframe(
            df.reset_index(),
            use_container_width=True,
            hide_index=True
        )