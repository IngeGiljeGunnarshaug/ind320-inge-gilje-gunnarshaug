import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from river import (
    linear_model,
    preprocessing,
    optim,
    time_series
)

from data_utils import (
    load_data,
    load_weather,
    aggregate_weekly_weather,
    download_exchange_rates,
    load_weekly_electricity
)

from map_utils import get_weather_locations


# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

DATA_SOURCES = {
    "weather": ["temperature_2m", "precipitation"],
    "electricity": ["export", "import"],
    "currency": ["EUR", "USD"]
}


DEFAULT_FORECAST_PARAMS = {
    "p": 3,
    "d": 0,
    "q": 2,
    "sp": 0,
    "sd": 1,
    "sq": 0,
    "m": 52,
    "history": 360,
    "horizon": 13,
    "learning_rate": 0.01,
    "sources": ("weather",),
    "confidence": 0.95
}


# --------------------------------------------------
# DATA PREPARATION
# --------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def prepare_forecast_data(
    area_type,
    area_number,
    start_date,
    end_date,
    sources=(),
    n_locations=5
):
    """Prepare weekly reservoir and external forecasting data."""

    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Download reservoir observations.
    reservoir = load_data(
        start_date,
        end_date,
        area_type
    )

    reservoir = reservoir.loc[
        reservoir["area_number"] == area_number
    ].copy()

    if reservoir.empty:

        raise ValueError(
            "No reservoir observations found."
        )

    # Prepare weekly reservoir observations.
    df = (
        reservoir.groupby("date")[["fill_fraction"]]
        .mean()
        .sort_index()
    )

    df.index = pd.to_datetime(df.index)

    if not (df.index.dayofweek == 6).all():

        raise ValueError(
            "Expected Sunday reservoir observation dates."
        )

    df = df.asfreq("W-SUN")

    # Include the complete first observation week.
    download_start = (
        df.index.min() - pd.Timedelta(days=6)
    )

    download_end = df.index.max()

    # --------------------------------------------------
    # WEATHER
    # --------------------------------------------------

    if "weather" in sources:

        points = get_weather_locations(
            area_type,
            area_number,
            n_locations=n_locations
        )

        variables = [
            "temperature_2m_mean",
            "precipitation_sum"
        ]

        daily_weather = load_weather(
            points,
            download_start,
            download_end,
            variables,
            resolution="daily"
        )

        df["temperature_2m"] = aggregate_weekly_weather(
            daily_weather,
            df.index,
            "temperature_2m_mean"
        )

        df["precipitation"] = aggregate_weekly_weather(
            daily_weather,
            df.index,
            "precipitation_sum"
        )

    # --------------------------------------------------
    # CURRENCY
    # --------------------------------------------------

    if "currency" in sources:

        currency = download_exchange_rates(
            download_start,
            download_end
        )

        currency_weekly = (
            currency.resample("W-SUN").mean()
        )

        df = df.join(
            currency_weekly[["EUR", "USD"]]
        )

    # --------------------------------------------------
    # ELECTRICITY
    # --------------------------------------------------

    if "electricity" in sources:

        electricity_weekly = load_weekly_electricity(
            area_type=area_type,
            area_number=area_number,
            start_date=download_start,
            end_date=download_end
        )

        df = df.join(
            electricity_weekly[["export", "import"]]
        )

    df.index.name = "date"

    return df


# --------------------------------------------------
# FORECASTING MODEL
# --------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def forecast_reservoir(
    df,
    sources=(),
    history=260,
    horizon=13,
    p=2,
    d=1,
    q=1,
    sp=1,
    sd=0,
    sq=0,
    m=52,
    confidence=0.95,
    learning_rate=0.01,
    calibration_size=52,
    min_train=104
):
    """
    Forecast reservoir filling using River SNARIMAX.

    Returns
    -------
    forecast : pd.DataFrame
        Predictions and empirical prediction intervals.

    metrics : dict
        Historical backtesting metrics.
    """

    if horizon < 1 or not 0 < confidence < 1:
        raise ValueError(
            "Invalid horizon or confidence."
        )

    if history < min_train:
        raise ValueError(
            "History must be >= min_train."
        )

    columns = [
        col
        for source in sources
        for col in DATA_SOURCES[source]
    ]

    data = df[["fill_fraction"] + columns].copy()

    data.index = pd.to_datetime(data.index)

    data = data.sort_index().asfreq("W-SUN")

    # Forward fill short gaps in external variables.
    if columns:

        data[columns] = data[columns].ffill(limit=2)

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Keep the most recent contiguous complete segment.
    complete = data.dropna()

    if complete.empty:
        raise ValueError(
            "No complete forecasting observations."
        )

    data = data.loc[:complete.index.max()]

    missing = data.isna().any(axis=1)

    if missing.any():

        last_missing = np.flatnonzero(missing)[-1]

        data = data.iloc[last_missing + 1:]

    if len(data) < min_train + horizon + 20:

        raise ValueError(
            "Insufficient complete history for reliable "
            "training and interval calibration."
        )

    warmup = (
        d + m * sd
        + max(p, q, m * sp, m * sq)
    )

    if min_train <= warmup:

        raise ValueError(
            "Increase min_train for these seasonal orders."
        )

    # Create a new River SNARIMAX model.
    def create_model():

        regressor = (
            preprocessing.StandardScaler()
            | linear_model.LinearRegression(
                optimizer=optim.SGD(learning_rate),
                intercept_lr=learning_rate
            )
        )

        return time_series.SNARIMAX(
            p=p,
            d=d,
            q=q,
            m=m,
            sp=sp,
            sd=sd,
            sq=sq,
            regressor=regressor
        )

    # Train the model sequentially.
    def fit_model(train):

        model = create_model()

        for _, row in train.iterrows():

            x = {
                col: float(row[col])
                for col in columns
            }

            model.learn_one(
                y=float(row["fill_fraction"]),
                x=x
            )

        return model

    # Estimate future external variables.
    def future_features(train):

        xs = []

        for step in range(1, horizon + 1):

            x = {}

            for col in columns:

                if (
                    col in [
                        "temperature_2m",
                        "precipitation"
                    ]
                    and len(train) >= 52
                    and step <= 52
                ):

                    # Seasonal naive weather forecast.
                    x[col] = float(
                        train[col].iloc[step - 53]
                    )

                else:

                    # Persistence forecast.
                    x[col] = float(
                        train[col].iloc[-1]
                    )

            xs.append(x)

        return xs

    # --------------------------------------------------
    # ROLLING-ORIGIN BACKTESTING
    # --------------------------------------------------

    errors = []
    baseline_errors = []

    first_origin = max(
        min_train,
        len(data) - horizon - calibration_size + 1
    )

    for origin in range(
        first_origin,
        len(data) - horizon + 1
    ):

        train = data.iloc[
            max(0, origin - history):origin
        ]

        actual = (
            data["fill_fraction"]
            .iloc[origin:origin + horizon]
            .to_numpy()
        )

        model = fit_model(train)

        predicted = np.clip(
            model.forecast(
                horizon=horizon,
                xs=future_features(train)
            ),
            0,
            1
        )

        errors.append(actual - predicted)

        # Seasonal naive benchmark.
        if len(train) >= 52:

            seasonal = (
                train["fill_fraction"].iloc[-52:]
            )

            baseline = np.resize(
                seasonal.to_numpy(),
                horizon
            )

            baseline_errors.append(
                actual - baseline
            )

    errors = np.asarray(errors)

    if errors.size == 0:
        raise ValueError(
            "No valid calibration forecasts."
        )

    if not np.isfinite(errors).all():

        raise ValueError(
            "Model produced invalid forecasts."
        )

    # --------------------------------------------------
    # FINAL MODEL
    # --------------------------------------------------

    train = data.tail(history)

    model = fit_model(train)

    prediction = np.clip(
        model.forecast(
            horizon=horizon,
            xs=future_features(train)
        ),
        0,
        1
    )

    if not np.isfinite(prediction).all():

        raise ValueError(
            "Final forecast is not finite."
        )

    # Empirical prediction intervals.
    alpha = 1 - confidence

    lower = prediction + np.quantile(
        errors,
        alpha / 2,
        axis=0
    )

    upper = prediction + np.quantile(
        errors,
        1 - alpha / 2,
        axis=0
    )

    future_dates = pd.date_range(
        start=data.index[-1] + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-SUN"
    )

    forecast = pd.DataFrame(
        {
            "prediction": prediction,
            "lower": np.clip(lower, 0, 1),
            "upper": np.clip(upper, 0, 1)
        },
        index=future_dates
    )

    forecast.index.name = "date"

    # Historical evaluation metrics.
    metrics = {
        "MAE": float(
            np.mean(np.abs(errors))
        ),

        "RMSE": float(
            np.sqrt(np.mean(errors ** 2))
        ),

        "MAE_week_1": float(
            np.mean(np.abs(errors[:, 0]))
        ),

        "MAE_final_week": float(
            np.mean(np.abs(errors[:, -1]))
        ),

        "training_observations": len(train),

        "calibration_forecasts": len(errors),

        "last_observation": data.index[-1],

        "confidence": confidence,

        "mean_interval_width": float(
            (
                forecast["upper"]
                - forecast["lower"]
            ).mean()
        )
    }

    if baseline_errors:

        metrics["baseline_MAE"] = float(
            np.mean(np.abs(baseline_errors))
        )

    return forecast, metrics

@st.cache_data(ttl=3600, show_spinner=False)
def load_actual_observations(
    area_type,
    area_number,
    start_date,
    end_date
):
    """Retrieve actual reservoir filling for the forecast period."""

    start_date = pd.Timestamp(start_date)
    end_date = min(
        pd.Timestamp(end_date),
        pd.Timestamp.today().normalize()
    )

    # No observations can exist for a completely future period.
    if start_date > end_date:
        return pd.Series(dtype=float, name="actual")

    df = load_data(
        start_date,
        end_date,
        area_type
    )

    df = df.loc[df["area_number"] == area_number]

    actual = (
        df.groupby("date")["fill_fraction"]
        .mean()
        .sort_index()
    )

    actual.index = pd.to_datetime(actual.index)
    actual.name = "actual"

    return actual

# --------------------------------------------------
# INTERACTIVE FORECAST PLOT WITH ANIMATION
# --------------------------------------------------

def plot_reservoir_forecast(
    historical,
    forecast,
    confidence=0.95,
    display_weeks=104,
    actual=None
):
    """Create an interactive forecast with animated predictions."""

    origin = (
        forecast.index.min()
        - pd.Timedelta(weeks=1)
    )

    observed = historical.loc[:origin].tail(
        display_weeks
    )

    last_observed = (
        observed["fill_fraction"]
        .dropna()
        .tail(1)
    )

    prediction = pd.concat([
        last_observed,
        forecast["prediction"]
    ])

    fig = go.Figure()

    # Trace 0: Historical observations.
    fig.add_trace(go.Scatter(
        x=observed.index,
        y=observed["fill_fraction"],
        mode="lines",
        name="Observed filling",
        line=dict(
            color="#007F86",
            width=2
        )
    ))

    # Trace 1: Lower prediction interval boundary.
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast["lower"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip"
    ))

    # Trace 2: Upper prediction interval boundary.
    fig.add_trace(go.Scatter(
        x=forecast.index,
        y=forecast["upper"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(86, 140, 181, 0.20)",
        name=f"{confidence:.0%} prediction interval",
        hoverinfo="skip"
    ))

    # Trace 3: Predicted reservoir filling.
    fig.add_trace(go.Scatter(
        x=prediction.index,
        y=prediction,
        mode="lines+markers",
        name="SNARIMAX forecast",
        line=dict(
            color="#568CB5",
            width=2.5,
            dash="dash"
        ),
        marker=dict(size=5)
    ))

    # Trace 4: Previous year's filling.
    previous = historical.loc[
        :origin,
        "fill_fraction"
    ].copy()

    previous.index += pd.Timedelta(weeks=52)

    previous = previous.loc[
        observed.index.min():forecast.index.max()
    ]

    fig.add_trace(go.Scatter(
        x=previous.index,
        y=previous,
        mode="lines",
        name="Previous year",
        line=dict(
            color="#9CA3AF",
            width=1.5,
            dash="dot"
        )
    ))

    # Trace 5: Actual filling during the forecast period.
    if actual is not None and actual.notna().any():

        actual = actual.reindex(forecast.index)

        actual_plot = pd.concat([
            last_observed,
            actual
        ])

        fig.add_trace(go.Scatter(
            x=actual_plot.index,
            y=actual_plot,
            mode="lines+markers",
            name="Actual filling",
            line=dict(
                color="#E9A23B",
                width=2.5
            ),
            marker=dict(size=5),
            connectgaps=False
        ))

    # Vertical forecast origin line.
    fig.add_shape(
        type="line",
        x0=origin,
        x1=origin,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(
            color="#16324F",
            width=1.5,
            dash="dot"
        )
    )

    fig.add_annotation(
        x=origin,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast origin",
        showarrow=False,
        yshift=10
    )

    # --------------------------------------------------
    # ANIMATION FRAMES
    # --------------------------------------------------

    frames = []

    for i in range(len(forecast) + 1):

        partial_forecast = forecast.iloc[:i]

        partial_prediction = pd.concat([
            last_observed,
            partial_forecast["prediction"]
        ])

        # Hide the entire prediction in the first frame.
        if i == 0:

            prediction_x = []
            prediction_y = []

        else:

            prediction_x = partial_prediction.index.tolist()
            prediction_y = partial_prediction.tolist()

        frames.append(
            go.Frame(
                name=str(i),

                data=[

                    # Lower prediction interval.
                    go.Scatter(
                        x=partial_forecast.index.tolist(),
                        y=partial_forecast["lower"].tolist()
                    ),

                    # Upper prediction interval.
                    go.Scatter(
                        x=partial_forecast.index.tolist(),
                        y=partial_forecast["upper"].tolist()
                    ),

                    # Predicted values.
                    go.Scatter(
                        x=prediction_x,
                        y=prediction_y
                    )
                ],

                # Only animate prediction traces.
                traces=[1, 2, 3]
            )
        )

    fig.frames = frames

    # --------------------------------------------------
    # PLOT CONFIGURATION
    # --------------------------------------------------

    fig.update_layout(

        title="Reservoir Filling Forecast — River SNARIMAX",

        xaxis_title="Date",
        yaxis_title="Reservoir filling",

        height=650,

        hovermode="x unified",

        margin=dict(
            l=50,
            r=30,
            t=100,
            b=150
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="left",
            x=0
        ),

        # Play and Pause buttons below the plot.
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "showactive": False,

                "x": 0.5,
                "xanchor": "center",

                "y": -0.30,
                "yanchor": "top",

                "buttons": [

                    {
                        "label": "▶ Play",
                        "method": "animate",

                        "args": [
                            [
                                str(i)
                                for i in range(
                                    len(forecast) + 1
                                )
                            ],
                            {
                                "mode": "immediate",
                                "fromcurrent": False,

                                "frame": {
                                    "duration": 250,
                                    "redraw": True
                                },

                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    },

                    {
                        "label": "❚❚ Pause",
                        "method": "animate",

                        "args": [
                            [None],
                            {
                                "mode": "immediate",

                                "frame": {
                                    "duration": 0,
                                    "redraw": False
                                },

                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    }
                ]
            }
        ]
    )

    # Keep axes fixed during animation.
    fig.update_yaxes(
        range=[0, 1],
        tickformat=".0%"
    )

    fig.update_xaxes(
        range=[
            observed.index.min(),
            forecast.index.max()
        ],
        rangeslider_visible=True
    )

    return fig