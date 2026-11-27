import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy.signal import spectrogram
from statsmodels.tsa.seasonal import STL

from scipy.fft import dct, idct
from sklearn.neighbors import LocalOutlierFactor


def prepare_weekly_series(df):
    """Prepare Norway's weekly reservoir fill ratio."""

    data = df.loc[df["area_number"] == 0].copy()

    if data.empty:
        raise ValueError("No national reservoir observations found.")

    series = (
        data.groupby("date")["fill_fraction"]
        .mean()
        .sort_index()
    )

    series.index = pd.to_datetime(series.index)

    # Ensure one observation per week.
    series = series.resample("7D").mean()
    series = series.interpolate(method="time", limit=2)

    if series.isna().any():
        raise ValueError("Missing weekly observations remain.")

    series.name = "fill_ratio"
    return series


def plot_stl(series, period=52, seasonal=13,
             trend=105, robust=True):
    """Interactive STL decomposition of weekly reservoir filling."""

    # Perform STL decomposition
    result = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        robust=robust
    ).fit()

    # Create four subplots with a shared time axis
    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "Observed",
            "Trend",
            "Seasonal",
            "Residual"
        ]
    )

    # Components and corresponding theme colours
    components = [
        (series, "Observed", "#007F86"),
        (result.trend, "Trend", "#16324F"),
        (result.seasonal, "Seasonal", "#E9A23B"),
        (result.resid, "Residual", "#568CB5")
    ]

    # Add each component to its own subplot
    for row, (values, name, color) in enumerate(
        components, start=1
    ):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=values,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    "%{y:.3f}<extra></extra>"
                )
            ),
            row=row,
            col=1
        )

    # Add zero reference line to residuals
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="#9CA3AF",
        row=4,
        col=1
    )

    # Customise figure layout
    fig.update_layout(
        title=(
            "STL Decomposition of Reservoir Filling"
            f"<br><sup>Seasonal={seasonal}, "
            f"Trend={trend}, Robust={robust}</sup>"
        ),
        height=850,
        showlegend=False,
        hovermode="x unified",
        margin=dict(l=60, r=30, t=100, b=50)
    )

    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Fill fraction", row=1, col=1)
    fig.update_yaxes(title_text="Trend", row=2, col=1)
    fig.update_yaxes(title_text="Seasonal", row=3, col=1)
    fig.update_yaxes(title_text="Residual", row=4, col=1)

    return fig, result



def plot_spectrogram(series, window_length=156, overlap=130):
    """Plot the spectrogram of weekly reservoir filling data."""

    # Prepare and validate the time series
    series = series.copy().sort_index()
    series.index = pd.to_datetime(series.index)

    if series.isna().any():
        raise ValueError("The time series contains missing values.")

    if not series.index.to_series().diff().dropna().eq(
        pd.Timedelta(weeks=1)
    ).all():
        raise ValueError("The time series must have weekly observations.")

    if not (0 <= overlap < window_length <= len(series)):
        raise ValueError("Require 0 <= overlap < window_length <= series length.")

    # Compute the spectrogram using one observation per week
    frequencies, times, power = spectrogram(
        series.to_numpy(),
        fs=52.0,
        window="hann",
        nperseg=window_length,
        noverlap=overlap,
        detrend="linear",
        scaling="spectrum",
        mode="psd"
    )

    # Convert window-centre offsets into calendar dates
    dates = series.index[0] + pd.to_timedelta(times * 52, unit="W")

    # Convert power to decibels for better visualization
    power_db = 10 * np.log10(np.maximum(power, 1e-12))

    # Create an interactive time-frequency heatmap
    fig = go.Figure(
        data=go.Heatmap(
            x=dates,
            y=frequencies,
            z=power_db,
            colorscale=[
                [0.00, "#16324F"],
                [0.35, "#246582"],
                [0.65, "#007F86"],
                [0.85, "#8CC5A8"],
                [1.00, "#E9A23B"]
            ],
            colorbar=dict(title="Power (dB)"),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Frequency: %{y:.2f} cycles/year<br>"
                "Power: %{z:.2f} dB"
                "<extra></extra>"
            )
        )
    )

    # Reference line for annual seasonality
    fig.add_hline(
        y=1,
        line_dash="dash",
        line_color="#FFFFFF",
        line_width=2,
        annotation_text="Annual cycle (1 cycle/year)",
        annotation_position="top left",
        annotation_font_color="#FFFFFF",
        annotation_bgcolor="#16324F"
    )

    # Customise figure layout
    fig.update_layout(
        title="Spectrogram of Reservoir Filling",
        xaxis_title="Year",
        yaxis_title="Frequency (cycles/year)",
        height=500,
        margin=dict(l=60, r=30, t=60, b=50)
    )

    # Focus on the low-frequency components
    fig.update_yaxes(
        range=[0, 4],
        dtick=1
    )

    return fig


def dct_highpass(series, cutoff=30):
    """Remove the lowest-frequency DCT components from a time series."""

    values = np.asarray(series, dtype=float)

    # Transform the time series into DCT coefficients
    coefficients = dct(values, norm="ortho")

    # Remove the lowest-frequency components
    coefficients[:cutoff] = 0

    # Reconstruct the high-frequency residual
    residual = idct(coefficients, norm="ortho")

    return residual

def robust_spc(residual, baseline, n_std=3):
    """Calculate robust SPC boundaries and identify outliers."""

    # Estimate the centre and scale from the full-year residuals
    median = np.median(residual)
    mad = np.median(np.abs(residual - median))
    robust_std = 1.4826 * mad

    # Construct boundaries on the original precipitation scale
    upper = baseline + median + n_std * robust_std
    lower = baseline + median - n_std * robust_std

    # Identify observations outside the boundaries
    outliers = (residual > median + n_std * robust_std) | (
        residual < median - n_std * robust_std
    )

    return lower, upper, outliers, robust_std

def plot_precipitation_outliers(weather, cutoff=100, n_std=3):
    """Plot average precipitation with DCT-based robust SPC limits."""

    # Extract the original precipitation series
    precipitation = weather["precipitation"]
    values = precipitation.to_numpy()

    # Remove low-frequency variation using the DCT
    residual = dct_highpass(values, cutoff=cutoff)
    baseline = values - residual

    # Calculate robust SPC limits
    lower, upper, outliers, robust_std = robust_spc(
        residual, baseline, n_std=n_std
    )

    # Store detected outliers with their original precipitation values
    outlier_df = pd.DataFrame({
        "time": precipitation.index[outliers],
        "precipitation": values[outliers],
        "lower_limit": lower[outliers],
        "upper_limit": upper[outliers]
    })

    # Create an interactive Plotly figure
    fig = go.Figure()

    # Original precipitation time series
    fig.add_trace(go.Scatter(
        x=precipitation.index,
        y=values,
        mode="lines",
        name="Average precipitation",
        line=dict(color="#007F86", width=1)
    ))

    # Upper control limit
    fig.add_trace(go.Scatter(
        x=precipitation.index,
        y=upper,
        mode="lines",
        name="Upper control limit",
        line=dict(
            color="#E9A23B",
            width=1.5,
            dash="dash"
        )
    ))

    # Lower control limit
    fig.add_trace(go.Scatter(
        x=precipitation.index,
        y=lower,
        mode="lines",
        name="Lower control limit",
        line=dict(
            color="#E9A23B",
            width=1.5,
            dash="dash"
        )
    ))

    # Highlight detected outliers
    fig.add_trace(go.Scatter(
        x=precipitation.index[outliers],
        y=values[outliers],
        mode="markers",
        name="Outliers",
        marker=dict(
            color="#C0392B",
            size=5
        )
    ))

    # Customise figure layout
    fig.update_layout(
        title=(
            f"NO3 Precipitation Outliers — "
            f"DCT cutoff={cutoff}, k={n_std}"
        ),
        xaxis_title="Date",
        yaxis_title="Average precipitation (mm/hour)",
        height=550,
        hovermode="x unified",
        margin=dict(l=40, r=30, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    # Add an interactive date range slider
    fig.update_xaxes(
        rangeslider_visible=True
    )

    # Print summary statistics
    print(f"Robust standard deviation: {robust_std:.4f}")
    print(f"Number of outliers: {outliers.sum()}")
    print(f"Outlier percentage: {outliers.mean() * 100:.2f}%")

    return fig, outlier_df

def plot_temperature_anomalies(weather, contamination=0.01, n_neighbors=20):
    """Detect and plot temperature anomalies using Local Outlier Factor."""

    # Extract the spatially averaged temperature series
    temperature = weather["temperature"].dropna()

    # LOF expects a 2D feature matrix
    X = temperature.to_numpy().reshape(-1, 1)

    # Fit LOF and classify observations
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    labels = lof.fit_predict(X)
    anomalies = labels == -1

    # Store anomalous observations and their LOF scores
    anomaly_df = pd.DataFrame({
        "time": temperature.index[anomalies],
        "temperature": temperature.to_numpy()[anomalies],
        "lof_score": -lof.negative_outlier_factor_[anomalies]
    })

    # Create an interactive Plotly figure
    fig = go.Figure()

    # Original temperature time series
    fig.add_trace(go.Scatter(
        x=temperature.index,
        y=temperature.to_numpy(),
        mode="lines",
        name="Average temperature",
        line=dict(
            color="#007F86",
            width=1
        ),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d %H:%M}<br>"
            "Temperature: %{y:.2f} °C"
            "<extra></extra>"
        )
    ))

    # Highlight detected LOF anomalies
    fig.add_trace(go.Scatter(
        x=temperature.index[anomalies],
        y=temperature.to_numpy()[anomalies],
        mode="markers",
        name="LOF anomalies",
        marker=dict(
            color="#C0392B",
            size=6
        ),
        customdata=anomaly_df["lof_score"].to_numpy(),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d %H:%M}<br>"
            "Temperature: %{y:.2f} °C<br>"
            "LOF score: %{customdata:.3f}"
            "<extra></extra>"
        )
    ))

    # Customise figure layout
    fig.update_layout(
        title=(
            f"NO3 Temperature Anomalies — "
            f"LOF ({contamination:.1%})"
        ),
        xaxis_title="Date",
        yaxis_title="Average temperature (°C)",
        height=550,
        hovermode="closest",
        margin=dict(l=40, r=30, t=80, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    # Add an interactive date range slider
    fig.update_xaxes(
        rangeslider_visible=True
    )

    # Print summary statistics
    print(f"Total observations: {len(temperature)}")
    print(f"Detected anomalies: {anomalies.sum()}")
    print(f"Anomaly percentage: {anomalies.mean() * 100:.2f}%")

    return fig, anomaly_df
