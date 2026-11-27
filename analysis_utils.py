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

def plot_weather_outliers(
    weather,
    variable="temperature_2m",
    variable_label="Temperature (°C)",
    cutoff=100,
    n_std=3
):
    """Plot weather observations with DCT-based robust SPC limits."""

    series = weather[variable].sort_index()
    values = series.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Weather series contains missing or invalid values.")

    if not 0 <= cutoff < len(values):
        raise ValueError("DCT cutoff must be smaller than the series length.")

    # Remove low-frequency variation.
    residual = dct_highpass(values, cutoff=cutoff)
    baseline = values - residual

    # Calculate robust SPC limits.
    lower, upper, outliers, robust_std = robust_spc(
        residual,
        baseline,
        n_std=n_std
    )

    # Store detected outliers.
    outlier_df = pd.DataFrame({
        "time": series.index[outliers],
        variable: values[outliers],
        "lower_limit": lower[outliers],
        "upper_limit": upper[outliers]
    })

    fig = go.Figure()

    # Original weather observations.
    fig.add_trace(go.Scatter(
        x=series.index,
        y=values,
        mode="lines",
        name=variable_label,
        line=dict(color="#007F86", width=1)
    ))

    # Upper control limit.
    fig.add_trace(go.Scatter(
        x=series.index,
        y=upper,
        mode="lines",
        name="Upper control limit",
        line=dict(
            color="#E9A23B",
            width=1.5,
            dash="dash"
        )
    ))

    # Lower control limit.
    fig.add_trace(go.Scatter(
        x=series.index,
        y=lower,
        mode="lines",
        name="Lower control limit",
        line=dict(
            color="#E9A23B",
            width=1.5,
            dash="dash"
        )
    ))

    # Highlight detected outliers.
    fig.add_trace(go.Scatter(
        x=series.index[outliers],
        y=values[outliers],
        mode="markers",
        name="Outliers",
        marker=dict(
            color="#C0392B",
            size=5
        )
    ))

    fig.update_layout(
        title=(
            f"{variable_label} — DCT & SPC"
            f"<br><sup>Cutoff={cutoff}, k={n_std}</sup>"
        ),
        xaxis_title="Date",
        yaxis_title=variable_label,
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

    fig.update_xaxes(rangeslider_visible=True)

    return fig, outlier_df

def plot_weather_anomalies(
    weather,
    variable="temperature_2m",
    variable_label="Temperature (°C)",
    contamination=0.01,
    n_neighbors=20
):
    """Detect weather anomalies using Local Outlier Factor."""

    series = weather[variable].sort_index().dropna()
    values = series.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Weather series contains invalid values.")

    if not 1 <= n_neighbors < len(series):
        raise ValueError(
            "Number of neighbors must be smaller than the series length."
        )

    # LOF expects a two-dimensional feature matrix.
    X = values.reshape(-1, 1)

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination
    )

    labels = lof.fit_predict(X)
    anomalies = labels == -1

    # Store detected anomalies.
    anomaly_df = pd.DataFrame({
        "time": series.index[anomalies],
        variable: values[anomalies],
        "lof_score": -lof.negative_outlier_factor_[anomalies]
    })

    fig = go.Figure()

    # Original weather observations.
    fig.add_trace(go.Scatter(
        x=series.index,
        y=values,
        mode="lines",
        name=variable_label,
        line=dict(
            color="#007F86",
            width=1
        ),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d %H:%M}<br>"
            f"{variable_label}: %{{y:.2f}}"
            "<extra></extra>"
        )
    ))

    # Highlight detected anomalies.
    fig.add_trace(go.Scatter(
        x=series.index[anomalies],
        y=values[anomalies],
        mode="markers",
        name="LOF anomalies",
        marker=dict(
            color="#C0392B",
            size=6
        ),
        customdata=anomaly_df["lof_score"].to_numpy(),
        hovertemplate=(
            "Date: %{x|%Y-%m-%d %H:%M}<br>"
            f"{variable_label}: %{{y:.2f}}<br>"
            "LOF score: %{customdata:.3f}"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title=(
            f"{variable_label} — LOF"
            f"<br><sup>Contamination={contamination:.1%}, "
            f"Neighbors={n_neighbors}</sup>"
        ),
        xaxis_title="Date",
        yaxis_title=variable_label,
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

    fig.update_xaxes(rangeslider_visible=True)

    return fig, anomaly_df