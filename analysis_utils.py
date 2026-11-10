
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    """STL decomposition of weekly reservoir filling."""

    result = STL(
        series,
        period=period,
        seasonal=seasonal,
        trend=trend,
        robust=robust
    ).fit()

    fig = result.plot()
    fig.set_size_inches(12, 9)

    fig.suptitle(
        "STL Decomposition of Reservoir Filling "
        f"(seasonal={seasonal}, trend={trend}, robust={robust})",
        fontsize=13
    )

    fig.tight_layout()
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

    # Plot the time-frequency representation
    fig, ax = plt.subplots(figsize=(14, 6))

    mesh = ax.pcolormesh(
        dates,
        frequencies,
        power_db,
        shading="auto",
        cmap="viridis"
    )

    # Reference line for annual seasonality
    ax.axhline(
        1,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="Annual cycle (1 cycle/year)"
    )

    ax.set(
        title="Spectrogram of Reservoir Filling",
        xlabel="Year",
        ylabel="Frequency (cycles/year)"
    )

    ax.set_ylim(0, 4)
    ax.set_yticks(np.arange(0, 5, 1))
    ax.legend()

    fig.colorbar(
        mesh,
        ax=ax,
        label="Power (dB)"
    )

    fig.tight_layout()

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

    # Plot the original precipitation and SPC boundaries
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        precipitation.index,
        values,
        color="steelblue",
        linewidth=0.7,
        label="Average precipitation"
    )

    ax.plot(
        precipitation.index,
        upper,
        color="darkorange",
        linestyle="--",
        linewidth=1,
        label="Upper control limit"
    )

    ax.plot(
        precipitation.index,
        lower,
        color="darkorange",
        linestyle="--",
        linewidth=1,
        label="Lower control limit"
    )

    # Highlight outliers using the original precipitation values
    ax.scatter(
        precipitation.index[outliers],
        values[outliers],
        color="red",
        s=12,
        zorder=3,
        label="Outliers"
    )

    ax.set(
        title=f"NO3 Precipitation Outliers — DCT cutoff={cutoff}, k={n_std}",
        xlabel="Date",
        ylabel="Average precipitation (mm/hour)"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

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

    # Plot the original temperature series
    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        temperature.index,
        temperature,
        color="steelblue",
        linewidth=0.7,
        label="Average temperature"
    )

    # Highlight anomalous observations
    ax.scatter(
        temperature.index[anomalies],
        temperature.to_numpy()[anomalies],
        color="red",
        s=15,
        zorder=3,
        label="LOF anomalies"
    )

    ax.set(
        title=f"NO3 Temperature Anomalies — LOF ({contamination:.1%})",
        xlabel="Date",
        ylabel="Average temperature (°C)"
    )

    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    # Print summary statistics
    print(f"Total observations: {len(temperature)}")
    print(f"Detected anomalies: {anomalies.sum()}")
    print(f"Anomaly percentage: {anomalies.mean() * 100:.2f}%")

    return fig, anomaly_df