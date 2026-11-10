
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


# def detect_spc_outliers(series, cutoff=0.02, threshold=3.5):
#     """
#     DCT high-pass filtering followed by robust SPC limits.

#     cutoff: Fraction of low-frequency DCT coefficients removed.
#     threshold: Number of robust standard deviations from median.
#     """
#     values = series.to_numpy(dtype=float)

#     if not 0 < cutoff < 1:
#         raise ValueError("Cutoff must be between 0 and 1.")

#     coefficients = dct(values, norm="ortho")

#     n_remove = max(1, int(len(coefficients) * cutoff))
#     coefficients[:n_remove] = 0

#     residual = idct(coefficients, norm="ortho")

#     median = np.median(residual)
#     mad = np.median(np.abs(residual - median))

#     robust_sigma = 1.4826 * mad

#     if robust_sigma == 0:
#         raise ValueError("MAD is zero; control limits cannot be calculated.")

#     lower_limit = median - threshold * robust_sigma
#     upper_limit = median + threshold * robust_sigma

#     outliers = (residual < lower_limit) | (residual > upper_limit)

#     result = pd.DataFrame({
#         "temperature": values,
#         "filtered_signal": residual,
#         "outlier": outliers
#     }, index=series.index)

#     return result, lower_limit, upper_limit


# def plot_spc(result, lower_limit, upper_limit):
#     """Plot the filtered signal and robust control limits."""

#     fig, ax = plt.subplots(figsize=(12, 5))

#     ax.plot(
#         result.index,
#         result["filtered_signal"],
#         linewidth=0.8,
#         label="High-pass filtered temperature"
#     )

#     flagged = result[result["outlier"]]

#     ax.scatter(
#         flagged.index,
#         flagged["filtered_signal"],
#         color="red",
#         s=15,
#         label="Outliers",
#         zorder=3
#     )

#     ax.axhline(
#         upper_limit,
#         color="orange",
#         linestyle="--",
#         label="Upper control limit"
#     )

#     ax.axhline(
#         lower_limit,
#         color="orange",
#         linestyle="--",
#         label="Lower control limit"
#     )

#     ax.set_title("Temperature Outlier Detection (DCT + SPC)")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Filtered temperature (°C)")
#     ax.grid(alpha=0.3)
#     ax.legend()
#     fig.autofmt_xdate()
#     fig.tight_layout()

#     return fig


# def detect_lof_anomalies(
#     series,
#     n_neighbors=20,
#     contamination=0.02
# ):
#     """Detect local temperature anomalies using LOF."""

#     if not 0 < contamination <= 0.5:
#         raise ValueError("Contamination must be between 0 and 0.5.")

#     # Include daily and annual time features so that LOF
#     # compares temperature in a temporal context.
#     hours = series.index.hour.to_numpy()
#     days = series.index.dayofyear.to_numpy()

#     features = pd.DataFrame({
#         "temperature": series.to_numpy(),
#         "hour_sin": np.sin(2 * np.pi * hours / 24),
#         "hour_cos": np.cos(2 * np.pi * hours / 24),
#         "year_sin": np.sin(2 * np.pi * days / 366),
#         "year_cos": np.cos(2 * np.pi * days / 366)
#     }, index=series.index)

#     # Standardize features to comparable scales.
#     std = features.std().replace(0, 1)
#     scaled = (features - features.mean()) / std

#     model = LocalOutlierFactor(
#         n_neighbors=n_neighbors,
#         contamination=contamination
#     )

#     labels = model.fit_predict(scaled)

#     result = pd.DataFrame({
#         "temperature": series,
#         "anomaly": labels == -1,
#         "lof_score": -model.negative_outlier_factor_
#     }, index=series.index)

#     return result


# def plot_lof(result):
#     """Plot temperature observations with LOF anomalies."""

#     fig, ax = plt.subplots(figsize=(12, 5))

#     ax.plot(
#         result.index,
#         result["temperature"],
#         linewidth=0.8,
#         label="Temperature"
#     )

#     anomalies = result[result["anomaly"]]

#     ax.scatter(
#         anomalies.index,
#         anomalies["temperature"],
#         color="red",
#         s=18,
#         label="LOF anomalies",
#         zorder=3
#     )

#     ax.set_title("Temperature Anomaly Detection (LOF)")
#     ax.set_xlabel("Date")
#     ax.set_ylabel("Temperature (°C)")
#     ax.grid(alpha=0.3)
#     ax.legend()
#     fig.autofmt_xdate()
#     fig.tight_layout()

#     return fig

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