
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