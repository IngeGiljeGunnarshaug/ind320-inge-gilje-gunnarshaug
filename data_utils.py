import pandas as pd
import requests
import streamlit as st


# NVE Magasinstatistikk API endpoint.
NVE_URL = "https://biapi.nve.no/magasinstatistikk/api/Magasinstatistikk/HentOffentligData"

# Readable names for the reservoir measurements.
MEASUREMENTS = {
    "fill_fraction": "Fill ratio",
    "capacity_twh": "Capacity (TWh)",
    "stored_energy_twh": "Stored energy (TWh)",
    "previous_week_fill_fraction": "Previous week's fill ratio",
    "fill_fraction_change": "Fill ratio change",
}

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_MEASUREMENTS = {
    "temperature_2m_mean": "Temperature (°C)",
    "precipitation_sum": "Precipitation (mm)",
    "rain_sum": "Rain (mm)",
    "snowfall_sum": "Snowfall (cm)",
    "wind_speed_10m_mean": "Wind speed (km/h)",
    "shortwave_radiation_sum": "Solar radiation (MJ/m²)",
    "et0_fao_evapotranspiration": "Evapotranspiration (mm)"
}

WEATHER_AGGREGATIONS = {
    "temperature_2m_mean": "mean",
    "precipitation_sum": "sum",
    "rain_sum": "sum",
    "snowfall_sum": "sum",
    "wind_speed_10m_mean": "mean",
    "shortwave_radiation_sum": "sum",
    "et0_fao_evapotranspiration": "sum"
}

HOURLY_WEATHER_MEASUREMENTS = {
    "temperature_2m": "Temperature (°C)",
    "precipitation": "Precipitation (mm/hour)",
    "rain": "Rain (mm/hour)",
    "snowfall": "Snowfall (cm/hour)",
    "relative_humidity_2m": "Relative humidity (%)",
    "wind_speed_10m": "Wind speed (km/h)",
    "shortwave_radiation": "Solar radiation (W/m²)",
    "et0_fao_evapotranspiration": "Evapotranspiration (mm/hour)"
}

@st.cache_data(ttl=3600, show_spinner="Downloading NVE data...")
def load_data(start_date, end_date, area_type):
    """
    Download reservoir data from NVE for the selected interval
    and area type, then prepare it for Streamlit.
    """

    # Retrieve observations for the selected date interval.
    params = {
        "datoFra": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "datoTil": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
    }

    response = requests.get(
        NVE_URL,
        params=params,
        timeout=60
    )
    response.raise_for_status()

    data = response.json()

    # Some API responses wrap observations in a data field.
    if isinstance(data, dict):
        data = data.get("data", data.get("value", data))

    df = pd.DataFrame(data)

    # Rename the Norwegian API columns.
    df = df.rename(columns={
        "dato_Id": "date",
        "omrType": "area_type",
        "omrnr": "area_number",
        "iso_aar": "iso_year",
        "iso_uke": "iso_week",
        "fyllingsgrad": "fill_fraction",
        "kapasitet_TWh": "capacity_twh",
        "fylling_TWh": "stored_energy_twh",
        "neste_Publiseringsdato": "next_publication_date",
        "fyllingsgrad_forrige_uke": "previous_week_fill_fraction",
        "endring_fyllingsgrad": "fill_fraction_change",
    })

    # Convert dates and numerical values to appropriate types.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["area_number"] = pd.to_numeric(
        df["area_number"], errors="coerce"
    )

    for column in MEASUREMENTS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

    # Keep the selected geographical area type.
    df = df.loc[df["area_type"] == area_type].copy()

    # Keep observations within the selected date interval.
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    df = df.loc[
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].copy()

    # Sort observations chronologically within each area.
    df = df.sort_values(
        ["area_number", "date"]
    ).reset_index(drop=True)

    return df

@st.cache_data(ttl=3600, show_spinner=False)
def load_weather(
    points,
    start_date,
    end_date,
    variables,
    resolution="daily"
):
    """Download ERA5 weather and calculate spatial averages."""

    if resolution not in ["daily", "hourly"]:
        raise ValueError("Resolution must be daily or hourly.")

    if points.empty:
        raise ValueError("No weather locations supplied.")

    params = {
        "latitude": ",".join(points["latitude"].astype(str)),
        "longitude": ",".join(points["longitude"].astype(str)),
        "start_date": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "end_date": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        resolution: ",".join(variables),
        "models": "era5",
        "timezone": "GMT"
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=120
    )
    response.raise_for_status()

    data = response.json()

    if isinstance(data, dict):
        data = [data]

    if len(data) != len(points):
        raise ValueError("Unexpected number of weather locations.")

    df = pd.concat([
        pd.DataFrame(location[resolution])
        for location in data
    ], ignore_index=True)

    df["time"] = pd.to_datetime(df["time"])

    return df.groupby("time")[list(variables)].mean().sort_index()