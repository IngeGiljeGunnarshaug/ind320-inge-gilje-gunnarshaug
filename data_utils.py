import pandas as pd
import requests
import streamlit as st
from io import StringIO
import os
from pathlib import Path
import time
import numpy as np

from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from concurrent.futures import ThreadPoolExecutor

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

ENTSOE_CONNECTIONS = {
    "NO_1": ["NO_2", "NO_3", "NO_5", "SE_3"],
    "NO_2": ["NO_1", "NO_5", "DK_1", "NL", "DE_LU", "GB"],
    "NO_3": ["NO_1", "NO_4", "NO_5", "SE_2"],
    "NO_4": ["NO_3", "SE_1", "SE_2", "FI"],
    "NO_5": ["NO_1", "NO_2", "NO_3"]
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

def aggregate_weekly_weather(
    weather,
    observation_dates,
    variable
):
    """Aggregate daily weather to the NVE observation weeks."""

    weather = weather.copy().sort_index()
    weather.index = pd.to_datetime(weather.index)

    aggregation = WEATHER_AGGREGATIONS[variable]

    dates = pd.DatetimeIndex(
        observation_dates
    ).sort_values().unique()

    values = []

    for end_date in dates:

        start_date = end_date - pd.Timedelta(days=6)

        daily = weather.loc[
            start_date:end_date,
            variable
        ].dropna()

        # Require a complete week.
        if len(daily) != 7:
            value = float("nan")

        elif aggregation == "sum":
            value = daily.sum()

        else:
            value = daily.mean()

        values.append(value)

    return pd.Series(
        values,
        index=dates,
        name=variable
    )

@st.cache_data(ttl=3600, show_spinner=False)
def download_exchange_rates(start_date, end_date):
    """Download EUR and USD exchange rates from Norges Bank."""

    url = (
        "https://data.norges-bank.no/api/data/"
        "EXR/B.EUR+USD.NOK.SP"
    )

    params = {
        "startPeriod": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
        "endPeriod": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
        "format": "csvfile"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(StringIO(response.text))

    df = df[["TIME_PERIOD", "BASE_CUR", "OBS_VALUE"]]

    df = df.rename(columns={
        "TIME_PERIOD": "date",
        "BASE_CUR": "currency",
        "OBS_VALUE": "exchange_rate"
    })

    df["date"] = pd.to_datetime(df["date"])

    df["exchange_rate"] = pd.to_numeric(
        df["exchange_rate"],
        errors="coerce"
    )

    df = df.pivot(
        index="date",
        columns="currency",
        values="exchange_rate"
    )

    df = df.sort_index()
    df.columns.name = None

    return df

@st.cache_resource
def get_entsoe_client():
    """Initialize the ENTSO-E API client."""

    env_path = Path(__file__).resolve().parent / ".env"

    load_dotenv(env_path)

    # Local environment variable.
    token = os.getenv("ENTSOE_API_TOKEN")

    # Streamlit Cloud fallback.
    if not token:

        try:
            token = st.secrets["ENTSOE_API_TOKEN"]

        except (KeyError, FileNotFoundError):
            token = None

    if not token:

        raise ValueError(
            "ENTSOE_API_TOKEN was not found in "
            ".env or Streamlit secrets."
        )

    return EntsoePandasClient(api_key=token)

def get_entsoe_connections(area_type, area_number):
    """Return electricity connections for the selected area."""

    if area_type == "EL":

        area = f"NO_{area_number}"

        if area not in ENTSOE_CONNECTIONS:
            raise ValueError(
                f"Unknown electricity area: {area}"
            )

        return [
            (area, neighbour)
            for neighbour in ENTSOE_CONNECTIONS[area]
        ]

    elif area_type == "NO":

        # Only international electricity transfers.
        return [
            (area, neighbour)
            for area, neighbours in ENTSOE_CONNECTIONS.items()
            for neighbour in neighbours
            if not neighbour.startswith("NO_")
        ]

    elif area_type == "VASS":

        raise ValueError(
            "Electricity transfers are not available "
            "for VASS regions."
        )

    else:

        raise ValueError(
            f"Unknown area type: {area_type}"
        )

def convert_flow_to_daily_energy(flow):
    """
    Convert physical electricity flows from MW to daily MWh.

    Supports hourly, 30-minute and 15-minute observations,
    allowing the temporal resolution to change between days.
    """

    if flow.empty:
        return pd.Series(dtype=float, name="mwh")

    flow = flow.copy().sort_index()

    flow.index = pd.to_datetime(
        flow.index,
        utc=True
    )

    flow = flow[
        ~flow.index.duplicated(keep="last")
    ]

    flow = pd.to_numeric(
        flow,
        errors="coerce"
    )

    daily_energy = {}

    resolutions = [
        pd.Timedelta(minutes=15),
        pd.Timedelta(minutes=30),
        pd.Timedelta(hours=1)
    ]

    for date, observations in flow.groupby(
        flow.index.normalize()
    ):

        daily_energy[date] = np.nan

        if not np.isfinite(
            observations.to_numpy()
        ).all():
            continue

        # Identify the resolution by checking whether
        # the timestamps form a complete daily grid.
        for resolution in resolutions:

            expected_index = pd.date_range(
                start=date,
                end=date + pd.Timedelta(days=1),
                freq=resolution,
                inclusive="left"
            )

            if observations.index.equals(expected_index):

                interval_hours = (
                    resolution.total_seconds() / 3600
                )

                daily_energy[date] = (
                    observations.sum() * interval_hours
                )

                break

    daily = pd.Series(
        daily_energy,
        dtype=float,
        name="mwh"
    ).sort_index()

    return daily.asfreq("D")

@st.cache_data(
    ttl=86400 * 7,
    show_spinner=False
)
def download_entsoe_year(
    source,
    destination,
    year
):
    """
    Download and cache one calendar year of electricity flows.
    """

    client = EntsoePandasClient(
        api_key=get_entsoe_client().api_key,
        timeout=60,
        retry_count=2,
        retry_delay=5
    )

    start = pd.Timestamp(
        year=year,
        month=1,
        day=1,
        tz="UTC"
    )

    end = pd.Timestamp(
        year=year + 1,
        month=1,
        day=1,
        tz="UTC"
    )

    # Do not request future observations.
    today = pd.Timestamp.now(
        tz="UTC"
    ).normalize()

    end = min(end, today)

    if start >= end:
        return pd.Series(dtype=float, name="mwh")

    request_start = time.perf_counter()

    print(
        f"START: {source} -> {destination} | {year}",
        flush=True
    )

    try:

        flow = client.query_crossborder_flows(
            country_code_from=source,
            country_code_to=destination,
            start=start,
            end=end
        )

    except NoMatchingDataError:

        print(
            f"NO DATA: {source} -> {destination} | {year}",
            flush=True
        )

        return pd.Series(dtype=float, name="mwh")

    if flow.empty:
        return pd.Series(dtype=float, name="mwh")

    flow.index = pd.to_datetime(
        flow.index,
        utc=True
    )

    flow = flow.loc[
        (flow.index >= start)
        & (flow.index < end)
    ]

    daily = convert_flow_to_daily_energy(flow)

    expected_dates = pd.date_range(
        start=start,
        end=end,
        freq="D",
        inclusive="left"
    )

    daily = daily.reindex(expected_dates)

    print(
        f"DONE: {source} -> {destination} | "
        f"{year} | "
        f"{time.perf_counter() - request_start:.2f}s | "
        f"{len(daily)} days | "
        f"{daily.isna().sum()} missing days",
        flush=True
    )

    return daily

@st.cache_data(
    ttl=3600,
    show_spinner=False
)
def download_entsoe_flow(
    source,
    destination,
    start_date,
    end_date
):
    """
    Retrieve electricity flows using cached annual downloads.
    """

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start.tzinfo is None:
        start = start.tz_localize("UTC")
    else:
        start = start.tz_convert("UTC")

    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    start = start.normalize()
    end = end.normalize()

    if start > end:
        raise ValueError(
            "Start date must precede end date."
        )

    results = []

    for year in range(
        start.year,
        end.year + 1
    ):

        daily = download_entsoe_year(
            source=source,
            destination=destination,
            year=year
        )

        if not daily.empty:
            results.append(daily)

    if not results:
        return pd.Series(dtype=float, name="mwh")

    result = pd.concat(results).sort_index()

    result = result[
        ~result.index.duplicated(keep="last")
    ]

    result = result.loc[start:end]

    result.name = "mwh"

    return result

@st.cache_data(ttl=86400, show_spinner=False)
def load_weekly_electricity(
    area_type,
    area_number,
    start_date,
    end_date,
    max_workers=4
):
    """Download electricity flows concurrently and aggregate weekly."""

    connections = get_entsoe_connections(
        area_type,
        area_number
    )

    imports = {}
    exports = {}

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:

        import_tasks = {}
        export_tasks = {}

        for area, neighbour in connections:

            connection = f"{area}-{neighbour}"

            export_tasks[connection] = executor.submit(
                download_entsoe_flow,
                area,
                neighbour,
                start_date,
                end_date
            )

            import_tasks[connection] = executor.submit(
                download_entsoe_flow,
                neighbour,
                area,
                start_date,
                end_date
            )

        for connection, task in import_tasks.items():

            imports[connection] = task.result()

        for connection, task in export_tasks.items():

            exports[connection] = task.result()

    # Combine all connections.
    import_df = pd.DataFrame(imports)
    export_df = pd.DataFrame(exports)

    if import_df.empty or export_df.empty:

        raise ValueError(
            "No electricity transfer data available "
            "for the selected area and period."
        )

    # Remove connections without observations.
    import_df = import_df.dropna(
        axis=1,
        how="all"
    )

    export_df = export_df.dropna(
        axis=1,
        how="all"
    )

    if import_df.empty or export_df.empty:

        raise ValueError(
            "No usable electricity connections found."
        )

    # Aggregate daily electricity transfers.
    daily = pd.DataFrame({

        "import": import_df.sum(
            axis=1,
            min_count=import_df.shape[1]
        ),

        "export": export_df.sum(
            axis=1,
            min_count=export_df.shape[1]
        )
    })

    daily = daily.asfreq("D")

    # Require seven complete days per week.
    weekly = daily.resample("W-SUN").sum(
        min_count=7
    )

    weekly.index = weekly.index.tz_localize(None)
    weekly.index.name = "date"

    return weekly

@st.cache_data(ttl=3600, show_spinner=False)
def load_connection_electricity(
    source,
    destination,
    year
):
    """
    Retrieve daily imports and exports for one connection.

    Data is downloaded directly from ENTSO-E and
    aggregated to daily transferred energy in MWh.
    """

    with ThreadPoolExecutor(max_workers=2) as executor:

        export_task = executor.submit(
            download_entsoe_year,
            source,
            destination,
            year
        )

        import_task = executor.submit(
            download_entsoe_year,
            destination,
            source,
            year
        )

        exports = export_task.result()
        imports = import_task.result()

    df = pd.DataFrame({
        "Import": imports,
        "Export": exports
    })

    if df.empty:
        raise ValueError(
            "No electricity data available "
            "for the selected connection and year."
        )

    df = df.sort_index()

    df.index.name = "date"

    return df