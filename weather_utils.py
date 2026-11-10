from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import time

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
LOCATIONS_FILE = BASE_DIR / "data" / "sampled_locations.csv"
WEATHER_DIR = BASE_DIR / "data" / "weather"

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "rain",
    "snowfall",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "et0_fao_evapotranspiration"
]

# Fixed configuration. Not selectable by app users.
REGION = "NO 3"
START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
BATCH_SIZE = 5
CELL_SIZE = 25000


@st.cache_data
def load_weather_locations():
    """Load the fixed locations from the assignment CSV."""
    locations = pd.read_csv(LOCATIONS_FILE)

    required = {
        "area_type", "region", "latitude", "longitude"
    }
    if not required.issubset(locations.columns):
        raise ValueError("Missing columns in sampled_locations.csv")

    return locations


@st.cache_data
def reduce_locations(region=REGION, cell_size=CELL_SIZE):
    """Keep one sampled location per 25 km grid cell."""
    locations = load_weather_locations()

    subset = locations.loc[
        (locations["area_type"] == "electricity")
        & (locations["region"] == region)
    ].copy()

    if subset.empty:
        raise ValueError(f"No locations found for {region}")

    gdf = gpd.GeoDataFrame(
        subset,
        geometry=gpd.points_from_xy(
            subset["longitude"], subset["latitude"]
        ),
        crs="EPSG:4326"
    ).to_crs("EPSG:3035")

    gdf["grid_x"] = np.floor(
        gdf.geometry.x / cell_size
    ).astype(int)

    gdf["grid_y"] = np.floor(
        gdf.geometry.y / cell_size
    ).astype(int)

    reduced = gdf.drop_duplicates(
        subset=["grid_x", "grid_y"]
    ).copy()

    return reduced.to_crs("EPSG:4326").reset_index(drop=True)


def get_cache_dir(
    points,
    region=REGION,
    start_date=START_DATE,
    end_date=END_DATE,
    batch_size=BATCH_SIZE
):
    """Create a unique cache directory for the request."""
    config = {
        "region": region,
        "start_date": start_date,
        "end_date": end_date,
        "batch_size": batch_size,
        "coordinates": points[
            ["latitude", "longitude"]
        ].values.tolist(),
        "variables": WEATHER_VARIABLES,
        "model": "era5",
        "elevation": "nan",
        "cell_selection": "nearest",
        "timezone": "GMT"
    }

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:12]

    return WEATHER_DIR / (
        f"{region.replace(' ', '')}_{config_hash}"
    )



def download_weather():
    """Download weather data concurrently with disk caching."""

    points = reduce_locations()
    cache_dir = get_cache_dir(points)
    cache_dir.mkdir(parents=True, exist_ok=True)

    batches = [
        (start // BATCH_SIZE, start, points.iloc[start:start + BATCH_SIZE])
        for start in range(0, len(points), BATCH_SIZE)
    ]

    def fetch_batch(batch_info):
        batch_number, start, batch = batch_info
        batch_path = cache_dir / f"batch_{batch_number:04d}.pkl"

        # Reuse previously downloaded batches.
        if batch_path.exists():
            return batch_number, pd.read_pickle(batch_path)

        params = {
            "latitude": ",".join(batch["latitude"].astype(str)),
            "longitude": ",".join(batch["longitude"].astype(str)),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "hourly": ",".join(WEATHER_VARIABLES),
            "models": "era5",
            "timezone": "GMT",
            "cell_selection": "nearest",
            "elevation": ",".join(["nan"] * len(batch))
        }

        for attempt in range(5):

            response = requests.get(
                OPEN_METEO_URL,
                params=params,
                timeout=120
            )

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")

                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = 60
                else:
                    wait = 60 * (attempt + 1)

                if attempt == 4:
                    raise RuntimeError(
                        "Open-Meteo rate limit reached. "
                        "Stop downloading and try again later. "
                        "Previously downloaded batches are cached."
                    )

                print(f"Rate limited. Waiting {wait:.0f} seconds.")
                time.sleep(wait)
                continue

            if response.status_code in (500, 502, 503, 504):
                if attempt == 4:
                    response.raise_for_status()

                time.sleep(2 ** (attempt + 1))
                continue

            response.raise_for_status()
            break

        data = response.json()

        if isinstance(data, dict):
            data = [data]

        if len(data) != len(batch):
            raise ValueError(
                "Unexpected number of locations in API response"
            )

        batch_results = []

        for index, (_, point) in enumerate(batch.iterrows()):
            item = data[index]
            df = pd.DataFrame(item["hourly"])

            df["region"] = REGION
            df["location_id"] = start + index
            df["requested_latitude"] = point["latitude"]
            df["requested_longitude"] = point["longitude"]
            df["grid_latitude"] = item["latitude"]
            df["grid_longitude"] = item["longitude"]

            batch_results.append(df)

        batch_df = pd.concat(batch_results, ignore_index=True)

        # Save completed batches.
        temp_path = batch_path.with_suffix(".tmp")
        batch_df.to_pickle(temp_path)
        temp_path.replace(batch_path)

        return batch_number, batch_df

    results = {}

    # Limit concurrency to avoid excessive API requests.
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {
            executor.submit(fetch_batch, batch): batch[0]
            for batch in batches
        }

        for future in as_completed(futures):
            batch_number, batch_df = future.result()
            results[batch_number] = batch_df

    weather = pd.concat(
        [results[i] for i in sorted(results)],
        ignore_index=True
    )

    weather["time"] = pd.to_datetime(weather["time"])

    return weather.sort_values(
        ["time", "location_id"]
    ).reset_index(drop=True)



@st.cache_data(ttl=3600, show_spinner="Loading weather data...")
def load_weather():
    """Load weather data, downloading missing batches if needed."""
    return download_weather()


def get_weather_summary(weather):
    """Calculate spatial averages for each hourly timestamp."""
    return (
        weather.groupby("time")
        .agg(
            temperature_2m=("temperature_2m", "mean"),
            precipitation=("precipitation", "mean"),
            rain=("rain", "mean"),
            snowfall=("snowfall", "mean"),
            relative_humidity_2m=(
                "relative_humidity_2m", "mean"
            ),
            wind_speed_10m=("wind_speed_10m", "mean"),
            shortwave_radiation=(
                "shortwave_radiation", "mean"
            ),
            et0_fao_evapotranspiration=(
                "et0_fao_evapotranspiration", "mean"
            )
        )
        .sort_index()
    )
