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


@st.cache_data(ttl=3600, show_spinner="Downloading NVE data...")
def load_data(start_date, end_date, area_type):
    """
    Download reservoir data from NVE for the selected interval
    and area type, then prepare it for Streamlit.
    """

    # Retrieve observations directly from the NVE API.
    response = requests.get(
        NVE_URL,
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