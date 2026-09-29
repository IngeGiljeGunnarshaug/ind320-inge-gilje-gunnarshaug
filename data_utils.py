from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_data():
    """Load, filter and prepare the reservoir data."""
    # Locate the CSV relative to this file.
    path = Path(__file__).parent / "data" / "reservoirs.csv"
    df = pd.read_csv(path)

    # Keep only the national series: area type NO, area number 0.
    df = df.loc[
        (df["omrType"] == "NO") & (df["omrnr"] == 0)
    ].copy()

    # Rename all columns to English.
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

    # Convert observation dates and sort from oldest to newest.
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    return df