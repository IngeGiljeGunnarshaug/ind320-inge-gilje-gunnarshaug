from pathlib import Path

import pandas as pd
import streamlit as st


# Move from views/data.py to the project root, then locate the CSV.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "reservoirs.csv"


@st.cache_data
def load_data():
    """Read the CSV once and reuse the result during subsequent reruns."""
    return pd.read_csv(DATA_PATH)


st.title("Reservoir Data")

# Stop with an understandable message if the file is missing.
if not DATA_PATH.is_file():
    st.error("CSV not found. Place reservoirs.csv in the project's data folder.")
    st.stop()

df = load_data()

st.write(f"The dataset contains **{len(df):,} rows** and **{len(df.columns)} columns**.")

# Display the imported data as a scrollable table.
st.dataframe(df, hide_index=True)
