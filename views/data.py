import streamlit as st

from data_utils import load_data


st.title("First-Month Summary")

# Load the prepared national series and select its first calendar month.
df = load_data()
months = df["date"].dt.to_period("M")
first_month = months.min()
month_df = df.loc[months == first_month]

# Map the requested measurements to readable names.
names = {
    "fill_fraction": "Fill ratio",
    "capacity_twh": "Capacity (TWh)",
    "stored_energy_twh": "Stored energy (TWh)",
    "previous_week_fill_fraction": "Previous week's fill ratio",
    "fill_fraction_change": "Fill ratio change",
}

# Each row contains a measurement name and its chronological monthly values.
summary = {
    "Measurement": list(names.values()),
    "Summary": [month_df[column].tolist() for column in names],
}

st.caption(f"Month: {first_month}")

st.dataframe(
    summary,
    column_config={
        "Summary": st.column_config.LineChartColumn(
            "Summary",
        ),
    },
    hide_index=True,
)