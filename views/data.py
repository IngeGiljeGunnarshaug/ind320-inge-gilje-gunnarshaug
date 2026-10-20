import streamlit as st
import pandas as pd

from data_utils import load_data, MEASUREMENTS


st.title("Reservoir Data")

st.write(
    "Compare reservoir measurements across geographical areas "
    "using data retrieved directly from NVE."
)

# Select the geographical area type.
area_type = st.radio(
    "Area type",
    ["EL", "NO", "VASS"],
    horizontal=True
)

# Select the period to display.
start_date, end_date = st.date_input(
    "Date interval",
    value=(
        pd.Timestamp("2025-01-01").date(),
        pd.Timestamp.today().date()
    )
)

if start_date > end_date:
    st.error("The start date must be before the end date.")
    st.stop()

# Retrieve and prepare NVE observations.
try:
    df = load_data(start_date, end_date, area_type)
except Exception as error:
    st.error(f"Unable to retrieve NVE data: {error}")
    st.stop()

if df.empty:
    st.warning("No observations found for this selection.")
    st.stop()

# Select the measurement used in the summary table.
measurement = st.selectbox(
    "Measurement",
    options=list(MEASUREMENTS),
    format_func=lambda x: MEASUREMENTS[x]
)

# Display a separate trend for every area number.
summary = []

for area_number, group in df.groupby("area_number"):
    group = group.sort_values("date")

    summary.append({
        "Area number": int(area_number),
        "Observations": len(group),
        "Latest value": group[measurement].iloc[-1],
        "Trend": group[measurement].tolist()
    })

st.subheader("Area Summary")

st.dataframe(
    pd.DataFrame(summary),
    column_config={
        "Trend": st.column_config.LineChartColumn(
            "Trend"
        ),
        "Latest value": st.column_config.NumberColumn(
            "Latest value",
            format="%.3f"
        )
    },
    hide_index=True,
    use_container_width=True
)

# Allow inspection of the underlying observations.
with st.expander("View raw data"):
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )