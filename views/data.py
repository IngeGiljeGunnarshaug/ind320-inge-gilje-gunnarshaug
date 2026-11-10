
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_utils import load_data, MEASUREMENTS


st.title("Reservoir Data")

st.write(
    "Explore Norwegian reservoir measurements using "
    "interactive tables and time-series plots."
)

# Use the global geographical area selection.
area_type = st.session_state["global_area_type"]

st.caption(f"Selected area type: {area_type}")

# Select the measurement.
measurement = st.selectbox(
    "Measurement",
    options=list(MEASUREMENTS),
    format_func=lambda x: MEASUREMENTS[x]
)

# Select the date interval using a monthly slider.
start_month, end_month = st.select_slider(
    "Date interval",
    options=pd.period_range(
        "2020-01",
        pd.Timestamp.today().to_period("M"),
        freq="M"
    ).astype(str).tolist(),
    value=(
        "2025-01",
        pd.Timestamp.today().strftime("%Y-%m")
    )
)

# Convert selected months to dates.
start_date = pd.Period(start_month).start_time.date()
end_date = pd.Period(end_month).end_time.date()

# Retrieve and prepare NVE observations.
try:
    df = load_data(start_date, end_date, area_type)
except Exception as error:
    st.error(f"Unable to retrieve NVE data: {error}")
    st.stop()

if df.empty:
    st.warning("No observations found for this selection.")
    st.stop()

# Create tabs for the table and plots.
tab_table, tab_plots = st.tabs(["Data Table", "Plots"])


# TAB 1: DATA TABLE
with tab_table:

    st.subheader("Area Summary")

    summary = []

    for area_number, group in df.groupby("area_number"):
        group = group.sort_values("date")

        summary.append({
            "Area number": int(area_number),
            "Observations": len(group),
            "Latest value": group[measurement].iloc[-1],
            "Trend": group[measurement].tolist()
        })

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


# TAB 2: PLOTS
with tab_plots:

    st.subheader("Reservoir Time Series")

    fig, ax = plt.subplots(figsize=(11, 5))

    for area_number, group in df.groupby("area_number"):
        group = group.sort_values("date")

        ax.plot(
            group["date"],
            group[measurement],
            label=f"Area {int(area_number)}",
            linewidth=1.5
        )

    ax.set_title(
        f"{MEASUREMENTS[measurement]} - {area_type}"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(MEASUREMENTS[measurement])
    ax.grid(alpha=0.3)
    ax.legend(title="Area number")

    fig.autofmt_xdate()
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)
