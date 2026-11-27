
import plotly.express as px
import pandas as pd
import streamlit as st
from state_utils import show_current_selection
from data_utils import load_data, MEASUREMENTS


st.title("Reservoir Data")

show_current_selection()

st.write(
    "Explore Norwegian reservoir measurements using "
    "interactive tables and time-series plots."
)

# Select geographical area type.
area_type = st.radio(
    "Area type",
    options=["EL", "NO", "VASS"],
    horizontal=True
)

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

    # Prepare data for Plotly.
    plot_df = df.sort_values("date").copy()

    plot_df["Area"] = (
        "Area " + plot_df["area_number"].astype(int).astype(str)
    )

    # Create an interactive time-series plot.
    fig = px.line(
        plot_df,
        x="date",
        y=measurement,
        color="Area",
        title=f"{MEASUREMENTS[measurement]} - {area_type}",
        labels={
            "date": "Date",
            measurement: MEASUREMENTS[measurement],
            "Area": "Area number"
        },
        color_discrete_sequence=[
            "#007F86",
            "#16324F",
            "#E9A23B",
            "#568CB5",
            "#729B79",
            "#9467BD",
            "#C76B6B"
        ]
    )

    # Customise the chart appearance.
    fig.update_traces(line=dict(width=2))

    fig.update_layout(
        hovermode="x unified",
        height=500,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    # Display the interactive plot.
    st.plotly_chart(
        fig,
        use_container_width=True,
        theme="streamlit"
    )

