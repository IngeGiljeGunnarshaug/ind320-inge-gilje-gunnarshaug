import calendar

import plotly.express as px
import pandas as pd
import streamlit as st

from mongo_utils import (
    get_connections,
    get_years,
    load_entsoe
)

# Consistent colours for electricity transfer directions.
DIRECTION_COLORS = {
    "Import": "#007F86",
    "Export": "#16324F"
}

st.title("ENTSO-E Electricity Transfers")

st.write(
    "Explore electricity imports and exports between Norway "
    "and neighbouring electricity markets."
)

# Establish two columns for the annual and monthly plots.
left, right = st.columns(2, gap="large")

# Retrieve the available connections from MongoDB.
try:
    connections = get_connections()
except Exception as error:
    st.error(f"MongoDB connection failed: {error}")
    st.stop()

if not connections:
    st.warning("No ENTSO-E data found in MongoDB.")
    st.stop()


# LEFT COLUMN: annual import/export pie chart.
with left:

    st.subheader("Annual Import and Export")

    connection = st.radio(
        "Connection",
        options=connections
    )

    years = get_years(connection)

    if not years:
        st.warning("No years available for this connection.")
        st.stop()

    year = st.selectbox(
        "Year",
        options=years
    )

    # Retrieve all months for the annual comparison.
    annual_df = load_entsoe(connection, year)

    if not annual_df.empty:

        # Sum transferred energy separately for import and export.
        annual_summary = (
            annual_df
            .groupby("direction")["mwh"]
            .sum()
        )

        if annual_summary.sum() > 0:

            # Create an interactive annual pie chart.
            fig = px.pie(
                names=annual_summary.index.str.title(),
                values=annual_summary.values,
                title=f"{connection} - {year}",
                color=annual_summary.index.str.title(),
                color_discrete_map=DIRECTION_COLORS,
                hole=0.4
            )

            # Customise labels and hover information.
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                hovertemplate=(
                    "%{label}<br>"
                    "Total energy: %{value:,.0f} MWh<br>"
                    "Share: %{percent}"
                    "<extra></extra>"
                )
            )

            fig.update_layout(
                height=450,
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.05,
                    xanchor="center",
                    x=0.5
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme="streamlit",
                key="annual_electricity_chart"
            )


            # Display the underlying annual totals.
            st.dataframe(
                annual_summary.rename(
                    "Total energy (MWh)"
                ),
                use_container_width=True
            )

        else:
            st.warning("No positive energy totals available.")

    else:
        st.warning("No annual data available.")


# RIGHT COLUMN: monthly import/export line plot.
with right:

    st.subheader("Monthly Electricity Transfers")

    # Allow multiple months to be selected.
    months = st.pills(
        "Select months",
        options=list(range(1, 13)),
        default=[1],
        selection_mode="multi",
        format_func=lambda m: calendar.month_name[m]
    )

    if months:

        # Retrieve only the selected months from MongoDB.
        monthly_df = load_entsoe(
            connection,
            year,
            months=months
        )

        if not monthly_df.empty:

            # Aggregate duplicate timestamps within each direction.
            monthly_summary = (
                monthly_df
                .groupby(["time", "direction"], as_index=False)
                ["mwh"]
                .sum()
            )

            # Prepare data for Plotly.
            plot_df = monthly_summary.sort_values("time").copy()

            plot_df["direction"] = (
                plot_df["direction"].str.title()
            )

            # Create an interactive electricity transfer plot.
            fig = px.line(
                plot_df,
                x="time",
                y="mwh",
                color="direction",
                title=f"{connection} - {year}",
                color_discrete_map=DIRECTION_COLORS,
                labels={
                    "time": "Date",
                    "mwh": "Electricity transfer (MWh)",
                    "direction": "Direction"
                }
            )

            fig.update_traces(
                line=dict(width=1.5)
            )

            fig.update_layout(
                height=450,
                hovermode="x unified",
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.25,
                    xanchor="center",
                    x=0.5
                )
            )

            # Add an interactive date range slider.
            fig.update_xaxes(
                rangeslider_visible=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                theme="streamlit",
                key="monthly_electricity_chart"
            )


        else:
            st.warning("No observations for the selected months.")

    else:
        st.info("Select at least one month.")


# Document the source and processing pipeline.
with st.expander("Data source"):

    st.write(
        "The electricity transfer data originates from the "
        "ENTSO-E Transparency Platform API."
    )

    st.write(
        "Data was downloaded and processed in a Jupyter Notebook, "
        "stored in Cassandra using Spark, "
        "extracted and prepared using Spark, and then "
        "uploaded to MongoDB."
    )

    st.write(
        "This Streamlit page retrieves the prepared observations "
        "directly from MongoDB."
    )

    st.write(
        "The annual pie chart compares total imported and "
        "exported electricity for the selected connection and year."
    )

    st.write(
        "The line plot displays electricity transfers over time "
        "for the selected months, with separate lines for "
        "import and export."
    )

    st.write(
        "Electricity transfers are measured in MWh."
    )