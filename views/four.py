import calendar

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from mongo_utils import (
    get_connections,
    get_years,
    load_entsoe
)


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

            fig, ax = plt.subplots(figsize=(6, 5))

            ax.pie(
                annual_summary.values,
                labels=annual_summary.index,
                autopct="%1.1f%%",
                startangle=90
            )

            ax.set_title(
                f"{connection} - {year}"
            )

            st.pyplot(fig)
            plt.close(fig)

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

            fig, ax = plt.subplots(figsize=(8, 5))

            # Draw one line for each direction.
            for direction, group in monthly_summary.groupby(
                "direction"
            ):

                group = group.sort_values("time")

                ax.plot(
                    group["time"],
                    group["mwh"],
                    label=direction,
                    linewidth=1.2
                )

            ax.set_title(
                f"{connection} - {year}"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Electricity transfer (MWh)")
            ax.grid(alpha=0.3)
            ax.legend(title="Direction")

            fig.autofmt_xdate()
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

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