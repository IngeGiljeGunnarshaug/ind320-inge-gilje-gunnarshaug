import pandas as pd
import streamlit as st
from pymongo import MongoClient


@st.cache_resource
def get_collection():
    """Connect to MongoDB Atlas and return the ENTSO-E collection."""

    client = MongoClient(
        f"mongodb+srv://{st.secrets['MONGODB_HOST']}/",
        username=st.secrets["MONGODB_USER"],
        password=st.secrets["MONGODB_PASSWORD"],
        authSource="admin",
        serverSelectionTimeoutMS=10000
    )

    # Verify that the connection works.
    client.admin.command("ping")

    return client["ind320"]["entsoe_flows"]


@st.cache_data(ttl=3600)
def get_connections():
    """Retrieve the available cross-border connections."""
    collection = get_collection()
    return sorted(collection.distinct("connection"))


@st.cache_data(ttl=3600)
def get_years(connection):
    """Retrieve available years for the selected connection."""
    collection = get_collection()

    years = collection.distinct(
        "year",
        {"connection": connection}
    )

    return sorted(
        [int(year) for year in years if year is not None],
        reverse=True
    )


@st.cache_data(ttl=3600)
def load_entsoe(connection, year, months=None):
    """
    Retrieve electricity transfers from MongoDB for a
    connection, year and optional selection of months.
    """

    collection = get_collection()

    query = {
        "connection": connection,
        "year": int(year)
    }

    # Restrict the query to the selected months if provided.
    if months is not None:
        query["month"] = {"$in": [int(m) for m in months]}

    # Exclude MongoDB's internal document identifier.
    cursor = collection.find(
        query,
        {"_id": 0}
    )

    df = pd.DataFrame(list(cursor))

    if df.empty:
        return df

    # Prepare timestamps and numerical values for plotting.
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df["mwh"] = pd.to_numeric(df["mwh"], errors="coerce")

    return df.sort_values("time").reset_index(drop=True)