import streamlit as st
import geopandas as gpd
import folium
from pathlib import Path
from shapely.geometry import Point
import pandas as pd
import numpy as np
from branca.colormap import LinearColormap
from data_utils import load_weather, WEATHER_AGGREGATIONS


DATA_DIR = Path(__file__).resolve().parent / "data"


@st.cache_data
def load_geojson(area_type):
    """Load geographical boundaries for the selected area type."""

    if area_type == "EL":

        gdf = gpd.read_file(
            DATA_DIR / "electricity.geojson"
        )

        # Convert "NO 1" to the numerical area ID 1.
        gdf["area_number"] = (
            gdf["budomr"]
            .str.extract(r"(\d+)", expand=False)
            .astype(int)
        )

        gdf["area_name"] = (
            gdf["budomr"].str.replace(" ", "")
        )

    elif area_type == "VASS":

        gdf = gpd.read_file(
            DATA_DIR / "water.geojson"
        )

        gdf["area_number"] = (
            gdf["area_number"].astype(int)
        )

    elif area_type == "NO":

        gdf = gpd.read_file(
            DATA_DIR / "electricity.geojson"
        )

        # Merge the five price areas into one national area.
        gdf = gdf[["geometry"]].dissolve()

        # Remove small disconnected polygons.
        gdf = gdf.to_crs(epsg=25833)
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[gdf.geometry.area >= 5_000_000]

        # Merge the remaining polygons.
        gdf = gdf.dissolve()
        gdf = gdf.to_crs(epsg=4326)

        gdf["area_number"] = 0
        gdf["area_name"] = "Norway"

    else:
        raise ValueError(
            f"Unsupported area type: {area_type}"
        )

    return gdf.to_crs(epsg=4326)

def get_map_values(df, map_date, measurement):
    """Get the latest available measurement for each area."""

    # Only use observations available at the selected date.
    observations = df.loc[
        df["date"] <= pd.Timestamp(map_date)
    ].copy()

    # Select the most recent observation for each area.
    observations = (
        observations
        .sort_values("date")
        .drop_duplicates("area_number", keep="last")
    )

    # Return a dictionary mapping area numbers to values.
    return observations.set_index(
        "area_number"
    )[measurement].to_dict()

def create_map(
    gdf,
    selected_area=None,
    values=None,
    measurement="fill_fraction",
    measurement_label="Fill ratio"
):
    """Create an interactive map with data-driven colouring."""

    gdf = gdf.copy()

    if values is None:
        values = {}

    # Connect each geographical area to its measurement.
    gdf["value"] = gdf["area_number"].map(values)

    valid_values = gdf["value"].dropna()

    # Use a fixed scale for reservoir fill fractions.
    # Other measurements use the observed value range.
    if measurement == "fill_fraction":
        vmin, vmax = 0, 1

    elif not valid_values.empty:
        vmin = float(valid_values.min())
        vmax = float(valid_values.max())

        # Avoid a colour scale with zero range.
        if vmin == vmax:
            vmax = vmin + 1

    else:
        vmin, vmax = 0, 1

    colormap = LinearColormap(
        colors=[
            "#E9A23B",
            "#B1D6CC",
            "#007F86"
        ],
        vmin=vmin,
        vmax=vmax,
        caption=measurement_label
    )

    # Format measurement values for the map tooltip.
    def format_value(value):

        if pd.isna(value):
            return "No data"

        if measurement == "fill_fraction":
            return f"{value:.1%}"

        return f"{value:.2f}"

    gdf["display_value"] = (
        gdf["value"].apply(format_value)
    )

    # Initialize the map.
    m = folium.Map(
        location=[65, 15],
        zoom_start=4,
        tiles=None,
        zoom_control=False,
        dragging=False,
        scrollWheelZoom=False,
        doubleClickZoom=False,
        touchZoom=False,
        boxZoom=False,
        keyboard=False
    )

    # Remove browser focus outlines around clicked polygons.
    m.get_root().header.add_child(
        folium.Element("""
            <style>
                .leaflet-interactive:focus {
                    outline: none !important;
                }
            </style>
        """)
    )

    def style_function(feature):

        properties = feature["properties"]

        area_number = properties["area_number"]
        value = properties["value"]

        is_selected = area_number == selected_area

        # Missing measurements are displayed in grey.
        fill_color = (
            colormap(value)
            if pd.notna(value)
            else "#BDBDBD"
        )

        return {
            "fillColor": fill_color,
            "fillOpacity": 0.55,
            "color": (
                "#16324F" if is_selected
                else "#007F86"
            ),
            "weight": 3.5 if is_selected else 1.2,
            "opacity": 1
        }

    # Add geographical polygons and measurement tooltips.
    folium.GeoJson(
        gdf.to_json(),
        name="Geographical areas",

        style_function=style_function,

        highlight_function=lambda feature: {
            "fillOpacity": 0.8
        },

        tooltip=folium.GeoJsonTooltip(
            fields=["area_name", "display_value"],
            aliases=["Area:", f"{measurement_label}:"]
        )

    ).add_to(m)

    # Display the measurement colour scale.
    colormap.add_to(m)

    return m

def get_clicked_area(gdf, lat, lon):
    """Return the area number at the clicked location."""

    point = Point(lon, lat)

    clicked = gdf[gdf.geometry.covers(point)]

    if clicked.empty:
        return None

    return int(clicked.iloc[0]["area_number"])

@st.cache_data
def get_weather_locations(area_type, area_number, n_locations=5):

    locations = pd.read_csv(
        DATA_DIR / "sampled_locations.csv"
    )

    vass_regions = {
        1: ["RØ", "RS"],
        2: ["RV"],
        3: ["RM", "RN"]
    }

    if area_type == "EL":

        points = locations.loc[
            (locations["area_type"] == "electricity") &
            (locations["region"] == f"NO {area_number}")
        ]

    elif area_type == "VASS":

        points = locations.loc[
            (locations["area_type"] == "precipitation") &
            (locations["region"].isin(vass_regions[area_number]))
        ]

    elif area_type == "NO":

        points = locations.loc[
            locations["area_type"] == "electricity"
        ]

    else:
        raise ValueError(f"Unknown area type: {area_type}")

    if points.empty:
        raise ValueError("No weather locations found")

    # Select one location per approximate 25 km grid cell.
    gdf = gpd.GeoDataFrame(
        points,
        geometry=gpd.points_from_xy(
            points["longitude"],
            points["latitude"]
        ),
        crs="EPSG:4326"
    ).to_crs("EPSG:3035")

    gdf["grid_x"] = np.floor(gdf.geometry.x / 25000)
    gdf["grid_y"] = np.floor(gdf.geometry.y / 25000)

    gdf = gdf.drop_duplicates(
        subset=["grid_x", "grid_y"]
    )

    # Select reproducible sampling locations.
    points = gdf.sample(
        n=min(n_locations, len(gdf)),
        random_state=42
    )

    return points[["latitude", "longitude"]].reset_index(drop=True)

@st.cache_data(ttl=3600, show_spinner="Downloading weather data...")
def get_weather_map_values(
    area_type,
    area_numbers,
    map_date,
    measurement,
    n_locations=5
):
    """Calculate weekly weather values for each geographical area."""

    end_date = pd.Timestamp(map_date)
    start_date = end_date - pd.Timedelta(days=6)

    values = {}

    for area_number in area_numbers:

        points = get_weather_locations(
            area_type,
            area_number,
            n_locations
        )

        weather = load_weather(
            points,
            start_date,
            end_date,
            [measurement]
        )

        aggregation = WEATHER_AGGREGATIONS[measurement]

        daily_values = weather[measurement].dropna()

        # Require observations for all seven days.
        if len(daily_values) != 7:
            values[area_number] = np.nan

        elif aggregation == "sum":
            values[area_number] = daily_values.sum()

        else:
            values[area_number] = daily_values.mean()

    return values