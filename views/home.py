import streamlit as st
from datetime import date
from data_utils import load_data, MEASUREMENTS
from state_utils import (
    update_area_type,
    update_selected_area,
    update_date_range,
    show_current_selection
)
from streamlit_folium import st_folium
from map_utils import (
    load_geojson,
    create_map,
    get_clicked_area,
    get_map_values
)

st.title("Norwegian Energy & Reservoir Dashboard")

st.markdown(
    """
    Explore Norwegian reservoir statistics, electricity transfers
    and meteorological data.

    Use the application to investigate historical patterns,
    identify anomalies, analyse relationships between weather
    and reservoir filling, and forecast future filling levels.
    """
)

st.divider()

# Geographical and temporal selection
st.subheader("Geographical Selection")

# Available geographical area types
AREA_TYPES = {
    "EL": "Price areas",
    "VASS": "Water areas",
    "NO": "Norway"
}

# Restore persistent values to temporary widget keys
st.session_state["_area_type"] = st.session_state.area_type

st.session_state["_date_range"] = (
    st.session_state.start_date,
    st.session_state.end_date
)

col1, col2, col3 = st.columns(3)

with st.container(gap=None):
    with st.container(horizontal=True, horizontal_alignment="center"):

        st.segmented_control(
            "Area type",
            options=list(AREA_TYPES),
            format_func=lambda x: AREA_TYPES[x],
            selection_mode="single",
            key="_area_type",
            on_change=update_area_type,
            label_visibility="collapsed"
        )

with col2:
    st.date_input(
        "Historical period",
        min_value=date(1995, 1, 1),
        max_value=date.today(),
        key="_date_range",
        on_change=update_date_range
    )

# Load reservoir data for the selected area type and period
df = load_data(
    st.session_state.start_date,
    st.session_state.end_date,
    st.session_state.area_type
)

# Identify available geographical areas
area_numbers = sorted(
    df["area_number"].dropna().astype(int).unique().tolist()
)

# Validate the current area selection
if st.session_state.selected_area not in area_numbers:
    st.session_state.selected_area = (
        area_numbers[0] if area_numbers else None
    )

show_current_selection()

if df.empty:
    st.warning("No reservoir data for the selected period.")


# Interactive geographical map
if (
    st.session_state.area_type in ["EL", "VASS", "NO"]
    and not df.empty
):

    # Load geographical boundaries.
    gdf = load_geojson(
        st.session_state.area_type
    )

    # Identify available weekly observation dates.
    available_dates = sorted(
        df["date"].dt.date.unique().tolist()
    )

    # Find the available date closest to the shared map date,
    # without selecting a future observation.
    valid_dates = [
        d for d in available_dates
        if d <= st.session_state.map_date
    ]

    default_date = (
        valid_dates[-1]
        if valid_dates
        else available_dates[0]
    )

    # Create controls for map colouring and observation date.
    map_col1, map_col2 = st.columns(2)

    with map_col1:

        measurement = st.selectbox(
            "Colour map by",
            options=[
                column for column in MEASUREMENTS
                if column in df.columns
            ],
            format_func=lambda x: MEASUREMENTS[x],
            key="map_measurement"
        )

    with map_col2:

        map_date = st.select_slider(
            "Map observation date",
            options=available_dates,
            value=default_date,
            format_func=lambda d: d.strftime("%d.%m.%Y"),
            key=(
                f"map_date_{st.session_state.area_type}_"
                f"{st.session_state.start_date}_"
                f"{st.session_state.end_date}"
            )
        )

    # Store the map date for use on other pages.
    st.session_state.map_date = map_date

    # Retrieve measurements for the selected date.
    values = get_map_values(
        df,
        map_date,
        measurement
    )

    # Generate the map with measurement-based colouring.
    m = create_map(
        gdf,
        selected_area=st.session_state.selected_area,
        values=values,
        measurement=measurement,
        measurement_label=MEASUREMENTS[measurement]
    )

    map_data = st_folium(
        m,
        height=550,
        use_container_width=True,
        returned_objects=["last_clicked"],
        key=f"area_map_{st.session_state.area_type}"
    )

    clicked = map_data.get("last_clicked")

    if clicked is not None:

        click_position = (
            clicked["lat"],
            clicked["lng"]
        )

        last_click_key = (
            f"last_map_click_{st.session_state.area_type}"
        )

        # Process each new click only once.
        if click_position != st.session_state.get(last_click_key):

            st.session_state[last_click_key] = click_position

            clicked_area = get_clicked_area(
                gdf,
                clicked["lat"],
                clicked["lng"]
            )

            if (
                clicked_area is not None
                and clicked_area in area_numbers
                and clicked_area != st.session_state.selected_area
            ):

                st.session_state.selected_area = clicked_area

                st.rerun()

st.divider()

# Navigation overview
st.subheader("Explore the application")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Data Exploration")
    st.write(
        "Explore historical reservoir statistics "
        "and electricity transfers."
    )

    st.page_link(
        "views/reservoir.py",
        label="Reservoir Data",
        icon=":material/database:"
    )

    st.page_link(
        "views/electricity.py",
        label="Electricity",
        icon=":material/bolt:"
    )

with col2:
    st.markdown("### Statistical Analysis")
    st.write(
        "Investigate seasonality, anomalies and "
        "relationships between variables."
    )

    st.page_link(
        "views/stl_spectrogram.py",
        label="STL & Spectrogram",
        icon=":material/show_chart:"
    )

    st.page_link(
        "views/outlier_anomalies.py",
        label="Outliers & Anomalies",
        icon=":material/warning:"
    )

    st.page_link(
        "views/correlation.py",
        label="Weather & Reservoir Correlation",
        icon=":material/scatter_plot:"
    )

with col3:
    st.markdown("### Predictive Analysis")
    st.write(
        "Forecast reservoir filling using statistical "
        "and machine learning models."
    )

    st.page_link(
        "views/forecasting.py",
        label="Reservoir Fill Fraction Forecasting",
        icon=":material/trending_up:"
    )

st.divider()

st.caption("Developed as part of the IND320 course at NMBU.")
