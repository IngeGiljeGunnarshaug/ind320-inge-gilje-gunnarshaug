import streamlit as st
from datetime import date


def initialize_session_state():
    """Initialize shared selections if they do not already exist."""

    defaults = {
        "area_type": "EL",
        "selected_area": 3,
        "start_date": date(2020, 1, 1),
        "end_date": date(2024, 12, 31),
        "map_date": date(2024, 12, 31),
        "forecast_horizon": 12
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def update_area_type():
    """Update the geographical area type."""

    area_type = st.session_state["_area_type"]

    if area_type is None:
        st.session_state["_area_type"] = (
            st.session_state.area_type
        )
        return

    if area_type != st.session_state.area_type:
        st.session_state.area_type = area_type
        st.session_state.selected_area = None


def update_selected_area():
    """Store the selected geographical area."""

    st.session_state.selected_area = (
        st.session_state["_selected_area"]
    )


def update_date_range():
    """Update the shared historical period and map date."""

    dates = st.session_state["_date_range"]

    if len(dates) == 2:
        start_date, end_date = dates

        st.session_state.start_date = start_date
        st.session_state.end_date = end_date
        st.session_state.map_date = end_date


def show_current_selection():
    """Display the shared geographical and temporal selection."""

    area_type = st.session_state.area_type
    area = st.session_state.selected_area

    if area is None:
        area_label = "No area selected"
    elif area_type == "EL":
        area_label = f"NO{area}"
    else:
        area_label = f"{area_type} {area}"

    start = st.session_state.start_date
    end = st.session_state.end_date

    st.caption(
        f"Selected area: {area_label} | "
        f"Period: {start:%d.%m.%Y} – {end:%d.%m.%Y}"
    )
