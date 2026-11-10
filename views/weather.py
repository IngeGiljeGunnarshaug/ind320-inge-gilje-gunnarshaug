
import matplotlib.pyplot as plt
import streamlit as st

from weather_utils import (
    load_weather,
    reduce_locations,
    get_weather_summary,
    REGION,
    START_DATE,
    END_DATE
)


st.title("Meteorological Data")

st.write(
    "Historical ERA5 meteorological observations retrieved "
    "from Open-Meteo using fixed sampled locations."
)

st.caption(
    f"Region: {REGION} | Period: {START_DATE} to {END_DATE}"
)

# Load the fixed weather dataset.
try:
    weather = load_weather()
except Exception as error:
    st.error(f"Unable to load weather data: {error}")
    st.stop()

if weather.empty:
    st.warning("No weather observations available.")
    st.stop()

# Calculate spatial averages.
hourly_weather = get_weather_summary(weather)

# Summary statistics.
col1, col2= st.columns(2)

col1.metric(
    "Locations",
    weather["location_id"].nunique()
)

col2.metric(
    "Hourly timestamps",
    len(hourly_weather)
)


# Select which meteorological variable to visualize.
st.subheader("Meteorological Measurements")

plot_choice = st.segmented_control(
    "Select plot",
    options=["Temperature", "Precipitation"],
    default="Temperature"
)

# Configure the selected plot.
if plot_choice == "Temperature":
    values = hourly_weather["temperature_2m"]
    ylabel = "Temperature (°C)"
    title = "Average Temperature"

else:
    values = hourly_weather["precipitation"]
    ylabel = "Precipitation (mm)"
    title = "Average Precipitation"

# Create the plot.
fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(
    hourly_weather.index,
    values,
    linewidth=0.7
)

ax.set_title(title)
ax.set_xlabel("Date")
ax.set_ylabel(ylabel)
ax.grid(alpha=0.3)

fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)


with st.expander("Data source and processing"):
    st.write(
        "The weather data comes from the Open-Meteo "
        "historical archive using the ERA5 model."
    )
    st.write(
        "Sampled locations are loaded from "
        "sampled_locations.csv and reduced using "
        "a 25 km spatial grid."
    )
    st.write(
        "Hourly measurements are averaged across "
        "the selected fixed locations."
    )
    st.write(
        "Completed download batches are cached locally."
    )
