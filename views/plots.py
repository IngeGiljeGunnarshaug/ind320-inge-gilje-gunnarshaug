import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_utils import load_data, MEASUREMENTS


st.title("Reservoir Plots")

st.write(
    "Select a geographical area type, measurement and interval "
    "to compare reservoir observations."
)

# Choose the geographical area type.
area_type = st.radio(
    "Area type",
    ["EL", "NO", "VASS"],
    horizontal=True
)

# Choose the measurement to visualize.
measurement = st.selectbox(
    "Measurement",
    options=list(MEASUREMENTS),
    format_func=lambda x: MEASUREMENTS[x]
)

# The slider controls the interval used when requesting data.
start_month, end_month = st.select_slider(
    "Download interval",
    options=pd.period_range(
        "2024-01",
        pd.Timestamp.today().to_period("M"),
        freq="M"
    ).astype(str).tolist(),
    value=(
        "2025-01",
        pd.Timestamp.today().strftime("%Y-%m")
    )
)

start_date = pd.Period(start_month).start_time.date()
end_date = pd.Period(end_month).end_time.date()

# Download the selected reservoir observations.
try:
    df = load_data(start_date, end_date, area_type)
except Exception as error:
    st.error(f"Unable to retrieve NVE data: {error}")
    st.stop()

if df.empty:
    st.warning("No observations found for this selection.")
    st.stop()

# Create one line per area number.
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