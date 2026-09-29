import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from data_utils import load_data


st.title("Reservoir Data Plots")

df = load_data()

# Use the same five measurements containing informative data
names = {
    "fill_fraction": "Fill ratio",
    "capacity_twh": "Capacity (TWh)",
    "stored_energy_twh": "Stored energy (TWh)",
    "previous_week_fill_fraction": "Previous week's fill ratio",
    "fill_fraction_change": "Fill ratio change",
}

selection = st.selectbox(
    "Measurement",
    ["All columns"] + list(names),
    format_func=lambda column: names.get(column, column),
)

# Select an inclusive month range, initially containing only the first month.
months = df["date"].dt.to_period("M")
options = sorted(months.astype(str).unique())

start, end = st.select_slider(
    "Months",
    options=options,
    value=(options[0], options[0]),
)

filtered = df.loc[
    (months >= pd.Period(start, freq="M"))
    & (months <= pd.Period(end, freq="M"))
]

columns = list(names) if selection == "All columns" else [selection]

fig, ax = plt.subplots(figsize=(10, 5))

if selection == "All columns":
    # Standardise using the full history so changing months keeps the same scale.
    values = df[list(names)]
    means = values.mean()
    stds = values.std()

    # Constant columns cannot be standardised; display them at zero.
    constant = values.nunique().eq(1)
    scaled = (values - means) / stds.mask(constant)

    for column in names:
        label = names[column]

        if constant[column]:
            scaled[column] = values[column].where(values[column].isna(), 0)
            label += " (constant; shown at 0)"

        ax.plot(
            filtered["date"],
            scaled.loc[filtered.index, column],
            label=label,
            linewidth=1.2,
        )

    ax.set_title("Reservoir Measurements — Standardised")
    ax.set_ylabel("Standard deviations from the mean")
    ax.legend()

else:
    # Show a single measurement in its original units.
    ax.plot(filtered["date"], filtered[selection], linewidth=1.2)
    ax.set_title(names[selection])
    ax.set_ylabel(names[selection])

ax.set_xlabel("Date")
ax.grid(alpha=0.3)
fig.autofmt_xdate()
fig.tight_layout()

st.pyplot(fig)
plt.close(fig)