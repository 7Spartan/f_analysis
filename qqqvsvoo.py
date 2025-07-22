import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Monthly return data (shortened here for brevity — replace with full list)
qqq_returns = [
    -6.47, 4.60, 7.71, 2.24, -7.39, -5.98, 7.26, -5.13, 13.17, 6.34, -0.17, 4.76,   # 2010
    2.83, 3.16, -0.45, 2.87, -1.22, -2.02, 1.67, -5.07, -4.49, 10.40, -2.69, -0.62, # 2011
    8.42, 6.41, 5.05, -1.17, -7.04, 3.62, 1.00, 5.19, 0.89, -5.28, 1.31, -0.46,     # 2012
    2.67, 0.34, 3.03, 2.54, 3.58, -2.39, 6.31, -0.40, 4.83, 4.96, 3.55, 2.92,       # 2013
    -1.92, 5.15, -2.73, -0.32, 4.49, 3.12, 1.18, 5.01, -0.76, 2.64, 4.55, -2.24,    # 2014
    -2.08, 7.22, -2.36, 1.92, 2.25, -2.48, 4.56, -6.82, -2.20, 11.37, 0.61, -1.59,  # 2015
    -6.91, -1.57, 6.85, -3.19, 4.37, -2.27, 7.15, 1.05, 2.21, -1.46, 0.44, 1.13,    # 2016
    5.14, 4.38, 2.03, 2.73, 3.90, -2.32, 4.06, 2.07, -0.29, 4.61, 1.97, 0.60,       # 2017
    8.76, -1.29, -4.08, 0.51, 5.67, 1.15, 2.80, 5.78, -0.28, -8.60, -0.26, -8.65,   # 2018
    9.01, 2.99, 3.92, 5.50, -8.23, 7.59, 2.33, -1.90, 0.92, 4.38, 4.07, 3.89,       # 2019
    3.04, -6.06, -7.29, 14.97, 6.60, 6.28, 7.35, 10.94, -5.78, -3.04, 11.23, 4.90,  # 2020
    0.26, -0.13, 1.71, 5.91, -1.20, 6.26, 2.86, 4.22, -5.68, 7.86, 2.00, 1.15,      # 2021
    -8.75, -4.48, 4.67, -13.60, -1.59, -8.91, 12.55, -5.13, -10.54, 4.00, 5.54, -9.01, # 2022
    10.64, -0.36, 9.49, 0.51, 7.88, 6.30, 3.86, -1.48, -5.08, -2.07, 10.82, 5.59,   # 2023
    1.82, 5.28, 1.27, -4.37, 6.15, 6.47, -1.68, 1.10, 2.62, -0.86, 5.35, 0.45,      # 2024
    2.16, -2.70, -7.59, 1.40, 9.18, 6.39                                           # 2025 YTD (Jan–Jun)
]

voo_returns = [
    -3.63, 3.12, 6.09, 1.55, -7.95, -5.17, 6.83, -4.50, 8.96, 3.82, 0.00, 6.68,     # 2010
    2.22, 3.47, 0.00, 2.94, -1.17, -1.68, -2.06, -5.52, -6.79, 10.74, -0.30, 1.12,  # 2011
    4.47, 4.28, 3.30, -0.64, -6.00, 4.14, 1.20, 2.51, 2.55, -1.96, 0.57, 1.02,      # 2012
    5.17, 1.33, 3.61, 2.09, 2.32, -1.50, 5.30, -3.08, 3.39, 4.47, 3.00, 2.64,       # 2013
    -3.53, 4.57, 0.88, 0.72, 2.29, 2.09, -1.38, 3.97, -1.38, 2.40, 2.76, -0.32,     # 2014
    -2.87, 5.58, -1.57, 1.00, 1.25, -1.95, 2.18, -6.14, -2.47, 8.45, 0.43, -1.74,   # 2015
    -4.91, -0.21, 6.87, 0.35, 1.75, 0.32, 3.68, 0.12, 0.04, -1.79, 3.73, 2.07,      # 2016
    1.78, 3.88, 0.13, 1.04, 1.40, 0.63, 2.06, 0.29, 2.04, 2.33, 3.06, 1.28,         # 2017
    5.59, -3.73, -2.48, 0.35, 2.42, 0.76, 3.56, 3.22, 0.58, -6.84, 1.89, -8.84,     # 2018
    7.92, 3.25, 1.92, 4.03, -6.35, 6.99, 1.46, -1.64, 1.97, 2.18, 3.63, 2.97,       # 2019
    -0.04, -8.10, -12.46, 12.79, 4.74, 1.83, 5.88, 6.97, -3.75, -2.55, 10.95, 3.75, # 2020
    -1.02, 2.77, 4.57, 5.29, 0.67, 2.26, 2.45, 2.95, -4.66, 7.04, -0.73, 4.55,      # 2021
    -5.24, -2.98, 3.78, -8.78, 0.26, -8.26, 9.20, -4.13, -9.21, 8.12, 5.51, -5.73,  # 2022
    6.29, -2.50, 3.71, 1.59, 0.48, 6.51, 3.29, -1.63, -4.74, -2.17, 9.17, 4.58,     # 2023
    1.61, 5.21, 3.28, -4.01, 5.03, 3.57, 1.16, 2.39, 2.18, -0.95, 5.89, -2.33,      # 2024
    2.69, -1.27, -5.61, -0.81, 6.28, 5.17                                           # 2025 YTD (Jan–Jun)
]

# Create date range
dates = pd.date_range(start="2010-01-01", periods=len(voo_returns), freq='MS')
df_voo = pd.DataFrame({"Date": dates, "Return": voo_returns})
df_qqq = pd.DataFrame({"Date": dates, "Return": qqq_returns})

# Streamlit UI
st.title("📈 VOO & QQQ Investment Simulator")

etf_choice = st.selectbox("Choose ETF:", ["VOO", "QQQ"])
holding_years = st.slider("Holding Period (Years)", 1, 15, 5)
initial_investment = st.number_input("Initial Investment ($)", value=200_000)
monthly_contribution = st.number_input("Monthly Contribution ($)", value=0)

df = df_voo if etf_choice == "VOO" else df_qqq

# Simulation
trajectories = []
monthly_periods = holding_years * 12


for start_idx in range(len(df) - monthly_periods):
    investment = initial_investment
    trajectory = [investment]
    time_labels = [df.loc[start_idx, "Date"]]

    for i in range(1, monthly_periods + 1):
        ret = df.loc[start_idx + i, "Return"]
        investment = investment * (1 + ret/100) + monthly_contribution
        trajectory.append(investment)
        time_labels.append(df.loc[start_idx + i, "Date"])

    years = len(trajectory) / 12
    cagr = (trajectory[-1] / initial_investment) ** (1 / years) - 1
    trajectories.append({
        "Start": df.loc[start_idx, "Date"],
        "X": time_labels,
        "Y": trajectory,
        "Final": trajectory[-1],
        "CAGR": cagr
    })

# Sort and pick best, worst, median
trajectories.sort(key=lambda x: x["CAGR"])
median_idx = len(trajectories) // 2
best = trajectories[-1]
worst = trajectories[0]
median = trajectories[median_idx]

# Plot trajectories
fig, ax = plt.subplots(figsize=(10, 5))
for t, label, color in zip([best, median, worst], ["Best", "Median", "Worst"], ["green", "blue", "red"]):
    ax.plot(t["X"], t["Y"], label=f"{label} Start: {t['Start'].strftime('%Y-%m')}\nCAGR: {t['CAGR']*100:.2f}%\nFinal: ${t['Final']:,.0f}", color=color)

ax.set_title(f"Growth of ${initial_investment:,.0f} in {etf_choice} over {holding_years} years")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value ($)")
ax.legend()
st.pyplot(fig)

# Histogram of final values
fig2, ax2 = plt.subplots()
final_vals = [t["Final"] for t in trajectories]
ax2.hist(final_vals, bins=20, color='gray', edgecolor='black')
ax2.set_title(f"Distribution of Final Values ({holding_years}-year horizon)")
ax2.set_xlabel("Final Portfolio Value ($)")
ax2.set_ylabel("Frequency")
st.pyplot(fig2)
