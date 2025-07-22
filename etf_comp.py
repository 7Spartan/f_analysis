import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ETF return data
etf_returns = {
    "QQQ": [25.58, 54.85, -32.58, 27.42, 48.62, 38.96, -0.12, 32.66, 7.1, 9.45, 19.18, 36.63, 18.12, 3.38, 19.91],
    "VOO": [24.98, 26.32, -18.19, 28.78, 18.29, 31.35, -4.5, 21.77, 12.17, 1.31, 13.55, 32.39, 16.0, 1.89, 14.76],
    "VGPMX": [10.03, 7.39, 19.49, 17.21, 42.6, 32.63, -32.27, 19.13, 50.64, -29.42, -16.69, -47.26, -14.85, -26.96, 35.55],
    "VTI": [23.81, 26.05, -19.51, 25.67, 21.03, 30.67, -5.21, 21.21, 12.83, 0.36, 12.54, 33.45, 16.45, 0.97, 17.42],
    "VEUSX": [2.03, 20.0, -16.05, 16.33, 6.44, 24.25, -14.79, 27.05, -0.63, -1.88, -6.55, 24.89, 20.99, -11.49, 5.01]
}
years = list(range(2024, 2009, -1))

st.title("ETF Returns Analyzer")
# layout to be two columns
col1, col2 = st.columns(2)

with col1:
    # ETF selector
    selected_etfs = st.multiselect("Select ETFs:", list(etf_returns.keys()), default=["QQQ", "VOO"])

with col2:
    # Streak length

    streak_length = st.selectbox("Select streak duration (years):", [3, 5, 10], index=1)

# # ETF selector
# selected_etfs = st.multiselect("Select ETFs:", list(etf_returns.keys()), default=["QQQ", "VOO"])

# # Streak length
# streak_length = st.selectbox("Select streak duration (years):", [3, 5, 10], index=1)

# --- Plot Annual Returns ---
st.subheader("Annual Returns")
fig, ax = plt.subplots(figsize=(10, 5))
for etf in selected_etfs:
    ax.plot(years, etf_returns[etf], label=etf, marker='o')

ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel("Year")
ax.set_ylabel("Annual Return (%)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Volatility / Sideways Market ---
st.subheader("Volatility (Standard Deviation of Returns)")
volatility = {etf: np.std(returns) for etf, returns in etf_returns.items()}
sorted_vol = sorted(volatility.items(), key=lambda x: x[1])
st.dataframe(pd.DataFrame(sorted_vol, columns=["ETF", "Volatility (%)"]))
sideways = sorted_vol[0][0]
st.success(f"Best 'sideways' ETF: **{sideways}** with volatility of {sorted_vol[0][1]:.2f}%")

# --- Best/Worst N-Year Streaks ---
st.subheader(f"Best and Worst {streak_length}-Year Streaks (CAGR)")
def get_streaks(returns, years, n):
    results = []
    for i in range(len(returns) - n + 1):
        start = years[i + n - 1]
        end = years[i]
        sub_returns = returns[i:i + n]
        cumulative = np.prod([(1 + r / 100) for r in sub_returns])
        cagr = (cumulative ** (1 / n)) - 1
        results.append((start, end, cagr * 100))
    return results

streak_data = []
for etf in selected_etfs:
    streaks = get_streaks(etf_returns[etf], years, streak_length)
    best = max(streaks, key=lambda x: x[2])
    worst = min(streaks, key=lambda x: x[2])
    streak_data.append({
        "ETF": etf,
        "Best Period": f"{best[0]}–{best[1]}",
        "Best CAGR (%)": best[2],
        "Worst Period": f"{worst[0]}–{worst[1]}",
        "Worst CAGR (%)": worst[2]
    })

st.dataframe(pd.DataFrame(streak_data))

# Optional: Bar Chart
st.subheader("CAGR Comparison")
df_cagr = pd.DataFrame(streak_data)
fig2, ax2 = plt.subplots()
width = 0.35
x = np.arange(len(df_cagr["ETF"]))

ax2.bar(x - width / 2, df_cagr["Best CAGR (%)"], width, label="Best", color="green")
ax2.bar(x + width / 2, df_cagr["Worst CAGR (%)"], width, label="Worst", color="red")
ax2.set_xticks(x)
ax2.set_xticklabels(df_cagr["ETF"])
ax2.set_ylabel("CAGR (%)")
ax2.set_title(f"Best vs Worst {streak_length}-Year CAGR")
ax2.legend()
st.pyplot(fig2)
