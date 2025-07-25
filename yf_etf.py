import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ETF Monthly Returns Explorer", layout="wide")
st.title("📈 ETF Monthly Returns Explorer (2010–2024)")

# --- Config ---
tickers = ["QQQ", "VOO", "VTI", "VGPMX", "VEUSX"]
names = {
    "QQQ": "Invesco QQQ Trust",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "VGPMX": "Vanguard Precious Metals Fund",
    "VEUSX": "Vanguard European Stock Index Fund"
}

# --- Download Data ---
data = yf.download(tickers, start="2010-01-01", end="2025-01-01", interval="1mo", auto_adjust=True, group_by="ticker", progress=False)

# --- Calculate Monthly Returns ---
returns = {}
for t in tickers:
    if t in data:
        series = data[t]["Close"].dropna()
    else:
        # fallback if data not grouped by ticker
        series = data["Close"][t].dropna()
    returns[t] = series.pct_change().dropna() * 100

# --- Sidebar ---
selected_tickers = st.sidebar.multiselect("Select ETFs", tickers, default=tickers[:2])
window = st.sidebar.slider("Streak Length (Years)", min_value=0.25, max_value=10.0, step=0.25, value=10.0)

# All available ETF returns
all_returns_dict = {
    "VOO": returns["VOO"],
    "QQQ": returns["QQQ"],
    "VTI": returns["VTI"],
    "VGPMX": returns["VGPMX"],
    "VEUSX": returns["VEUSX"]
}

# User selects which ETFs to view
selected_tickers = st.multiselect(
    "Select ETFs",
    options=list(all_returns_dict.keys()),
    default=["VOO", "QQQ"]
)

# Filter dictionary to selected tickers only
selected_returns_dict = {k: all_returns_dict[k] for k in selected_tickers}

# --- Main Chart ---
st.subheader("📊 Monthly Returns Line Chart")
returns_df = pd.DataFrame(selected_returns_dict)
st.line_chart(returns_df)

# for t in selected_tickers:
#     st.line_chart(returns[t], height=200, use_container_width=True)

# --- Histogram ---
st.subheader("📉 Histogram of Monthly Returns")
fig, ax = plt.subplots()
for t in selected_tickers:
    returns[t].hist(bins=50, alpha=0.6, label=t, ax=ax)
ax.set_title("Distribution of Monthly Returns")
ax.set_xlabel("Monthly Return (%)")
ax.legend()
st.pyplot(fig)

# --- Streak Analysis ---
def get_rolling_return(series, years):
    months = int(round(years * 12))  # Convert fractional years to months
    return (series / 100 + 1).rolling(months).apply(lambda x: (x.prod() - 1) * 100)


st.subheader(f"⏳ Best & Worst {window}-Year Return Streaks")
for t in selected_tickers:
    roll = get_rolling_return(returns[t].dropna(), window)
    st.markdown(f"**{names[t]} ({t})**")
    st.markdown(f"🔼 Best {window}-year return: `{roll.max():.2f}%`")
    st.markdown(f"🔽 Worst {window}-year return: `{roll.min():.2f}%`")
    st.line_chart(roll.dropna(), height=150)

# --- Raw Data Table ---
with st.expander("🔍 Show Raw Monthly Return Data"):
    for t in selected_tickers:
        df = returns[t].dropna().to_frame().rename(columns={t: "Monthly Return (%)"})
        df.index = df.index.strftime("%Y-%m")
        st.markdown(f"**{names[t]} ({t})**")
        st.dataframe(df)
