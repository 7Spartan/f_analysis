import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="ETF Monthly Returns Explorer", layout="wide")
st.title("📈 ETF Monthly Returns Explorer (2010–2024)")

# --- Config ---
default_tickers = ["QQQ", "VOO", "VTI", "VGPMX", "VEUSX"]
names = {
    "QQQ": "Invesco QQQ Trust",
    "VOO": "Vanguard S&P 500 ETF",
    "VTI": "Vanguard Total Stock Market ETF",
    "VGPMX": "Vanguard Precious Metals Fund",
    "VEUSX": "Vanguard European Stock Index Fund"
}

@st.cache_data(show_spinner="📥 Downloading data...")
def load_data(tickers):
    data = yf.download(
        tickers,
        start="2010-01-01",
        end="2025-01-01",
        interval="1mo",
        auto_adjust=True,
        group_by="ticker",
        progress=False
    )
    return data

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Controls")
selected_tickers = st.sidebar.multiselect("Select ETFs to Compare", default_tickers, default=default_tickers[:2])

custom_ticker_input = st.sidebar.text_input("Add Custom Tickers (comma-separated, e.g. MSFT, NVDA, AAPL)", "")
if custom_ticker_input:
    custom_tickers = [ticker.strip().upper() for ticker in custom_ticker_input.split(",") if ticker.strip()]
    selected_tickers = list(set(selected_tickers + custom_tickers))  # avoid duplicates

st.sidebar.markdown("---")
window = st.sidebar.slider("Streak Length (Years)", min_value=0.25, max_value=10.0, step=0.25, value=10.0)

# --- Load and Process Data ---
data = load_data(selected_tickers)

returns = {}
for t in selected_tickers:
    try:
        series = data[t]["Close"].dropna() if isinstance(data.columns, pd.MultiIndex) else data["Close"][t].dropna()
        returns[t] = series.pct_change().dropna() * 100
    except Exception as e:
        st.error(f"⚠️ Error loading {t}: {e}")

# --- Line Chart ---
st.subheader("📊 Monthly Returns Line Chart")
returns_df = pd.DataFrame(returns)
st.line_chart(returns_df)

# --- Histogram ---
st.subheader("📉 Histogram of Monthly Returns")
fig, ax = plt.subplots()
for t in returns:
    returns[t].hist(bins=50, alpha=0.6, label=t, ax=ax)
ax.set_title("Distribution of Monthly Returns")
ax.set_xlabel("Monthly Return (%)")
ax.legend()
st.pyplot(fig)

# --- Rolling Streak Analysis ---
def get_rolling_return(series, years):
    months = int(round(years * 12))
    return (series / 100 + 1).rolling(months).apply(lambda x: (x.prod() - 1) * 100)

st.subheader(f"⏳ Best & Worst {window}-Year Return Streaks")
for t in returns:
    roll = get_rolling_return(returns[t].dropna(), window)
    st.markdown(f"**{names.get(t, t)} ({t})**")
    st.markdown(f"🔼 Best {window}-year return: `{roll.max():.2f}%`")
    st.markdown(f"🔽 Worst {window}-year return: `{roll.min():.2f}%`")
    st.line_chart(roll.dropna(), height=150)

# --- Raw Data Table ---
with st.expander("🔍 Show Raw Monthly Return Data"):
    for t in returns:
        df = returns[t].dropna().to_frame().rename(columns={t: "Monthly Return (%)"})
        df.index = df.index.strftime("%Y-%m")
        st.markdown(f"**{names.get(t, t)} ({t})**")
        st.dataframe(df)
