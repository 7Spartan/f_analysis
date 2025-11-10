import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# Currency conversion rates (approximate, you may want to fetch live rates)
CURRENCY_RATES = {
    'INR_TO_USD': 0.012,  # 1 INR ≈ 0.012 USD
    'INR_TO_CAD': 0.017,  # 1 INR ≈ 0.017 CAD
}

def get_currency_from_ticker(ticker):
    """Determine currency based on ticker suffix"""
    if ticker.endswith('.NS') or ticker.endswith('.BO'):
        return 'INR'
    elif ticker.endswith('.TO'):
        return 'CAD'
    else:
        return 'USD'

def convert_to_base_currency(amount, from_currency, to_currency='CAD'):
    """Convert amount from one currency to base currency"""
    if from_currency == to_currency:
        return amount
    
    if from_currency == 'INR':
        if to_currency == 'USD':
            return amount * CURRENCY_RATES['INR_TO_USD']
        elif to_currency == 'CAD':
            return amount * CURRENCY_RATES['INR_TO_CAD']
    elif from_currency == 'USD' and to_currency == 'CAD':
        return amount * 1.35  # Approximate USD to CAD
    elif from_currency == 'CAD' and to_currency == 'USD':
        return amount / 1.35
    
    return amount  # Fallback if conversion not defined

st.set_page_config(page_title="TFSA Dividend Strategy Analyzer", layout="wide")
st.title("💰 TFSA Dividend Strategy Analyzer")

st.markdown("""
**Two Analysis Modes:**
- **Forward Projection** (default): Uses assumed growth rates to project future performance
- **Historical Backtest** ✨: See actual performance using real historical prices & dividends from any start year (2010-2025)

**New: Global Dividend Stocks** 🌍
- Added Indian stocks (NSE: .NS suffix)
- Smart filtering based on yield and payout ratio
- Quality scoring system to find best opportunities
- **Currency Conversion**: All prices/dividends automatically converted to your selected currency (CAD/USD)
  - Indian Rupees (INR) → CAD/USD
  - Exchange rates: 1 INR ≈ 0.012 USD ≈ 0.017 CAD
""")

# --- Popular Dividend Stocks/ETFs ---
DIVIDEND_TICKERS = {
    "ETFs": {
        "VYM": "Vanguard High Dividend Yield ETF",
        "SCHD": "Schwab U.S. Dividend Equity ETF",
        "DGRO": "iShares Core Dividend Growth ETF",
        "VIG": "Vanguard Dividend Appreciation ETF",
        "JEPI": "JPMorgan Equity Premium Income ETF",
        "QYLD": "Global X NASDAQ 100 Covered Call ETF",
    },
    "Canadian ETFs": {
        "VDY.TO": "Vanguard FTSE Canadian High Dividend Yield",
        "XDV.TO": "iShares Canadian Select Dividend",
        "CDZ.TO": "iShares S&P/TSX Canadian Dividend Aristocrats",
        "ZDV.TO": "BMO Canadian Dividend ETF",
    },
    "Blue Chip Stocks": {
        "JNJ": "Johnson & Johnson",
        "PG": "Procter & Gamble",
        "KO": "Coca-Cola",
        "PEP": "PepsiCo",
        "MCD": "McDonald's",
        "O": "Realty Income (Monthly Dividend REIT)",
    },
    "High Yield": {
        "T": "AT&T",
        "MO": "Altria Group",
        "VZ": "Verizon",
        "BTI": "British American Tobacco",
    },
    "Indian Stocks": {
        "HDFCBANK.NS": "HDFC Bank",
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy Services",
        "INFY.NS": "Infosys",
        "ITC.NS": "ITC Limited",
        "HINDUNILVR.NS": "Hindustan Unilever",
        "COALINDIA.NS": "Coal India",
        "ONGC.NS": "Oil and Natural Gas Corporation",
        "NTPC.NS": "NTPC Limited",
        "POWERGRID.NS": "Power Grid Corporation",
        "SBIN.NS": "State Bank of India",
        "VEDL.NS": "Vedanta Limited",
    }
}

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Portfolio Configuration")

# Currency Selection
base_currency = st.sidebar.selectbox(
    "Display Currency",
    ["CAD", "USD"],
    help="All prices and dividends will be converted to this currency"
)

# TFSA Starting Amount
initial_investment = st.sidebar.number_input(
    f"Initial TFSA Investment (${base_currency})", 
    min_value=1000, 
    max_value=500000, 
    value=60000, 
    step=5000
)

# Annual Contribution
annual_contribution = st.sidebar.number_input(
    f"Annual TFSA Contribution (${base_currency})", 
    min_value=0, 
    max_value=10000, 
    value=7000, 
    step=500
)

# Investment Duration
years = st.sidebar.slider("Investment Duration (Years)", 1, 30, 10)

# Historical Backtest Start Year
st.sidebar.subheader("📅 Historical Backtest")
use_historical = st.sidebar.checkbox("Use Historical Data (Backtest)", value=False)
start_year = st.sidebar.slider("Investment Start Year", 2010, 2025, 2015) if use_historical else None

# Dividend Reinvestment
drip = st.sidebar.checkbox("Enable DRIP (Dividend Reinvestment Plan)", value=True)

st.sidebar.markdown("---")

# Smart Stock Filtering
st.sidebar.subheader("🎯 Smart Stock Filter")
use_smart_filter = st.sidebar.checkbox("Enable Quality Filter", value=False)

if use_smart_filter:
    with st.sidebar.expander("📋 Filter Criteria"):
        min_yield = st.slider("Min Dividend Yield (%)", 0.0, 10.0, 2.0, 0.5)
        max_payout = st.slider("Max Payout Ratio (%)", 10, 100, 70, 5)
        st.info("""
        **Quality Criteria:**
        - Minimum yield ensures income
        - Lower payout ratio = safer dividends
        - Stocks meeting criteria shown with ✅
        """)

st.sidebar.markdown("---")

# Stock Selection
st.sidebar.subheader("📊 Select Dividend Investments")

selected_tickers = []

# Auto-select quality stocks option
if use_smart_filter:
    auto_select = st.sidebar.checkbox("Auto-Select Quality Stocks", value=False)
    if auto_select:
        st.sidebar.info("Will auto-select stocks meeting criteria after loading data")

for category, tickers in DIVIDEND_TICKERS.items():
    with st.sidebar.expander(f"🔹 {category}"):
        for ticker, name in tickers.items():
            if st.checkbox(f"{ticker} - {name}", key=ticker):
                selected_tickers.append(ticker)

# Custom tickers
custom_input = st.sidebar.text_input(
    "Add Custom Tickers (comma-separated)", 
    placeholder="AAPL, MSFT, ENB.TO"
)
if custom_input:
    custom_tickers = [t.strip().upper() for t in custom_input.split(",") if t.strip()]
    selected_tickers.extend(custom_tickers)

# Remove duplicates
selected_tickers = list(set(selected_tickers))

if not selected_tickers:
    st.warning("⚠️ Please select at least one dividend stock or ETF from the sidebar.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.info(f"**{len(selected_tickers)} investments selected**")

# --- Data Loading ---
@st.cache_data(ttl=3600, show_spinner="📥 Fetching dividend & price data...")
def load_dividend_data(tickers, years_back=5, start_year=None, base_currency='CAD'):
    """Load historical dividend and price data with currency conversion"""
    data = {}
    if start_year:
        # For backtesting, load from specific start year to now
        start_date = f"{start_year}-01-01"
    else:
        start_date = (datetime.now() - timedelta(days=years_back*365)).strftime('%Y-%m-%d')
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            
            # Determine source currency
            source_currency = get_currency_from_ticker(ticker)
            
            # Get dividend history
            dividends = stock.dividends
            if dividends.empty:
                st.warning(f"⚠️ No dividend data found for {ticker}")
                continue
            
            # Convert dividends to base currency
            dividends = dividends * convert_to_base_currency(1, source_currency, base_currency)
            
            # Get price history
            hist = stock.history(start=start_date, auto_adjust=True)
            
            # Convert prices to base currency
            if not hist.empty:
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in hist.columns:
                        hist[col] = hist[col] * convert_to_base_currency(1, source_currency, base_currency)
            
            # Get current info
            info = stock.info
            current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else None)
            
            # Convert current price to base currency
            if current_price:
                current_price = current_price * convert_to_base_currency(1, source_currency, base_currency)
            
            dividend_yield = info.get('dividendYield', None)
            payout_ratio = info.get('payoutRatio', None)
            
            # Calculate trailing 12-month dividend (already converted)
            # Ensure timezone-aware comparison
            if dividends.index.tz is not None:
                # Dividend index is timezone-aware
                now = pd.Timestamp.now(tz=dividends.index.tz)
            else:
                # Dividend index is timezone-naive
                now = pd.Timestamp.now()
            
            one_year_ago = now - pd.Timedelta(days=365)
            recent_divs = dividends[dividends.index > one_year_ago]
            ttm_dividend = recent_divs.sum() if not recent_divs.empty else 0
            
            data[ticker] = {
                'dividends': dividends,
                'prices': hist['Close'],
                'current_price': current_price,
                'dividend_yield': dividend_yield,
                'ttm_dividend': ttm_dividend,
                'payout_ratio': payout_ratio,
                'name': info.get('longName', ticker),
                'source_currency': source_currency,
                'converted_to': base_currency
            }
            
        except Exception as e:
            st.error(f"❌ Error loading {ticker}: {str(e)}")
            continue
    
    return data

# Load data
with st.spinner("Loading dividend data..."):
    dividend_data = load_dividend_data(
        selected_tickers, 
        start_year=start_year if use_historical else None,
        base_currency=base_currency
    )

if not dividend_data:
    st.error("❌ No valid dividend data loaded. Please check your ticker symbols.")
    st.stop()

# --- Display Current Dividend Metrics ---
st.subheader("📊 Current Dividend Metrics")

# Build metrics dataframe with quality indicators
metrics_df = []
quality_stocks = []

for ticker, data in dividend_data.items():
    div_yield = data['dividend_yield'] * 100 if data['dividend_yield'] else None
    payout = data['payout_ratio'] * 100 if data['payout_ratio'] else None
    
    # Check if meets quality criteria
    meets_criteria = True
    quality_marker = ""
    
    if use_smart_filter:
        if div_yield is not None and payout is not None:
            if div_yield >= min_yield and payout <= max_payout:
                quality_marker = "✅"
                quality_stocks.append(ticker)
            else:
                quality_marker = "❌"
                meets_criteria = False
        else:
            quality_marker = "⚠️"  # Missing data
    
    # Show original currency if converted
    currency_note = ""
    if data['source_currency'] != base_currency:
        currency_note = f" ({data['source_currency']}→{base_currency})"
    
    metrics_df.append({
        'Quality': quality_marker,
        'Ticker': ticker + currency_note,
        'Name': data['name'][:40],
        'Current Price': f"${data['current_price']:.2f}" if data['current_price'] else "N/A",
        'TTM Dividend': f"${data['ttm_dividend']:.2f}",
        'Dividend Yield': f"{div_yield:.2f}%" if div_yield else "N/A",
        'Payout Ratio': f"{payout:.1f}%" if payout else "N/A"
    })

# Display metrics
df_display = pd.DataFrame(metrics_df)

# Show currency info
st.info(f"💱 **Display Currency: {base_currency}** - All prices and dividends converted to {base_currency}")

if use_smart_filter:
    st.info(f"🎯 **{len(quality_stocks)}** stocks meet quality criteria (min yield: {min_yield}%, max payout: {max_payout}%)")
    
st.dataframe(df_display, use_container_width=True)

# Show quality recommendations
if use_smart_filter and quality_stocks:
    st.success(f"**Quality Picks:** {', '.join(quality_stocks)}")
    
# Add a "Scan All Stocks" feature
with st.expander("🔍 Scan All Available Stocks (Beta)"):
    st.write("Scan through all available dividend stocks to find the best opportunities:")
    
    if st.button("Scan All Dividend Stocks"):
        with st.spinner("Scanning all stocks... this may take a minute"):
            all_tickers = []
            for category, tickers in DIVIDEND_TICKERS.items():
                all_tickers.extend(tickers.keys())
            
            scan_data = load_dividend_data(
                all_tickers, 
                start_year=start_year if use_historical else None,
                base_currency=base_currency
            )
            
            if scan_data:
                scan_results = []
                for ticker, data in scan_data.items():
                    div_yield = data['dividend_yield'] * 100 if data['dividend_yield'] else 0
                    payout = data['payout_ratio'] * 100 if data['payout_ratio'] else 0
                    
                    # Calculate quality score
                    score = 0
                    if div_yield >= 2.0:
                        score += 1
                    if div_yield >= 3.0:
                        score += 1
                    if payout < 70 and payout > 0:
                        score += 1
                    if payout < 50 and payout > 0:
                        score += 1
                    
                    scan_results.append({
                        'Ticker': ticker,
                        'Name': data['name'][:35],
                        'Yield': div_yield,
                        'Payout Ratio': payout,
                        'Quality Score': score,
                        'Price': data['current_price']
                    })
                
                # Sort by quality score
                scan_df = pd.DataFrame(scan_results).sort_values('Quality Score', ascending=False)
                
                st.subheader("📊 Stock Rankings")
                st.write("**Quality Score**: 0-4 (Higher is better)")
                st.dataframe(scan_df, use_container_width=True)
                
                # Show top picks
                top_picks = scan_df.head(5)
                st.success(f"**Top 5 Quality Picks:** {', '.join(top_picks['Ticker'].tolist())}")

# --- Portfolio Allocation ---
st.markdown("---")
st.subheader("💼 Portfolio Allocation")

col1, col2 = st.columns([2, 1])

with col1:
    st.write("Allocate your $60,000 investment across selected securities:")
    allocations = {}
    total_allocation = 0
    
    for ticker in dividend_data.keys():
        default_alloc = 100 // len(dividend_data)
        alloc = st.slider(
            f"{ticker} allocation (%)", 
            0, 100, 
            default_alloc, 
            5, 
            key=f"alloc_{ticker}"
        )
        allocations[ticker] = alloc / 100
        total_allocation += alloc
    
    if abs(total_allocation - 100) > 0.1:
        st.error(f"⚠️ Total allocation is {total_allocation}%, must equal 100%")
        st.stop()

with col2:
    st.metric("Total Allocation", f"{total_allocation}%")
    
    # Show dollar amounts
    st.write("**Initial Investment per Ticker:**")
    for ticker, alloc in allocations.items():
        amount = initial_investment * alloc
        st.write(f"• {ticker}: ${amount:,.0f}")

# --- Dividend Projection Simulation ---
st.markdown("---")
st.subheader("📈 Dividend Growth Projection")

# Growth assumptions
col1, col2, col3 = st.columns(3)
with col1:
    avg_div_growth = st.slider("Assumed Annual Dividend Growth (%)", 0.0, 10.0, 3.0, 0.5) / 100
with col2:
    avg_price_growth = st.slider("Assumed Annual Price Appreciation (%)", -5.0, 15.0, 5.0, 0.5) / 100
with col3:
    div_freq = st.selectbox("Dividend Frequency", ["Quarterly", "Monthly", "Annual"], index=0)

freq_map = {"Quarterly": 4, "Monthly": 12, "Annual": 1}
periods_per_year = freq_map[div_freq]

# Run simulation
def simulate_dividend_portfolio_historical(initial, annual_contrib, allocations, dividend_data, 
                                           start_year, duration_years, drip):
    """Backtest using actual historical prices and dividends"""
    
    # Initialize holdings at start_year
    shares = {}
    
    # Detect timezone from first ticker's price data
    first_ticker = list(allocations.keys())[0]
    sample_prices = dividend_data[first_ticker]['prices']
    
    # Create timezone-aware timestamps matching the data
    if sample_prices.index.tz is not None:
        # Create naive timestamp first, then localize to match data timezone
        start_date = pd.Timestamp(f'{start_year}-01-01').tz_localize(sample_prices.index.tz)
        end_date = pd.Timestamp(f'{start_year + duration_years}-12-31').tz_localize(sample_prices.index.tz)
    else:
        start_date = pd.Timestamp(f'{start_year}-01-01')
        end_date = pd.Timestamp(f'{start_year + duration_years}-12-31')
    
    # Get initial prices and purchase shares
    for ticker, alloc in allocations.items():
        initial_amount = initial * alloc
        hist_prices = dividend_data[ticker]['prices']
        
        # Find first available price on or after start date
        start_prices = hist_prices[hist_prices.index >= start_date]
        if start_prices.empty:
            st.warning(f"No price data for {ticker} starting {start_year}")
            shares[ticker] = 0
            continue
        
        start_price = start_prices.iloc[0]
        shares[ticker] = initial_amount / start_price if start_price > 0 else 0
    
    # Create monthly timeline (timezone-aware if needed)
    if sample_prices.index.tz is not None:
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS', tz=sample_prices.index.tz)
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    
    timeline = []
    total_values = []
    dividend_incomes = []
    cumulative_dividends = []
    share_counts = {ticker: [] for ticker in allocations.keys()}
    
    total_dividends_received = 0
    next_contrib_year = start_year + 1
    
    for date in date_range:
        # Calculate current portfolio value
        portfolio_value = 0
        for ticker, num_shares in shares.items():
            hist_prices = dividend_data[ticker]['prices']
            # Get price at this date
            prices_at_date = hist_prices[hist_prices.index <= date]
            if not prices_at_date.empty:
                current_price = prices_at_date.iloc[-1]
                portfolio_value += num_shares * current_price
        
        # Get dividends paid in this month
        period_dividend = 0
        for ticker, num_shares in shares.items():
            divs = dividend_data[ticker]['dividends']
            # Get dividends paid in this specific month
            # Calculate end of month for comparison
            next_month = date + pd.DateOffset(months=1)
            month_divs = divs[(divs.index >= date) & (divs.index < next_month)]
            if not month_divs.empty:
                div_amount = month_divs.sum() * num_shares
                period_dividend += div_amount
                total_dividends_received += div_amount
                
                # Reinvest if DRIP enabled
                if drip and div_amount > 0:
                    hist_prices = dividend_data[ticker]['prices']
                    prices_at_div = hist_prices[hist_prices.index <= date]
                    if not prices_at_div.empty:
                        reinvest_price = prices_at_div.iloc[-1]
                        if reinvest_price > 0:
                            shares[ticker] += div_amount / reinvest_price
        
        # Add annual contribution at start of each year
        if date.year == next_contrib_year and date.month == 1 and annual_contrib > 0:
            for ticker, alloc in allocations.items():
                contrib_amount = annual_contrib * alloc
                hist_prices = dividend_data[ticker]['prices']
                prices_at_date = hist_prices[hist_prices.index <= date]
                if not prices_at_date.empty:
                    contrib_price = prices_at_date.iloc[-1]
                    if contrib_price > 0:
                        shares[ticker] += contrib_amount / contrib_price
            next_contrib_year += 1
        
        # Record data
        years_elapsed = (date - start_date).days / 365.25
        timeline.append(years_elapsed)
        total_values.append(portfolio_value)
        dividend_incomes.append(period_dividend * 12)  # Annualized
        cumulative_dividends.append(total_dividends_received)
        
        for ticker in allocations.keys():
            share_counts[ticker].append(shares[ticker])
    
    return pd.DataFrame({
        'Year': timeline,
        'Portfolio_Value': total_values,
        'Annual_Dividend_Income': dividend_incomes,
        'Cumulative_Dividends': cumulative_dividends,
        **{f'Shares_{ticker}': share_counts[ticker] for ticker in allocations.keys()}
    })

def simulate_dividend_portfolio(initial, annual_contrib, allocations, dividend_data, 
                                years, div_growth, price_growth, drip, periods_per_year):
    """Simulate dividend portfolio growth over time"""
    
    # Initialize holdings
    shares = {}
    for ticker, alloc in allocations.items():
        initial_amount = initial * alloc
        price = dividend_data[ticker]['current_price']
        shares[ticker] = initial_amount / price if price else 0
    
    # Tracking arrays
    timeline = []
    total_values = []
    dividend_incomes = []
    cumulative_dividends = []
    share_counts = {ticker: [] for ticker in allocations.keys()}
    
    total_dividends_received = 0
    
    for year in range(years + 1):
        for period in range(periods_per_year):
            # Calculate current portfolio value and dividends
            portfolio_value = 0
            period_dividend = 0
            
            for ticker, num_shares in shares.items():
                # Update price
                current_price = dividend_data[ticker]['current_price']
                price_at_period = current_price * ((1 + price_growth) ** (year + period / periods_per_year))
                portfolio_value += num_shares * price_at_period
                
                # Calculate dividend for this period
                annual_div = dividend_data[ticker]['ttm_dividend'] * ((1 + div_growth) ** year)
                period_div = (annual_div / periods_per_year) * num_shares
                period_dividend += period_div
                total_dividends_received += period_div
                
                # Reinvest if DRIP enabled
                if drip and period_div > 0:
                    shares[ticker] += period_div / price_at_period
            
            # Add annual contribution at year start (period 0)
            if period == 0 and year > 0 and annual_contrib > 0:
                for ticker, alloc in allocations.items():
                    contrib_amount = annual_contrib * alloc
                    price = dividend_data[ticker]['current_price'] * ((1 + price_growth) ** year)
                    shares[ticker] += contrib_amount / price if price else 0
            
            # Record data
            time_point = year + period / periods_per_year
            timeline.append(time_point)
            total_values.append(portfolio_value)
            dividend_incomes.append(period_dividend * periods_per_year)  # Annualized
            cumulative_dividends.append(total_dividends_received)
            
            for ticker in allocations.keys():
                share_counts[ticker].append(shares[ticker])
    
    return pd.DataFrame({
        'Year': timeline,
        'Portfolio_Value': total_values,
        'Annual_Dividend_Income': dividend_incomes,
        'Cumulative_Dividends': cumulative_dividends,
        **{f'Shares_{ticker}': share_counts[ticker] for ticker in allocations.keys()}
    })

# Run simulation
if use_historical:
    st.info(f"📊 Running historical backtest from {start_year} to {start_year + years}")
    results = simulate_dividend_portfolio_historical(
        initial_investment,
        annual_contribution,
        allocations,
        dividend_data,
        start_year,
        years,
        drip
    )
else:
    results = simulate_dividend_portfolio(
        initial_investment,
        annual_contribution,
        allocations,
        dividend_data,
        years,
        avg_div_growth,
        avg_price_growth,
        drip,
        periods_per_year
    )

# --- Results Visualization ---
if use_historical:
    st.success(f"✅ Historical Backtest: {start_year} to {start_year + years} (Using Real Historical Data)")
else:
    st.info("📈 Forward Projection (Using Assumed Growth Rates)")

col1, col2, col3, col4 = st.columns(4)

final_value = results['Portfolio_Value'].iloc[-1]
total_invested = initial_investment + (annual_contribution * years)
total_gain = final_value - total_invested
final_annual_income = results['Annual_Dividend_Income'].iloc[-1]
cumulative_divs = results['Cumulative_Dividends'].iloc[-1]

with col1:
    st.metric("Final Portfolio Value", f"${final_value:,.0f}")
with col2:
    st.metric("Total Invested", f"${total_invested:,.0f}")
with col3:
    st.metric("Total Gain", f"${total_gain:,.0f}", f"{(total_gain/total_invested)*100:.1f}%")
with col4:
    st.metric("Final Annual Dividend Income", f"${final_annual_income:,.0f}")

st.metric(
    "Total Dividends Received", 
    f"${cumulative_divs:,.0f}",
    help="Total dividend payments received over the investment period"
)

# Portfolio Value Chart
chart_title = f'Portfolio Value Over Time ({start_year}-{start_year+years})' if use_historical else 'Portfolio Value Over Time (Projection)'
fig_value = px.line(
    results, 
    x='Year', 
    y='Portfolio_Value',
    title=chart_title,
    labels={'Portfolio_Value': f'Portfolio Value ({base_currency})', 'Year': 'Years'}
)
fig_value.add_hline(
    y=total_invested, 
    line_dash="dash", 
    line_color="gray",
    annotation_text=f"Total Invested: ${total_invested:,.0f}"
)
fig_value.update_layout(yaxis_tickformat='$,.0f', height=400)
st.plotly_chart(fig_value, use_container_width=True)

# Dividend Income Chart
fig_income = px.line(
    results, 
    x='Year', 
    y='Annual_Dividend_Income',
    title='Annual Dividend Income Over Time',
    labels={'Annual_Dividend_Income': f'Annual Dividend Income ({base_currency})', 'Year': 'Years'}
)
fig_income.update_traces(line_color='green', line_width=3)
fig_income.update_layout(yaxis_tickformat='$,.0f', height=400)
st.plotly_chart(fig_income, use_container_width=True)

# Cumulative Dividends Chart
fig_cum_div = go.Figure()
fig_cum_div.add_trace(go.Scatter(
    x=results['Year'],
    y=results['Cumulative_Dividends'],
    mode='lines',
    fill='tozeroy',
    name='Cumulative Dividends',
    line=dict(color='lightgreen', width=2)
))
fig_cum_div.update_layout(
    title='Cumulative Dividends Received',
    xaxis_title='Years',
    yaxis_title='Cumulative Dividends ($)',
    yaxis_tickformat='$,.0f',
    height=400
)
st.plotly_chart(fig_cum_div, use_container_width=True)

# Share accumulation (if multiple tickers)
if len(allocations) > 1:
    st.markdown("---")
    st.subheader("📊 Share Accumulation")
    
    share_cols = [col for col in results.columns if col.startswith('Shares_')]
    share_data = results[['Year'] + share_cols].copy()
    share_data.columns = ['Year'] + [col.replace('Shares_', '') for col in share_cols]
    
    fig_shares = px.line(
        share_data, 
        x='Year', 
        y=share_data.columns[1:],
        title='Share Count Over Time (with DRIP)' if drip else 'Share Count Over Time',
        labels={'value': 'Number of Shares', 'variable': 'Ticker'}
    )
    fig_shares.update_layout(height=400)
    st.plotly_chart(fig_shares, use_container_width=True)

# --- Comparison: DRIP vs No DRIP ---
st.markdown("---")
st.subheader("🔄 DRIP Impact Analysis")

col1, col2 = st.columns(2)

# Run both scenarios
if use_historical:
    results_drip = simulate_dividend_portfolio_historical(
        initial_investment, annual_contribution, allocations, dividend_data,
        start_year, years, True
    )
    results_no_drip = simulate_dividend_portfolio_historical(
        initial_investment, annual_contribution, allocations, dividend_data,
        start_year, years, False
    )
else:
    results_drip = simulate_dividend_portfolio(
        initial_investment, annual_contribution, allocations, dividend_data,
        years, avg_div_growth, avg_price_growth, True, periods_per_year
    )
    results_no_drip = simulate_dividend_portfolio(
        initial_investment, annual_contribution, allocations, dividend_data,
        years, avg_div_growth, avg_price_growth, False, periods_per_year
    )

with col1:
    final_with_drip = results_drip['Portfolio_Value'].iloc[-1]
    st.metric("Final Value WITH DRIP", f"${final_with_drip:,.0f}")
    
with col2:
    final_without_drip = results_no_drip['Portfolio_Value'].iloc[-1]
    drip_difference = final_with_drip - final_without_drip
    st.metric(
        "Final Value WITHOUT DRIP", 
        f"${final_without_drip:,.0f}",
        delta=f"-${drip_difference:,.0f}"
    )

# Comparison chart
fig_comparison = go.Figure()
fig_comparison.add_trace(go.Scatter(
    x=results_drip['Year'],
    y=results_drip['Portfolio_Value'],
    mode='lines',
    name='With DRIP',
    line=dict(color='green', width=3)
))
fig_comparison.add_trace(go.Scatter(
    x=results_no_drip['Year'],
    y=results_no_drip['Portfolio_Value'],
    mode='lines',
    name='Without DRIP',
    line=dict(color='orange', width=3)
))
fig_comparison.update_layout(
    title='Portfolio Value: DRIP vs No DRIP',
    xaxis_title='Years',
    yaxis_title='Portfolio Value ($)',
    yaxis_tickformat='$,.0f',
    height=500
)
st.plotly_chart(fig_comparison, use_container_width=True)

# --- Key Insights ---
st.markdown("---")
st.subheader("💡 Key Insights")

avg_yield = np.mean([d['dividend_yield'] for d in dividend_data.values() if d['dividend_yield']])
initial_annual_income = initial_investment * avg_yield

col1, col2 = st.columns(2)

with col1:
    st.write("**📈 Growth Metrics:**")
    st.write(f"• Average portfolio yield: {avg_yield*100:.2f}%")
    st.write(f"• Initial annual dividend income: ${initial_annual_income:,.0f}")
    st.write(f"• Final annual dividend income: ${final_annual_income:,.0f}")
    st.write(f"• Dividend income growth: {((final_annual_income/initial_annual_income - 1)*100):.1f}%")
    
    yield_on_cost = (final_annual_income / initial_investment) * 100
    st.write(f"• **Yield on Cost (Year {years}):** {yield_on_cost:.2f}%")

with col2:
    st.write("**💰 Returns:**")
    total_return = ((final_value / total_invested) - 1) * 100
    cagr = ((final_value / initial_investment) ** (1 / years) - 1) * 100
    
    st.write(f"• Total return: {total_return:.1f}%")
    st.write(f"• CAGR: {cagr:.2f}%")
    st.write(f"• DRIP impact: ${drip_difference:,.0f} ({(drip_difference/final_without_drip)*100:.1f}%)")

st.info("""
**📚 Understanding Dividends:**
- **Dividend Yield** = Annual Dividend per Share / Stock Price
- **DRIP** = Dividend Reinvestment Plan (automatically buys more shares with dividends)
- **Yield on Cost** = Current annual dividend income / Original investment (shows how your yield grows over time)
- **Payout Ratio** = Dividends / Earnings (lower is safer, higher growth potential)

**🎯 Quality Criteria Explained:**
- **Minimum Yield 2-3%**: Ensures meaningful income generation
- **Payout Ratio < 70%**: Company retains earnings for growth and dividend increases
- **Payout Ratio < 50%**: Very safe, lots of room for dividend growth
- **Quality Score 4/4**: Meets all safety and income thresholds

**🇮🇳 Indian Stocks Notes:**
- Dividends in INR, prices may differ from CAD/USD
- Tax implications may vary for Indian securities
- NSE stocks use .NS suffix (National Stock Exchange)
- Consider currency exchange rates for TFSA planning
""")

# --- Raw Data Table ---
with st.expander("📋 View Detailed Year-by-Year Data"):
    # Sample every year for cleaner display
    yearly_data = results[results['Year'].apply(lambda x: x == int(x))].copy()
    yearly_data['Year'] = yearly_data['Year'].astype(int)
    
    display_cols = ['Year', 'Portfolio_Value', 'Annual_Dividend_Income', 'Cumulative_Dividends']
    yearly_data_display = yearly_data[display_cols].copy()
    
    for col in ['Portfolio_Value', 'Annual_Dividend_Income', 'Cumulative_Dividends']:
        yearly_data_display[col] = yearly_data_display[col].apply(lambda x: f"${x:,.0f}")
    
    st.dataframe(yearly_data_display, use_container_width=True)
