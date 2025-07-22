import pandas as pd
import numpy as np
import plotly.graph_objs as go
from plotly.colors import sample_colorscale

# Quarterly return data
data = {
    "Year": [2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016, 2015],
    "Q1": [-4.29, 10.45, 7.47, -4.49, 6.19, -19.61, 13.65, -0.76, 5.97, 1.36, 0.97],
    "Q2": [10.87, 4.30, 8.74, -16.17, 8.48, 20.55, 4.27, 3.42, 3.08, 2.45, 0.26],
    "Q3": [None, 5.88, -3.31, -4.82, 0.55, 8.90, 1.75, 7.65, 4.46, 3.82, -6.52],
    "Q4": [None, 2.44, 11.80, 7.40, 11.02, 12.20, 9.03, -13.54, 6.68, 3.92, 7.07]
}

# Prepare the data
df = pd.DataFrame(data)
df.set_index("Year", inplace=True)
df = df[::-1]  # chronological order
df_long = df.stack().reset_index()
df_long.columns = ["Year", "Quarter", "Return"]
df_long["Date"] = df_long["Year"].astype(str) + " " + df_long["Quarter"]

# Investment simulation
initial_investment = 200_000
trajectories = []

for start_idx in range(len(df_long)):
    start_label = df_long.loc[start_idx, "Date"]
    investment = initial_investment
    trajectory = [investment]
    time_labels = [start_label]

    for i in range(start_idx + 1, len(df_long)):
        ret = df_long.loc[i, "Return"]
        date = df_long.loc[i, "Date"]
        if pd.notna(ret):
            investment *= (1 + ret / 100)
            trajectory.append(investment)
            time_labels.append(date)

    if len(trajectory) > 1:
        # Calculate CAGR
        years = len(trajectory) / 4  # since each quarter = 1/4 year
        cagr = (trajectory[-1] / initial_investment) ** (1 / years) - 1
        trajectories.append({
            "Start": start_label,
            "X": time_labels,
            "Y": trajectory,
            "Final": trajectory[-1],
            "CAGR": cagr
        })

# Normalize final values for colormap
final_values = np.array([t["Final"] for t in trajectories])
min_val, max_val = final_values.min(), final_values.max()
normalized = (final_values - min_val) / (max_val - min_val)
colors = sample_colorscale("Turbo", normalized)

# Identify best and worst
best_idx = final_values.argmax()
worst_idx = final_values.argmin()

# Plot
fig = go.Figure()

for i, t in enumerate(trajectories):
    cagr_pct = t["CAGR"] * 100
    hover_text = [f"Start: {t['Start']}<br>Value: ${v:,.0f}<br>CAGR: {cagr_pct:.2f}%" for v in t["Y"]]
    fig.add_trace(go.Scatter(
        x=t["X"],
        y=t["Y"],
        mode='lines',
        line=dict(color=colors[i], width=2),
        name=f"{t['Start']} ({cagr_pct:.2f}% CAGR)",
        hoverinfo='text',
        text=hover_text,
        showlegend=False
    ))

# Add best point marker
best = trajectories[best_idx]
fig.add_trace(go.Scatter(
    x=[best["X"][-1]],
    y=[best["Y"][-1]],
    mode='markers+text',
    marker=dict(color='green', size=12, symbol='star'),
    text=["🏆 Best Start"],
    textposition='top center',
    name='Best Start'
))

# Add worst point marker
worst = trajectories[worst_idx]
fig.add_trace(go.Scatter(
    x=[worst["X"][-1]],
    y=[worst["Y"][-1]],
    mode='markers+text',
    marker=dict(color='red', size=12, symbol='x'),
    text=["💀 Worst Start"],
    textposition='top center',
    name='Worst Start'
))

# Layout
fig.update_layout(
    title="📈 Growth of $200K Invested in Different Quarters (CAGR Included)",
    xaxis_title="Quarter",
    yaxis_title="Portfolio Value ($)",
    hovermode="closest",
    template="plotly_white",
    height=750
)

fig.show()
