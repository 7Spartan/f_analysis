import numpy as np
import matplotlib.pyplot as plt

# --- Inputs ---
initial_investment = 200_000
monthly_contribution = 2000
annual_contribution = monthly_contribution * 12
years = 10
mean_return = 0.10
volatility = 0.10
num_simulations = 1000
plot_samples = 100                  # Number of simulations to plot
np.random.seed(42)

# Adjusted log-mean return for lognormal compounding
log_mean_return = mean_return - 0.5 * volatility**2

# --- Simulation ---
all_trajectories = []
final_values = []

for _ in range(num_simulations):
    value = initial_investment
    yearly_values = [value]
    for _ in range(years):
        value += annual_contribution
        annual_r = np.random.normal(loc=log_mean_return, scale=volatility)
        value *= np.exp(annual_r)
        yearly_values.append(value)
    all_trajectories.append(yearly_values)
    final_values.append(value)

# --- Analysis ---
actual_investment = initial_investment + (annual_contribution * years)
final_values = np.array(final_values)
median = np.median(final_values)
p10 = np.percentile(final_values, 10)
p90 = np.percentile(final_values, 90)
mean = np.mean(final_values)
# --- Print Summary ---
print(f"Actual Investment: ${actual_investment:,.0f}")
print(f"Initial Investment: ${initial_investment:,.0f}")
print(f"Monthly Contribution: ${monthly_contribution:,.0f} (${annual_contribution:,.0f}/yr)")
print(f"Years: {years}")
print(f"Expected CAGR: {mean_return*100:.2f}%")
print(f"Volatility: {volatility*100:.2f}%")
print(f"Simulations: {num_simulations}")
print("—" * 40)
print(f"Median Value after {years} years: ${median:,.0f}")
print(f"10th Percentile (pessimistic): ${p10:,.0f}")
print(f"90th Percentile (optimistic): ${p90:,.0f}")
print(f"Mean Value: ${mean:,.0f}")

# --- Plot Overlay of Simulations ---
years_range = np.arange(0, years + 1)

plt.figure(figsize=(12, 6))
for traj in all_trajectories[:plot_samples]:
    plt.plot(years_range, traj, alpha=0.4, linewidth=1)

plt.title(f"Portfolio Value Over Time\n({plot_samples} Simulations, Monthly Contributions, Annual Compounding)")
plt.xlabel("Years")
plt.ylabel("Portfolio Value (CAD)")
plt.grid(True)
plt.tight_layout()
plt.show()
