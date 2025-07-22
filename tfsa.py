import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
initial_contribution = 66_000          # Current TFSA contribution room
annual_contribution = 7_000            # Ongoing annual contribution
years = 5  # Investment duration in years
mean_return = 0.10                     # Expected annual return (CAGR)
volatility = 0.15                      # Estimated annual volatility of S&P 500
num_simulations = 10000
np.random.seed(42)

# Adjusted log-mean return for lognormal modeling
log_mean_return = mean_return - 0.5 * volatility**2

# --- Monte Carlo Simulation ---
final_values = []
trajectories = []

for _ in range(num_simulations):
    value = initial_contribution
    yearly_values = [value]
    for _ in range(years):
        value += annual_contribution
        annual_r = np.random.normal(loc=log_mean_return, scale=volatility)
        value *= np.exp(annual_r)
        yearly_values.append(value)
    final_values.append(value)
    trajectories.append(yearly_values)

final_values = np.array(final_values)
trajectories = np.array(trajectories)

# --- Statistics ---
median = np.median(final_values)
p10 = np.percentile(final_values, 10)
p90 = np.percentile(final_values, 90)
mean = np.mean(final_values)

# --- Print Summary ---
actual_investment = initial_contribution + (annual_contribution * years)
print(f"Actual Investment: ${actual_investment:,.0f}")
print(f"Initial Contribution: ${initial_contribution:,.0f}")
print(f"Annual Contribution: ${annual_contribution:,.0f}")
print(f"Years: {years}")
print(f"Expected CAGR: {mean_return * 100:.2f}%")
print(f"10th Percentile (pessimistic): ${p10:,.0f}")
print(f"90th Percentile (optimistic): ${p90:,.0f}")
print(f"Median Value after {years} years: ${median:,.0f}")
print(f"Mean Value: ${mean:,.0f}")


# --- Plot ---
plt.figure(figsize=(12, 6))
for traj in trajectories[:100]:
    plt.plot(range(years + 1), traj, color='gray', alpha=0.3)

plt.plot(range(years + 1), np.median(trajectories, axis=0), color='red', label='Median Trajectory', linewidth=2)
plt.title("Monte Carlo Simulation of TFSA Growth\n(S&P 500 Investment, 30 Years, $66k + $7k/year)")
plt.xlabel("Years")
plt.ylabel("Portfolio Value (CAD)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

median, p10, p90, mean
