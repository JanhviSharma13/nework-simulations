import pandas as pd
import matplotlib.pyplot as plt

# Load and clean
df = pd.read_csv("dql_per_metrics1.csv")
df.columns = df.columns.str.strip()  # Clean up any accidental whitespace

# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["Episode"], df["Reward"], color="steelblue", linewidth=2, label="DQL-PER")

# Styling
plt.title("Episode vs. Total Reward (DQL-PER)", fontsize=14)
plt.xlabel("Episode", fontsize=12)
plt.ylabel("Total Reward", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()

