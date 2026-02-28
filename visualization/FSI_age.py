from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------
# Load data
# ----------------------------
project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "modelling" / "personal_finance_model_input.csv"
df = pd.read_csv(file_path)

# Convert to numeric
df["PAGEMIEG"] = pd.to_numeric(df["PAGEMIEG"], errors="coerce")
df["FSI"] = pd.to_numeric(df["FSI"], errors="coerce")

df = df.dropna(subset=["PAGEMIEG", "FSI"])

# Enforce FSI cap
FSI_CAP = 1000
df = df[df["FSI"] <= FSI_CAP]

# ----------------------------
# Correlation strength classifier
# ----------------------------
def corr_strength_label(value):
    a = abs(value)
    if a < 0.2:
        return "negligible"
    elif a < 0.4:
        return "weak"
    elif a < 0.6:
        return "moderate"
    elif a < 0.8:
        return "strong"
    else:
        return "very strong"

# ----------------------------
# Compute correlations
# ----------------------------
x = df["PAGEMIEG"]
y = df["FSI"]

pearson = stats.pearsonr(x, y)[0]
spearman = stats.spearmanr(x, y)[0]
r2 = stats.linregress(x, y).rvalue ** 2

strength = corr_strength_label(spearman)

# ----------------------------
# Group mean trend
# ----------------------------
grouped = df.groupby("PAGEMIEG")["FSI"].mean()

plt.figure(figsize=(8,6))
plt.plot(grouped.index, grouped.values, marker="o")

plt.xlabel("Age Group")
plt.ylabel("FSI")
plt.title("FSI vs Age Group (FSI ≤ 1000)")

textstr = (
    f"Pearson = {round(pearson,4)}\n"
    f"Spearman = {round(spearman,4)} ({strength})\n"
    f"R² (linear) = {round(r2,4)}"
)

plt.text(
    0.05, 0.95, textstr,
    transform=plt.gca().transAxes,
    fontsize=10,
    va="top",
    bbox=dict(boxstyle="round", alpha=0.3)
)

plt.tight_layout()
plt.show()
