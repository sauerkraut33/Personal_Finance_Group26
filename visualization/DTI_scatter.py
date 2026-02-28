from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------
# Load CSV (DTI.py under visualization/)
# ----------------------------
project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "modelling" / "personal_finance_model_input.csv"
df = pd.read_csv(file_path)

# ----------------------------
# Settings
# ----------------------------
Y_COL = "DTI"
X_COLS = [
    ("PAGEMIEG", "Age Group"),                 # expected negative
    ("PATTCRU", "Credit Card Payment"),        # expected negative
    ("PLFFPTME", "Work Status 2022"),          # expected negative
    ("PNBEARG", "Number of Earners"),          # expected positive
]

DTI_CAP = 1000  # set to None if you don't want filtering

# ----------------------------
# Helpers
# ----------------------------
def corr_strength_label(value: float) -> str:
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

def scatter_with_regression(x, y, x_label, y_label="DTI", y_cap=None):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    pearson_corr = stats.pearsonr(x, y)[0]
    spearman_corr = stats.spearmanr(x, y)[0]

    x_vals = np.linspace(x.min(), x.max(), 200)
    y_vals = slope * x_vals + intercept

    # round to 4 decimals
    slope_r = round(slope, 4)
    intercept_r = round(intercept, 4)
    r2_r = round(r_value**2, 4)
    pearson_r = round(pearson_corr, 4)
    spearman_r = round(spearman_corr, 4)

    pearson_strength = corr_strength_label(pearson_corr)
    spearman_strength = corr_strength_label(spearman_corr)

    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, alpha=0.4)
    plt.plot(x_vals, y_vals)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    cap_text = f" ({y_label} ≤ {y_cap})" if y_cap is not None else ""
    plt.title(f"{y_label} vs {x_label}{cap_text}")

    textstr = (
        f"Slope = {slope_r}\n"
        f"Intercept = {intercept_r}\n"
        f"R² = {r2_r}\n"
        f"Pearson = {pearson_r} ({pearson_strength})\n"
        f"Spearman = {spearman_r} ({spearman_strength})\n"
        f"Primary strength: {spearman_strength}"
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

# ----------------------------
# Clean + numeric conversion
# ----------------------------
needed_cols = [Y_COL] + [c for c, _ in X_COLS]
for col in needed_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=needed_cols)

if DTI_CAP is not None:
    df = df[df[Y_COL] <= DTI_CAP]

# ----------------------------
# Make 4 graphs
# ----------------------------
for x_col, x_label in X_COLS:
    scatter_with_regression(df[x_col], df[Y_COL], x_label=x_label, y_label="DTI", y_cap=DTI_CAP)
