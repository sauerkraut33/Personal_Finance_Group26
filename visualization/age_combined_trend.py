from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Load data
# ----------------------------
project_root = Path(__file__).resolve().parents[1]
file_path = project_root / "modelling" / "personal_finance_model_input.csv"
df = pd.read_csv(file_path)

# ----------------------------
# Columns needed
# ----------------------------
cols = [
    "PAGEMIEG",          # Age Group
    "DTI",
    "FSI",
    "Liquidity",
    "Behavior Score",
    "PWDPRMOR",          # Mortgage debt
    "PWDSLOAN",          # Student loan debt
    "PWDSTCRD",          # Credit card debt
    "PWDSTLOC",          # LOC debt
    "PEFATINC",          # After-tax income
    "PWASTDEP",          # Bank deposits
    "PWATFS",            # TFSA
]

for c in cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["PAGEMIEG", "DTI", "FSI", "PEFATINC"])

# ----------------------------
# Optional caps (edit as needed)
# ----------------------------
FSI_CAP = 1000
DTI_CAP = 1000
df = df[(df["FSI"] <= FSI_CAP) & (df["DTI"] <= DTI_CAP)]

# ----------------------------
# Derived drivers
# ----------------------------
df["Total Debt"] = df["PWDPRMOR"] + df["PWDSLOAN"] + df["PWDSTCRD"] + df["PWDSTLOC"]
df["Deposits + TFSA"] = df["PWASTDEP"] + df["PWATFS"]  # liquidity numerator

# ----------------------------
# Group means by Age Group
# ----------------------------
grouped = df.groupby("PAGEMIEG").mean(numeric_only=True)

# Variables to plot (data columns)
variables = [
    "DTI",
    "FSI",
    "Total Debt",
    "PEFATINC",
    "Deposits + TFSA",
    "Behavior Score",
]

# Display names for legend
label_map = {
    "DTI": "DTI",
    "FSI": "FSI",
    "Total Debt": "Total Debt",
    "PEFATINC": "After-Tax Income",
    "Deposits + TFSA": "Deposits + TFSA",
    "Behavior Score": "Behavior Score",
}

# ----------------------------
# Standardize each variable (z-score across age groups)
# ----------------------------
standardized = grouped[variables].apply(
    lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) != 0 else (x - x.mean())
)

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(10, 7))

for col in variables:
    display_name = label_map[col]

    if col == "FSI":
        # Highlight FSI
        plt.plot(
            standardized.index,
            standardized[col],
            marker="o",
            linewidth=3,
            markersize=8,
            color="black",
            label=display_name
        )
    else:
        plt.plot(
            standardized.index,
            standardized[col],
            marker="o",
            linewidth=1.5,
            markersize=5,
            alpha=0.6,
            label=display_name
        )

plt.axhline(0, linestyle="--", linewidth=1)

plt.xlabel("Age Group")
plt.ylabel("Standardized Mean (z-score)")
plt.title("DTI, FSI, and Key Drivers Across Age Groups (Standardized Trends)")

plt.legend()
plt.tight_layout()
plt.show()
