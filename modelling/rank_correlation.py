from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# load data
file_path = Path(__file__).resolve().parent / "personal_finance_model_input.csv"
df = pd.read_csv(file_path)

# descriptive labels
labels = {
    "PAGEMIEG": "Age Group",
    "PATTCRU": "Credit Card Payment",
    "PATTSITC": "COVID Financial Impact",
    "PATTSKP": "Skipped Payments",
    "PEDUCMIE": "Education Level",
    "PEFATINC": "After-Tax Income",
    "PFMTYPG": "Family Type",
    "PFTENUR": "Home Ownership",
    "PLFFPTME": "Work Status 2022",
    "PNBEARG": "Number of Earners",
    "PPVRES": "Province",
    "PWAPRVAL": "Home Value",
    "PWASTDEP": "Bank Deposits",
    "PWATFS": "TFSA Balance",
    "PWDPRMOR": "Mortgage Debt",
    "PWDSLOAN": "Student Loan Debt",
    "PWDSTCRD": "Credit Card Debt",
    "PWDSTLOC": "Line of Credit Debt",
    "PWNETWPG": "Net Worth",
    "DTI": "DTI",
    "Liquidity": "Liquidity",
    "Behavior Score": "Behavior Score",
    "FSI": "FSI",
}

# numeric conversion
df_numeric = df.apply(pd.to_numeric, errors="coerce")

# Spearman rank correlation
corr_matrix = df_numeric.corr(method="spearman")

# custom red (+) to green (-) gradient WITHOUT forcing white at 0
cmap = LinearSegmentedColormap.from_list(
    "blue_red",
    ["blue","white","red"]
)

plt.figure(figsize=(14, 12))
plt.imshow(corr_matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(label="Spearman Rank Correlation")

names = [labels.get(col, col) for col in corr_matrix.columns]

plt.xticks(range(len(names)), names, rotation=90)
plt.yticks(range(len(names)), names)

plt.title("Rank Correlation Matrix (All Columns)")
plt.tight_layout()
plt.show()
