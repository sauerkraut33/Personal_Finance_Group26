# data_processing/rank_correlation.py

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# load processed data
file_path = Path(__file__).resolve().parent / "personal_finance_cleaned.csv"
df = pd.read_csv(file_path)


# column code → actual descriptive name
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
}


# ensure numeric ranking works
df_numeric = df.apply(pd.to_numeric, errors="coerce")

# full rank correlation matrix
corr_matrix = df_numeric.corr(method="spearman")


# plot heatmap
plt.figure(figsize=(12, 10))
plt.imshow(corr_matrix, aspect="auto")
plt.colorbar(label="Spearman Rank Correlation")


# use descriptive labels on both axes
names = [labels.get(col, col) for col in corr_matrix.columns]

plt.xticks(range(len(names)), names, rotation=90)
plt.yticks(range(len(names)), names)

plt.title("Rank Correlation Matrix")
plt.tight_layout()
plt.show()