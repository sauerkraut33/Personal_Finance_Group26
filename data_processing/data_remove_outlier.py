# data_processing/data_remove_outlier.py

from pathlib import Path
import re
import pandas as pd
from data_processing.data_load import df


def to_number(x):
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = s.replace("$", "").replace(",", "").replace(" ", "")
    s = re.sub(r"[^0-9\.\-]", "", s)

    if s in {"", "-", ".", "-."}:
        return pd.NA

    val = float(s)
    return -val if neg else val


# 1) copy + clean net worth to numeric
df_clean = df.copy()
df_clean["PWNETWPG"] = df_clean["PWNETWPG"].apply(to_number)

# 2) rearrange column order: move PWNETWPG to the front (keep others same order)
cols = list(df_clean.columns)
cols.remove("PWNETWPG")
df_clean = df_clean[["PWNETWPG"] + cols]

# 3) remove IQR outliers based on PWNETWPG (drop entire row)
s = df_clean["PWNETWPG"].dropna()
Q1 = s.quantile(0.25)
Q3 = s.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df_clean[
    df_clean["PWNETWPG"].isna() | df_clean["PWNETWPG"].between(lower, upper, inclusive="both")
].copy()

# 4) save processed data to a new file
output_path = Path(__file__).resolve().parent / "personal_finance_cleaned.csv"
df_clean.to_csv(output_path, index=False)

# 5) print number of rows left
print("Rows left:", len(df_clean))
print("Saved to:", output_path)