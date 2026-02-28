#FSI
#column1(name: DTI): (PWDPRMOR+PWDSLOAN+PWDSTCRD+PWDSTLOC)/PEFATINC
#column2(name: Liquidity ): (PWASTDEP+PWATFS)/PEFATINC
#column3(name: Behavior Score): g (if PATTSKP ==2 and PATTCRU >1, g==0, otherwise, g==1)
#column4(name: FSI): (w1 = 0.5, w2 = 0.3,w3 = 0.2), FSI = w1(DTI)+w2(Liquidity)+w3(Behavior Score)

# modelling/add_features.py

from pathlib import Path
import numpy as np
import pandas as pd

# --- 1) load cleaned file (it's in data_processing) ---
project_root = Path(__file__).resolve().parents[1]
clean_path = project_root / "data_processing" / "personal_finance_cleaned.csv"
df = pd.read_csv(clean_path)

# --- 2) make sure needed columns are numeric where required ---
num_cols = [
    "PWDPRMOR", "PWDSLOAN", "PWDSTCRD", "PWDSTLOC",
    "PEFATINC", "PWASTDEP", "PWATFS"
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["PATTSKP"] = pd.to_numeric(df["PATTSKP"], errors="coerce")
df["PATTCRU"] = pd.to_numeric(df["PATTCRU"], errors="coerce")

# avoid divide-by-zero
income = df["PEFATINC"].replace(0, np.nan)

# --- 3) add 4 columns ---
df["DTI"] = (df["PWDPRMOR"] + df["PWDSLOAN"] + df["PWDSTCRD"] + df["PWDSTLOC"]) / income
df["Liquidity"] = (df["PWASTDEP"] + df["PWATFS"]) / income

df["Behavior Score"] = np.where(
    (df["PATTSKP"] == 2) & (df["PATTCRU"] > 1),
    0,
    1
)

w1, w2, w3 = 0.5, 0.3, 0.2
df["FSI"] = w1 * df["DTI"] + w2 * df["Liquidity"] + w3 * df["Behavior Score"]

# --- 4) save to csv (in modelling directory) ---
out_path = Path(__file__).resolve().parent / "personal_finance_model_input.csv"
df.to_csv(out_path, index=False)

print("Saved to:", out_path)
print("Rows:", len(df))
print("New columns added:", ["DTI", "Liquidity", "Behavior Score", "FSI"])