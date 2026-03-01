import pandas as pd

file_path = r"C:\Users\Wendy\Desktop\personal_finance_dataset.xlsx"

df = pd.read_excel(
    file_path,
    sheet_name="datathon_finance",
    engine="openpyxl"
)

print(df.shape)
print(df.head())
