from pathlib import Path

import pandas as pd

file_path = Path(__file__).resolve().parent / "personal_finance_model_input.csv"
df = pd.read_csv(file_path)
print(df[["DTI", "Liquidity", "Behavior Score"]].corr())


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X = df[["DTI","Liquidity","Behavior Score"]].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
pca.fit(X_scaled)


print(pca.explained_variance_ratio_)