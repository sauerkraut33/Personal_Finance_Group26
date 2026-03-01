from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = Path(__file__).resolve().parent / "../modelling/personal_finance_model_input.csv"
df = pd.read_csv(file_path)
import matplotlib.pyplot as plt

def age_trend_plot(df):
    d = df[["PAGEMIEG", "FSI"]].dropna()

    # 计算各年龄组平均值
    means = d.groupby("PAGEMIEG")["FSI"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(means.index, means.values, marker="o")
    plt.xlabel("Age Group")
    plt.ylabel("Mean FSI")
    plt.title("FSI Across Age Groups")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

age_trend_plot(df)

def age_trend_plot(df):
    d = df[["PAGEMIEG", "DTI"]].dropna()

    # 计算各年龄组平均值
    means = d.groupby("PAGEMIEG")["DTI"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(means.index, means.values, marker="o")
    plt.xlabel("Age Group")
    plt.ylabel("Mean DTI")
    plt.title("DTI Across Age Groups")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

age_trend_plot(df)

def age_trend_plot(df):
    d = df[["PAGEMIEG", "Liquidity"]].dropna()

    # 计算各年龄组平均值
    means = d.groupby("PAGEMIEG")["Liquidity"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(means.index, means.values, marker="o")
    plt.xlabel("Age Group")
    plt.ylabel("Mean Liquidity")
    plt.title("Liquidity Across Age Groups")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

age_trend_plot(df)

def age_trend_plot(df):
    d = df[["PAGEMIEG", "Behavior Score"]].dropna()

    # 计算各年龄组平均值
    means = d.groupby("PAGEMIEG")["Behavior Score"].mean()

    plt.figure(figsize=(6,4))
    plt.plot(means.index, means.values, marker="o")
    plt.xlabel("Age Group")
    plt.ylabel("Mean Behavior Score")
    plt.title("Behavior Score Across Age Groups")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

age_trend_plot(df)