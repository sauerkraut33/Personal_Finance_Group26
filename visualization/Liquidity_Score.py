from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = Path(__file__).resolve().parent / "../modelling/personal_finance_model_input.csv"
list = pd.read_csv(file_path)

import numpy as np
import matplotlib.pyplot as plt

def scatter_with_trend(df, x, y, method="spearman"):
    d = df[[x, y]].dropna()

    r = d[x].corr(d[y], method=method)

    m, b = np.polyfit(d[x].to_numpy(), d[y].to_numpy(), 1)

    xs = np.linspace(d[x].min(), d[x].max(), 200)
    ys = m * xs + b

    plt.figure(figsize=(6,4))

    plt.scatter(d[x], d[y], alpha=0.5, color="gray")

    plt.plot(xs, ys, color="red", linewidth=2, label="Regression Line")

    plt.xlabel(x)
    plt.ylabel(y)

    plt.title(f"{y} vs {x}  ({method} r = {r:.3f})")

    equation_text = f"{y} = {m:.3f} × {x} + {b:.3f}"
    plt.text(
        0.05, 0.95, equation_text,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.7)
    )

    plt.tight_layout()
    plt.show()
# Liquidity Score Plot with Net Worth
scatter_with_trend(list, "PFTENUR", "Liquidity")