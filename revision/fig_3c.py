import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# make a df out of the data
data = {
    "Model": ["BLAST", "SCREEN", "Squidly 15B", "Squidly 3B"],
    "Precision": [0.6948051948051948, 0.69004, 0.8627, 0.8129],
    "Recall": [0.24883720930232558, 0.50386, 0.5288, 0.6058],
    "F1": [0.3664383561643836, 0.57934, 0.6557, 0.6942],
}

df = pd.DataFrame(data)

# plot the data as a bar plot with stacked bars... with precision as the background, f1 in the middle and recall in the foreground
plt.figure(figsize=(5, 4))
bar_width = 0.75
x = np.arange(len(df))
plt.bar(x, df["Precision"], bar_width, label="Precision", color="grey")
plt.bar(x, df["F1"], bar_width, label="F1 Score", color="blue", alpha=1.0)
plt.bar(x, df["Recall"], bar_width, label="Recall", color="lightblue", alpha=0.9)
plt.xlabel("Tool")
plt.ylabel("Scores")
plt.title("Precision, F1 Score, and Recall by Model")
plt.xticks(x, df["Model"])
# create a custom legend
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="grey", label="Precision"),
    Patch(facecolor="blue", label="F1 Score"),
    Patch(facecolor="lightblue", label="Recall"),
]
plt.legend(handles=legend_elements, loc="upper right")
plt.tight_layout()
plt.savefig("Model_precision_F1_recall_CATALODB_fig3C.svg")
