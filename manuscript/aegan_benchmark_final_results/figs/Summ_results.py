import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

# Load the data glob the files
file_paths = glob.glob("/scratch/project/squid/code_modular/aegan_bench_final_results/*.txt")

# Initialize a dictionary to store metrics data
metrics_data = {}

# Function to extract metrics from a single file
def parse_file(file_path):
    metrics = {
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1": [],
        "best_threshold": [],
        "AUC_ROC": [],
        "AUC_PR": []
    }
    with open(file_path, 'r') as file:
        for line in file:
            for metric in metrics.keys():
                if line.startswith(metric + ":"):
                    metrics[metric].append(float(line.split(":")[1].strip()))
    return metrics

# Process each file and collect data
for file_path in file_paths:
    file_name = os.path.basename(file_path)
    metrics = parse_file(file_path)
    metrics_data[file_name] = metrics

# Create a DataFrame for each metric
metric_dfs = {}
for metric in ["Accuracy", "Precision", "Recall", "F1", "best_threshold", "AUC_ROC", "AUC_PR"]:
    rows = []
    for file_name, metrics in metrics_data.items():
        values = np.array(metrics[metric])
        row = {
            "File": file_name,
            "Mean": values.mean(),
            "Std": values.std(),
            "Max": values.max(),
            "Min": values.min()
        }
        rows.append(row)
    metric_dfs[metric] = pd.DataFrame(rows)

# Define local output directory
output_dir = "./summary/"
os.makedirs(output_dir, exist_ok=True)

# Save DataFrames to CSV
for metric, df in metric_dfs.items():
    df.to_csv(f"{output_dir}{metric}_summary.csv", index=False)

f1_df = metric_dfs["F1"]

# map the file names as so:
# big_1.txt -> Scheme 1 (ESM 15B)
# big_2.txt -> Scheme 2 (ESM 15B)
# big_3.txt -> Scheme 3 (ESM 15B)
# med_1.txt -> Scheme 1 (ESM 3B)
# med_2.txt -> Scheme 2 (ESM 3B)
# med_3.txt -> Scheme 3 (ESM 3B)
# raw.txt -> Raw LSTM (ESM 3B)

# map the file names to the model names
model_map = {
    "big_1.txt": "S1 (ESM 15B)",
    "big_2.txt": "S2 (ESM 15B)",
    "big_3.txt": "S3 (ESM 15B)",
    "med_1.txt": "S1 (ESM 3B)",
    "med_2.txt": "S2 (ESM 3B)",
    "med_3.txt": "S3 (ESM 3B)",
    "med_raw_LSTM.txt": "LSTM (ESM 3B)"
}

# now map the file names to the model names
f1_df["File"] = f1_df["File"].map(model_map)

# drop the Max and Min cols
f1_df = f1_df.drop(columns=["Max", "Min"])
# multiple the mean and std by 100 to get percentage
f1_df[["Mean", "Std"]] *= 100
shen_model = {"File": "Shen et al. (AEGAN)", "Mean": 83.22, "Std": 0.40}
shen_df = pd.DataFrame([shen_model])
f1_df = pd.concat([f1_df, shen_df], ignore_index=True)

print(f1_df)

# Sort the DataFrame by Mean F1 score in descending order
f1_df = f1_df.sort_values(by="Mean", ascending=False)

# Plot the F1 mean and standard deviation as a bar plot
fig, ax = plt.subplots(figsize=(8, 6))  # Smaller, more compact figure
colors = ["#AEC6CF"] * len(f1_df)  # Default pastel blue for all bars
# make top two bars light blue
colors[0:2] = ["#87CEEB"]*2
colors[2] = "#AEC6CF"  # Pastel orange for the shen
colors[3:5] = ["#FFB6C1"]*2  # Pastel pink for med models.
colors[5] = "#E6E6FA" # Lavender for the raw model
colors[6:8] = ["#77DD77"]*2 # Gross green for the worst models

print(colors)

bar_width = 0.6  # Skinnier bars
x_positions = range(len(f1_df))  # X positions for bars

# Bar plot with error bars
bars = ax.bar(
    x_positions, f1_df["Mean"], yerr=f1_df["Std"], color=colors, capsize=5, edgecolor="black", alpha=0.8, width=bar_width
)

for bar, mean, std in zip(bars, f1_df["Mean"], f1_df["Std"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.3,
        f"{mean:.2f}±{std:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        rotation=12  # Rotate the text 45 degrees
    )
    
# Plot individual results as dots along the bars
#for i, file in enumerate(f1_df["File"]):
#    if file != "Shen et al. (AEGAN)":  # Skip Shen et al., as it's an aggregate
#        individual_values = metrics_data[file]
#        individual_values = individual_values['F1']
#        x_jitter = [i + (np.random.uniform(-0.2, 0.2)) for _ in individual_values]  # Add jitter for spread
#        ax.scatter(
#            x_jitter, individual_values, color="red", alpha=0.7, s=10, label=None if i > 0 else "Individual Results"
#        )

# Customize plot appearance
ax.set_ylabel("F1 Score (%)", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
ax.set_title("Uni3175 Test Results", fontsize=14)
ax.set_ylim(40, 90)  # Adjust based on expected F1 values
ax.set_xticks(x_positions)
ax.set_xticklabels(f1_df["File"], rotation=45, ha="right", fontsize=10)

# Adjust margins to fit all labels
plt.subplots_adjust(bottom=0.3, top=0.9)

# Add a legend for individual results
#ax.legend(["Individual Results"], loc="lower right", fontsize=10, frameon=False)

# Save the plot as an SVG file
output_path = f"{output_dir}F1_score_comparison_final.png"
plt.savefig(output_path, format="png")
plt.show()
