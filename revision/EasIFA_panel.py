import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support

filter_overlap = True


def calculate_stats(df, id_col, true_col, pred_col, seq_col):
    # Check the agreement
    predictions = []
    true = []
    missing = 0
    for seq_label, res_sq, res_pred, seq in df[
        [id_col, true_col, pred_col, seq_col]
    ].values:
        res_sq = res_sq.split("|")
        if not res_pred or not isinstance(res_pred, str):
            res_pred = ""
        res_pred = res_pred.split("|")
        if len(res_pred) > 0:
            try:
                chosen_res_seq = [int(i) for i in res_pred]
            except:
                chosen_res_seq = []
                missing += 1
        res_sq = [int(i) for i in res_sq]
        for pos in range(0, len(seq)):
            if pos in res_sq:
                true.append(1)
            else:
                true.append(0)
            if pos in chosen_res_seq:
                predictions.append(1)
            else:
                predictions.append(0)
    precision, recall, f1, support = precision_recall_fscore_support(true, predictions)
    false_pos_rate = sum((1 - np.array(true)) * np.array(predictions)) / sum(
        1 - np.array(true)
    )
    return precision[1], recall[1], f1[1], support[1], false_pos_rate


def add_value_labels(bars):
    for bar in bars:
        # if bar is squidly, skip
        # if bar.get_x() >= 1.625:
        #    continue
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )


def plot_grouped_bars(
    labels,
    recalls,
    FPRs,
    precisions,
    title,
    filename,
    squidly_recalls=None,
    squidly_FPRs=None,
    squidly_precisions=None,
):
    x = np.arange(len(labels))
    width = 0.25  # narrower so three bars fit cleanly
    fig, ax = plt.subplots()

    bars1 = ax.bar(x - width, recalls, width, label="Recall", capsize=5)
    bars2 = ax.bar(x, FPRs, width, label="FPR", capsize=5)
    bars3 = ax.bar(x + width, precisions, width, label="Precision", capsize=5)

    ax.set_ylabel("Percentage (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    # Overlay dots for Squidly replicates if provided
    squidly_index = labels.index("Squidly")
    if squidly_recalls is not None:
        ax.scatter(
            [x[squidly_index] - width] * len(squidly_recalls),
            squidly_recalls,
            color="black",
            marker="o",
            s=20,
            alpha=0.7,
        )
    if squidly_FPRs is not None:
        ax.scatter(
            [x[squidly_index]] * len(squidly_FPRs),
            squidly_FPRs,
            color="black",
            marker="o",
            s=20,
            alpha=0.7,
        )
    if squidly_precisions is not None:
        ax.scatter(
            [x[squidly_index] + width] * len(squidly_precisions),
            squidly_precisions,
            color="black",
            marker="o",
            s=20,
            alpha=0.7,
        )

    fig.tight_layout()
    plt.savefig(filename, format="svg")

# Sorry it's messy!
def main():

    # load the EasIFA_benchmark file from ensemble
    ensemble_df = pd.read_pickle("15B_EasIFA_test_input_fasta_squidly_ensemble.pkl")
    ensemble_df = pd.read_pickle("3B_EasIFA_test_input_fasta_squidly_ensemble.pkl")
    print(ensemble_df.head())
    print(ensemble_df.columns)
    print(ensemble_df)

    benchmark_df = (
        "/Users/silicon_palace/dev/squidly/EasIFA_benchmark_catalytic_only.csv"
    )
    benchmark_df = pd.read_csv(benchmark_df)

    ensemble_CR_posis = []
    for i, row in ensemble_df.iterrows():
        if row["Squidly_Ensemble_Residues"] == "":
            ensemble_CR_posis.append([])
        else:
            ensemble_CR_posis.append(
                [int(x) for x in row["Squidly_Ensemble_Residues"].split("|")]
            )

    ensemble_df["ensemble_CR_posis_ensemble_final_y"] = ensemble_CR_posis
    ensemble_df["entry"] = ensemble_df.index
    benchmark_df = benchmark_df.merge(
        ensemble_df[
            ["entry", "ensemble_CR_posis_ensemble_final_y", "Squidly_Ensemble_Residues"]
        ],
        left_on="Entry",
        right_on="entry",
        how="left",
    )

    # get the EasIFA results into the | format
    EasIFA_CR_posis = []
    for i, row in benchmark_df.iterrows():
        if pd.isna(row["predict_active_label"]):
            EasIFA_CR_posis.append("")
        else:
            pred_labels = []
            prediction = (
                row["predict_active_label"]
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
                .split(",")
            )
            for j, label in enumerate(prediction):
                if int(label) == 2:
                    pred_labels.append(j)
            pred_labels = "|".join([str(x) for x in pred_labels])
            EasIFA_CR_posis.append(pred_labels)
    benchmark_df["EasIFA_CR_posis"] = EasIFA_CR_posis
    print(
        benchmark_df[["Entry", "EasIFA_CR_posis", "ensemble_CR_posis_ensemble_final_y"]]
    )

    EasIFA_precision, EasIFA_recall, EasIFA_f1, EasIFA_support, EasIFA_fpr = (
        calculate_stats(
            benchmark_df, "label", "true_CR_labels", "EasIFA_CR_posis", "Sequence"
        )
    )
    print(
        f"EasIFA precision: {EasIFA_precision*100:.2f}%, recall: {EasIFA_recall*100:.2f}%, f1: {EasIFA_f1*100:.2f}%, support: {EasIFA_support}, FPR: {EasIFA_fpr*100:.2f}%"
    )

    precision, recall, f1, support, fpr = calculate_stats(
        benchmark_df, "label", "true_CR_labels", "Squidly_Ensemble_Residues", "Sequence"
    )
    print(
        f"Ensemble precision: {precision*100:.2f}%, recall: {recall*100:.2f}%, f1: {f1*100:.2f}%, support: {support}, FPR: {fpr*100:.2f}%"
    )

    # plot ensemble results with AEGAN and EasIFA
    labels = ["AEGAN", "EasIFA", "Squidly"]
    FPRs = [8.65, EasIFA_fpr * 100, fpr * 100]
    recalls = [91.78, EasIFA_recall * 100, recall * 100]
    precisions = [7.85, EasIFA_precision * 100, precision * 100]
    plot_grouped_bars(
        labels,
        recalls,
        FPRs,
        precisions,
        "Recall, FPR, and Precision by Model",
        (
            "EasIFA_benchmark_results_summary_ensemble_filtered_overlap.svg"
            if filter_overlap
            else "EasIFA_benchmark_results_summary_ensemble_no_filtering.svg"
        ),
    )

    ECs = []
    for i, row in benchmark_df.iterrows():
        if row["true_CR_labels"] != "":
            if pd.isna(row["ec"]):
                continue
            ecs = row["ec"].split(";")
            ecs_tier1 = set()
            for ec in ecs:
                tier1 = ec.split(".")[0]
                ecs_tier1.add(tier1)
            ECs.extend(list(ecs_tier1))
    EC_series = pd.Series(ECs)
    EC_counts = EC_series.value_counts().sort_index()
    plt.figure(figsize=(8, 8))
    colors = plt.cm.tab20.colors  # Use a colormap with enough distinct colors
    EC_counts.plot.pie(
        autopct=lambda x: "{:.0f}".format(x * EC_counts.sum() / 100),
        startangle=140,
        colors=colors,
    )
    plt.title("EC numbers (tier 1) of benchmark subset")
    plt.ylabel("")  # Hide y-label for better aesthetics
    plt.tight_layout()
    # plt.savefig("EasIFA_benchmark_EC_distribution_pie_chart.svg", format='svg')
    plt.close()

    print(benchmark_df.columns)
    diff_data = []
    for i, row in benchmark_df.iterrows():
        if row["true_CR_labels"] == "":
            continue
        true_labels = set(row["true_CR_labels_list"])
        if row["EasIFA_CR_posis"] == "":
            EasIFA_pred_labels = set()
        else:
            EasIFA_pred_labels = set(
                [int(x) for x in row["EasIFA_CR_posis"].split("|")]
            )
        EasIFA_TP = len(true_labels.intersection(EasIFA_pred_labels))
        EasIFA_FP = len(EasIFA_pred_labels - true_labels)

        Squidly_pred_labels = set(
            [int(x) for x in row["ensemble_CR_posis_ensemble_final_y"]]
        )
        Squidly_TP = len(true_labels.intersection(Squidly_pred_labels))
        Squidly_FP = len(Squidly_pred_labels - true_labels)

        diff_TP = Squidly_TP - EasIFA_TP
        diff_FP = Squidly_FP - EasIFA_FP

        binding_residues = set()
        site_labels = eval(row["site_labels"])
        site_types = eval(row["site_types"])
        for j, sites in enumerate(site_labels):
            for site in sites:
                if site_types[j] == 0:
                    binding_residues.add(site - 1)
        Squidly_FP_binding = len(Squidly_pred_labels.intersection(binding_residues))

        diff_data.append(
            {
                "Entry": row["Entry"],
                "ec": row["ec"],
                "predictions_EasIFA": row["EasIFA_CR_posis"],
                "predictions_Squidly": row["ensemble_CR_posis_ensemble_final_y"],
                "diff_TP": diff_TP,
                "diff_FP": diff_FP,
                "Squidly_TP": Squidly_TP,
                "EasIFA_TP": EasIFA_TP,
                "Squidly_FP": Squidly_FP,
                "EasIFA_FP": EasIFA_FP,
                "Squidly_FP_binding": Squidly_FP_binding,
            }
        )

    diff_df = pd.DataFrame(diff_data)

    print(diff_df)
    diff_df.to_csv(
        "EasIFA_benchmark_per_sequence_differential_prediction.csv", index=False
    )

    def get_first_ec_tier(ec):
        if pd.isna(ec):
            return "Unknown"
        return ec.split(";")[0].split(".")[0]

    diff_df["first_ec_tier"] = diff_df["ec"].apply(get_first_ec_tier)
    diff_df = diff_df.sort_values(by=["first_ec_tier", "Entry"])
    plt.figure(figsize=(12, 8))
    x = np.arange(len(diff_df))
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.scatter(
        x,
        diff_df["diff_TP"],
        color="blue",
        label="Difference in True Positives (Squidly - EasIFA)",
        alpha=0.7,
    )
    plt.scatter(
        x,
        diff_df["diff_FP"],
        color="red",
        label="Difference in False Positives (Squidly - EasIFA)",
        alpha=0.7,
    )
    plt.xticks(x, diff_df["ec"], rotation=90)
    plt.xlabel("Sequences (ordered by EC number)")
    plt.ylabel("Difference in Counts")
    plt.title("Per-sequence Differential Prediction: Squidly vs EasIFA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "EasIFA_benchmark_per_sequence_differential_prediction.svg", format="svg"
    )
    plt.close()

    # get the benchmark_df
    benchmark_df.to_csv("EasIFA_benchmark.csv")

    # get the benchmark with the 66 catalytic residues
    benchmark_df = benchmark_df[benchmark_df["true_CR_labels"] != ""]
    # Figure 3C, data read from std output of tools
    data = {
        "Model": ["BLAST", "SCREEN", "Squidly 15B", "Squidly 3B"],
        "Precision": [0.6948051948051948, 0.69004, 0.8627, 0.8129],
        "Recall": [0.24883720930232558, 0.50386, 0.5288, 0.6058],
        "F1": [0.3664383561643836, 0.57934, 0.6557, 0.6942],
    }

    df = pd.DataFrame(data)

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
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="grey", label="Precision"),
        Patch(facecolor="blue", label="F1 Score"),
        Patch(facecolor="lightblue", label="Recall"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig("Model_precision_F1_recall_CATALODB_fig3C.svg")


if __name__ == "__main__":
    main()
