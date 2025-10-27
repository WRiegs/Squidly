import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import seaborn as sns

plt.rcParams["svg.fonttype"] = "none"
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


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


def extract_catalytic_residues(active_site_str):
    if pd.isna(active_site_str):
        return []
    sites = []
    parts = active_site_str.split(";")
    for part in parts:
        part = part.strip()
        if part.startswith("ACT_SITE"):
            try:
                site = int(part.split()[1]) - 1  # convert to 0-based index
                sites.append(site)
            except (IndexError, ValueError):
                continue
    return sites


def main():
    results_3B = pd.read_pickle("3B_CATALODB_input_fasta_squidly_ensemble.pkl")
    results_3B["id"] = [str(i) for i in results_3B.index]
    results_15B = pd.read_pickle("15B_CATALODB_input_fasta_squidly_ensemble.pkl")
    results_15B["id"] = [str(i) for i in results_15B.index]
    metadata = pd.read_csv("All_metadata.tsv", sep="\t")
    merged_3B = pd.merge(results_3B, metadata, left_on="id", right_on="Entry")
    merged_15B = pd.merge(results_15B, metadata, left_on="id", right_on="Entry")
    dfs = []

    for df in [merged_3B, merged_15B]:
        df["EC_1_cl_list"] = (
            df["EC number"]
            .str.split(";")
            .apply(lambda x: [ec.split(".")[0] for ec in x if isinstance(ec, str)])
        )
        df["true_CR_indices"] = df["Active site"].apply(extract_catalytic_residues)
        df["true_CR_labels"] = [
            "|".join([str(i) for i in row["true_CR_indices"]])
            for _, row in df.iterrows()
        ]
        df["true_CR_binary"] = df.apply(
            lambda row: [
                1 if i in row["true_CR_indices"] else 0
                for i in range(len(row["Sequence"]))
            ],
            axis=1,
        )
        df["Squidly_Ensemble_Residues"] = df["Squidly_Ensemble_Residues"].fillna("")
        df["predicted_CR_binary"] = df.apply(
            lambda row: [
                (
                    1
                    if i
                    in [
                        int(j)
                        for j in row["Squidly_Ensemble_Residues"].split("|")
                        if len(row["Squidly_Ensemble_Residues"]) > 1
                    ]
                    else 0
                )
                for i in range(len(row["Sequence"]))
            ],
            axis=1,
        )
        precision, recall, f1, support, fpr = calculate_stats(
            df, "id", "true_CR_labels", "Squidly_Ensemble_Residues", "Sequence"
        )
        print(
            f"Ensemble precision: {precision*100:.2f}%, recall: {recall*100:.2f}%, f1: {f1*100:.2f}%, support: {support}, FPR: {fpr*100:.2f}%"
        )

        # count how many sequences have at least one true positive
        for i, row in df.iterrows():
            # check if any of true_CR_labels == Squidly_Ensemble_Residues
            true_set = set(row["true_CR_indices"])
            pred_set = set(
                [
                    int(j)
                    for j in row["Squidly_Ensemble_Residues"].split("|")
                    if len(row["Squidly_Ensemble_Residues"]) > 1
                ]
            )
            intersection = true_set.intersection(pred_set)
            df.at[i, "has_true_positive"] = 1 if len(intersection) > 0 else 0
        num_with_true_positive = df["has_true_positive"].sum()
        print(
            f"Number of sequences with at least one true positive: {num_with_true_positive} out of {len(df)}"
        )

        for i, row in df.iterrows():
            df.at[i, "mean"] = [float(x) for x in row["mean"]]
            df.at[i, "variance"] = [float(x) for x in row["variance"]]

        cols = defaultdict(list)
        unc_df = df
        print(unc_df.columns)
        for mean_prob in [
            0.1,
            0.15,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.7,
            0.75,
            0.8,
            0.85,
            0.9,
            0.95,
        ]:
            for vari in [0.05, 0.0725, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]:
                # flatten all the mean and variance lists
                all_means = []
                all_vars = []
                for _, row in unc_df.iterrows():
                    all_means.extend(row["mean"])
                    all_vars.extend(row["variance"])
                all_means = np.array(all_means)
                all_vars = np.array(all_vars)
                preds_prob = 1.0 * all_means > mean_prob
                preds_var = 1.0 * all_vars < vari
                cols[f"m{mean_prob}_v{vari}"] = preds_prob * preds_var

        rows = []
        for c in tqdm(cols):
            c_preds = cols[c]
            true_flat = []
            for true_list in unc_df["true_CR_binary"].values:
                true_flat.extend(true_list)
            precision, recall, f1, support = precision_recall_fscore_support(
                list(true_flat), list(c_preds)
            )
            rows.append(
                [
                    c,
                    c.split("_")[0][1:],
                    c.split("_")[1][1:],
                    precision[1],
                    recall[1],
                    f1[1],
                    support[1],
                ]
            )
        pred_df = pd.DataFrame(
            rows,
            columns=[
                "label",
                "mean_pred",
                "variance",
                "precision",
                "recall",
                "f1",
                "support",
            ],
        )

        plt.rcParams["figure.figsize"] = (8, 5)
        model = "3B" if df.equals(merged_3B) else "15B"
        df_plot = pd.DataFrame(pred_df, columns=["mean_pred", "variance", "f1"])
        pivot = df_plot.pivot(index="mean_pred", columns="variance", values="f1")
        sns.heatmap(pivot, annot=True, cmap="viridis")
        plt.ylabel("Probability cutoff")
        plt.xlabel("Variance cutoff")
        plt.title(f"F1 {model}")
        plt.savefig(f"{model}_F1_CataloDB.svg", bbox_inches="tight")
        plt.close()

        plt.rcParams["figure.figsize"] = (8, 5)
        df_plot = pd.DataFrame(pred_df, columns=["mean_pred", "variance", "precision"])
        pivot = df_plot.pivot(index="mean_pred", columns="variance", values="precision")
        sns.heatmap(pivot, annot=True, cmap="viridis")
        plt.ylabel("Probability cutoff")
        plt.xlabel("Variance cutoff")
        plt.title(f"Precision {model}")
        plt.savefig(f"{model}_precision_CataloDB.svg", bbox_inches="tight")
        plt.close()

        plt.rcParams["figure.figsize"] = (8, 5)
        df_plot = pd.DataFrame(pred_df, columns=["mean_pred", "variance", "recall"])
        pivot = df_plot.pivot(index="mean_pred", columns="variance", values="recall")
        sns.heatmap(pivot, annot=True, cmap="viridis")
        plt.ylabel("Probability cutoff")
        plt.xlabel("Variance cutoff")
        plt.title(f"Recall {model}")
        plt.savefig(f"{model}_recall_CataloDB.svg", bbox_inches="tight")
        plt.close()

        FPR_list = []
        TPR_list = []
        F1_list = []
        PP_list = []
        precisions = []
        recalls = []
        totals_list = []
        for cl in np.arange(1, 7):
            df_cl = df[
                df["EC_1_cl_list"].apply(
                    lambda x: str(cl) in x if isinstance(x, list) else False
                )
            ]
            y_true = np.concatenate(df_cl["true_CR_binary"].values)
            y_pred = np.concatenate(df_cl["predicted_CR_binary"].values)
            y_true_catalytic = [i for i, val in enumerate(y_true) if val == 1]
            y_pred_catalytic = [i for i in y_true_catalytic if y_pred[i] == 1]
            # number of catalytic residues truely annotated
            if len(y_true) != len(y_pred):
                raise ValueError(f"Lengths do not match for EC class {cl}")
            f1 = f1_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            FPR_list.append(fpr)
            TPR_list.append(tpr)
            F1_list.append(f1)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precisions.append(precision)
            recalls.append(recall)
            PP_list.append(len(y_pred_catalytic))
            totals_list.append(len(y_true_catalytic))
        df_ec = pd.DataFrame(
            {
                "EC_class": np.arange(1, 7),
                "FPR": FPR_list,
                "TPR": TPR_list,
                "F1": F1_list,
                "PP": PP_list,
                "Total": totals_list,
                "Precision": precisions,
                "Recall": recalls,
            }
        )
        dfs.append(df_ec)
    df_ec_3B, df_ec_15B = dfs
    print("3B EC-specific results:")
    print(df_ec_3B)
    print("15B EC-specific results:")
    print(df_ec_15B)
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(1, 7)
    order = [3, 2, 1, 4, 5, 6]
    df_ec_3B = df_ec_3B.set_index("EC_class").loc[order].reset_index()
    df_ec_15B = df_ec_15B.set_index("EC_class").loc[order].reset_index()

    x = np.arange(len(order))
    bar_width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(x, df_ec_15B["F1"], bar_width, label="15B", color="skyblue")
    plt.bar(x + bar_width, df_ec_3B["F1"], bar_width, label="3B", color="salmon")
    plt.xlabel("EC Class")
    plt.ylabel("F1 Score")
    plt.title("F1 Scores by EC Class")
    plt.xticks(x + bar_width / 2, order)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig("CataloDB_EC_specific_F1_scores.svg")
    plt.close()

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    order = [3, 2, 1, 4, 5, 6]
    df_ec_3B = df_ec_3B.set_index("EC_class").loc[order].reset_index()
    df_ec_15B = df_ec_15B.set_index("EC_class").loc[order].reset_index()
    x = np.arange(len(order))
    bar_width = 0.35
    plt.figure(figsize=(10, 6))
    plt.bar(
        x + bar_width,
        df_ec_3B["Total"],
        bar_width,
        label="Total Catalytic Residues",
        color="lightgrey",
        alpha=0.5,
    )
    plt.bar(
        x, df_ec_15B["Total"], bar_width, color="lightgrey", alpha=0.5
    )  # just to make the legend work
    plt.bar(
        x,
        df_ec_15B["PP"],
        bar_width,
        label="15B Predicted Catalytic Residues",
        color="skyblue",
    )
    plt.bar(
        x + bar_width,
        df_ec_3B["PP"],
        bar_width,
        label="3B Predicted Catalytic Residues",
        color="salmon",
    )
    plt.xlabel("EC (tier 1)")
    plt.ylabel("# Catalytic Residues")
    plt.title("Predicted vs Total Catalytic Residues by EC")
    plt.xticks(x + 1.5 * bar_width, order)
    plt.legend()
    plt.tight_layout()
    plt.savefig("CataloDB_EC_specific_predicted_vs_total.svg")
    plt.close()

    plt.figure(figsize=(12, 8))
    bar_width = 0.35
    order = [3, 2, 1, 4, 5, 6]
    df_ec_3B = df_ec_3B.set_index("EC_class").loc[order].reset_index()
    df_ec_15B = df_ec_15B.set_index("EC_class").loc[order].reset_index()
    x = np.arange(len(order))
    bar_width = 0.35
    plt.figure(figsize=(12, 8))
    plt.bar(x, df_ec_15B["Precision"], bar_width, label="15B Precision", color="grey")
    plt.bar(
        x, df_ec_15B["F1"], bar_width, label="15B F1 Score", color="blue", alpha=1.0
    )
    plt.bar(
        x,
        df_ec_15B["Recall"],
        bar_width,
        label="15B Recall",
        color="lightblue",
        alpha=0.9,
    )
    plt.bar(
        x + bar_width,
        df_ec_3B["Precision"],
        bar_width,
        label="3B Precision",
        color="grey",
    )
    plt.bar(
        x + bar_width,
        df_ec_3B["F1"],
        bar_width,
        label="3B F1 Score",
        color="red",
        alpha=1.0,
    )
    plt.bar(
        x + bar_width,
        df_ec_3B["Recall"],
        bar_width,
        label="3B Recall",
        color="salmon",
        alpha=0.9,
    )
    plt.xlabel("EC (tier 1)")
    plt.ylabel("Scores")
    plt.title("Precision, F1 Score, and Recall by EC Class")
    plt.xticks(x + bar_width / 2, order)
    plt.ylim(0, 1)
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="grey", label="Precision"),
        Patch(facecolor="blue", label="15B F1 Score"),
        Patch(facecolor="lightblue", label="15B Recall"),
        Patch(facecolor="red", label="3B F1 Score"),
        Patch(facecolor="salmon", label="3B Recall"),
    ]
    plt.legend(handles=legend_elements, loc="upper right")
    plt.tight_layout()
    plt.savefig("CataloDB_EC_specific_precision_F1_recall.svg")
    plt.close()

    order = [3, 2, 1, 4, 5, 6]
    df_ec_3B = df_ec_3B.set_index("EC_class").loc[order].reset_index()
    df_ec_15B = df_ec_15B.set_index("EC_class").loc[order].reset_index()
    x = np.arange(len(order))
    plt.figure(figsize=(6, 4))
    plt.scatter(
        x,
        df_ec_15B["Precision"],
        label="15B Precision",
        color="grey",
        marker="^",
        s=100,
    )
    plt.scatter(
        x, df_ec_15B["F1"], label="15B F1 Score", color="blue", marker="^", s=100
    )
    plt.scatter(
        x, df_ec_15B["Recall"], label="15B Recall", color="red", marker="^", s=100
    )
    plt.scatter(
        x + 0.15,
        df_ec_3B["Precision"],
        label="3B Precision",
        color="grey",
        marker="o",
        s=100,
    )
    plt.scatter(
        x + 0.15, df_ec_3B["F1"], label="3B F1 Score", color="blue", marker="o", s=100
    )
    plt.scatter(
        x + 0.15,
        df_ec_3B["Recall"],
        label="3B Recall",
        color="red",
        marker="o",
        s=100,
    )
    plt.xlabel("EC (tier 1)")
    plt.ylabel("Scores")
    plt.title("Precision, F1 Score, and Recall by EC Class")
    plt.xticks(x + 0.05, order)
    plt.ylim(0, 1)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("CataloDB_EC_specific_precision_F1_recall_scatter.svg")


if __name__ == "__main__":
    main()
