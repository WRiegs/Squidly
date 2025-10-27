import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

diff_df = pd.read_csv("EasIFA_benchmark_per_sequence_differential_prediction.csv")
print(diff_df.head())


# now we need to order the df by EC number using only the first EC tier
def get_first_ec_tier(ec):
    if pd.isna(ec):
        return "Unknown"
    return ec.split(";")[0].split(".")[0]


def main():

    diff_df["first_ec_tier"] = diff_df["ec"].apply(get_first_ec_tier)
    order = [3, 2, 5, 1]
    diff_df["first_ec_tier"] = pd.Categorical(
        diff_df["first_ec_tier"],
        categories=[str(i) for i in order] + ["Unknown"],
        ordered=True,
    )
    diff_df = diff_df.sort_values(by=["first_ec_tier", "ec"])
    diff_df = diff_df.reset_index(drop=True)
    diff_df["Squidly_FP_binding"] = diff_df["Squidly_FP_binding"].apply(
        lambda x: x if x > 0 else np.nan
    )
    print(diff_df.head())
    diff_df["Squidly_FP_binding_booled"] = diff_df.apply(
        lambda row: (
            row["diff_FP"] if not pd.isna(row["Squidly_FP_binding"]) else np.nan
        ),
        axis=1,
    )

    plt.figure(figsize=(10, 5))
    x = np.arange(len(diff_df))
    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.scatter(
        x,
        diff_df["diff_TP"],
        color="blue",
        label="Difference in True Positives (Squidly - EasIFA)",
        alpha=0.7,
        marker="o",
        s=70,
    )
    plt.scatter(
        x,
        diff_df["diff_FP"],
        color="red",
        label="Difference in False Positives (Squidly - EasIFA)",
        alpha=0.7,
        marker="x",
        s=70,
    )
    plt.scatter(
        x,
        diff_df["Squidly_FP_binding_booled"],
        color="green",
        label="Squidly False Positives That are Binding Sites",
        alpha=0.7,
        marker="o",
        s=90,
        facecolors="none",
        edgecolors="green",
        linewidths=1.5,
    )
    plt.xticks(x, diff_df["ec"], rotation=90)
    plt.xlabel("EC Number")
    plt.ylabel("Number of Differences (Squidly - EasIFA)")
    plt.title("Per-sequence Differential Prediction: Squidly vs EasIFA")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        "EasIFA_benchmark_per_sequence_differential_prediction.svg", format="svg"
    )
    plt.close()

    print(diff_df.columns)

    agree = 0
    disagree = 0
    total = 0
    tp_agree = 0
    tp_total = 0
    fp_agree = 0
    fp_total = 0
    for i, row in diff_df.iterrows():
        squidly_TP = row["Squidly_TP"]
        easifa_TP = row["EasIFA_TP"]
        diff_TP = abs(row["diff_TP"])
        if squidly_TP == 0 and easifa_TP == 0:
            tp_overlap = 1.0
        else:
            tp_overlap = (squidly_TP + easifa_TP - diff_TP) / (squidly_TP + easifa_TP)
        squidly_FP = row["Squidly_FP"]
        easifa_FP = row["EasIFA_FP"]
        diff_FP = row["diff_FP"]
        if squidly_FP == 0 and easifa_FP == 0:
            fp_overlap = 1.0
        else:
            fp_overlap = (squidly_FP + easifa_FP - diff_FP) / (squidly_FP + easifa_FP)
        if not pd.isna(tp_overlap) and not pd.isna(fp_overlap):
            avg_overlap = (tp_overlap + fp_overlap) / 2
            agree += avg_overlap
            disagree += 1 - avg_overlap
            total += 1
            tp_agree += tp_overlap
            tp_total += 1
            fp_agree += fp_overlap
            fp_total += 1
    print(
        f"Average agreement between Squidly and EasIFA: {agree/total:.2%} ({agree:.1f} out of {total})"
    )
    print(
        f"Average agreement on True Positives: {tp_agree/tp_total:.2%} ({tp_agree:.1f} out of {tp_total})"
    )
    print(
        f"Average agreement on False Positives: {fp_agree/fp_total:.2%} ({fp_agree:.1f} out of {fp_total})"
    )


if __name__ == "__main__":
    main()
