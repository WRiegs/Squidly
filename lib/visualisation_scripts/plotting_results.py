import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import glob

def create_parser():
    parser = argparse.ArgumentParser(
        description="Plot the results of the evaluation of reproduced results"
    )
    parser.add_argument(
        "--results",
        type=str,
        help="Path to the results directory containing the reproduced results",
    )
    parser.add_argument(
        "--out",
        type=str,
        help="Path to the output directory",
    )
    return parser


def main():
    argparser = create_parser()
    args = argparser.parse_args()
    
    # First get the raw ESM2 model results
    esm2_dir1 = "/scratch/project/squid/code_modular/reproduction_runs/esm2/reproduction_runs/reproducing_ESM2_models_dataset_1_2024-11-25"
    esm2_dir2 = "/scratch/project/squid/code_modular/reproduction_runs/esm2/reproduction_runs/reproducing_ESM2_models_dataset_2_2024-11-25"
    paths = [esm2_dir1, esm2_dir2]
    
    esm2LSTM_raw_f1s = []
    esm2LSTM_mean_f1s = []
    esm2LSTM_std_f1s = []
    for path in paths:
        lstm_files = glob.glob(f"{path}/ESM2raw_*/results/left_out_super_valid_set_evaluation_metrics.txt", recursive=True)
        f1s = []
        for file in lstm_files:
            # open the txt file and extract the f1 score on line 4 (1 indexed)
            with open(file, "r") as f:
                f1_score = f.readlines()[3].split()[1]
                f1 = float(f1_score)
                f1s.append(f1)
        # get the mean and standard deviation of the f1 scores
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        esm2LSTM_raw_f1s.append(f1s)
        esm2LSTM_mean_f1s.append(mean_f1)
        esm2LSTM_std_f1s.append(std_f1)
        
    
    
    # use glob to find all sub directories in the results directory that start with reproducing_squidly
    results_dirs = glob.glob(f"{args.results}/reproducing_squidly*")
    
    LSTM_raw_f1s = []
    LSTM_mean_f1s = []
    LSTM_std_f1s = []
    XG_raw_f1s = []
    XG_mean_f1s = []
    XG_std_f1s = []
    for dir in results_dirs:
        # collate all the LSTM files in the directory
        lstm_files = glob.glob(f"{dir}/Scheme*/LSTM/results/left_out_super_valid_set_evaluation_metrics.txt", recursive=True)
        XGB_files = glob.glob(f"{dir}/Scheme*/_XGBoostevaluation_metrics.txt", recursive=True)
        
        f1s = []
        for file in lstm_files:
            # open the txt file and extract the f1 score on line 4 (1 indexed)
            with open(file, "r") as f:
                f1_score = f.readlines()[3].split()[1]
                f1 = float(f1_score)
                f1s.append(f1)
                
        # get the mean and standard deviation of the f1 scores
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        LSTM_raw_f1s.append(f1s)
        LSTM_mean_f1s.append(mean_f1)
        LSTM_std_f1s.append(std_f1)
        
        f1s=[]
        for file in XGB_files:
            # open the txt file and extract the f1 score on line 4 (1 indexed)
            with open(file, "r") as f:
                f1_score = f.readlines()[3].split()[1]
                f1 = float(f1_score)
                f1s.append(f1)
        
        # get the mean and standard deviation of the f1 scores
        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s)
        XG_raw_f1s.append(f1s)
        XG_mean_f1s.append(mean_f1)
        XG_std_f1s.append(std_f1)
        
        # Create empty dataframes for mean and standard deviation F1 scores
    mean_df = pd.DataFrame(index=["1_LSTM", "1_XGBoost", "raw_ESM2", "2_LSTM", "2_XGBoost", "3_LSTM", "3_XGBoost"], columns=["dataset_1", "dataset_2", "dataset_3"])
    std_df = pd.DataFrame(index=["1_LSTM", "1_XGBoost", "raw_ESM2", "2_LSTM", "2_XGBoost", "3_LSTM", "3_XGBoost"], columns=["dataset_1", "dataset_2", "dataset_3"])

    # Fill the dataframes with the mean and std F1 scores
    for i, dir in enumerate(results_dirs):
        print(dir)
        scheme = dir.split("/")[-1].split("_")[3]
        dataset = dir.split("/")[-1].split("_")[5]
        mean_df.loc[f"{scheme}_LSTM", f"dataset_{dataset}"] = LSTM_mean_f1s[i]
        std_df.loc[f"{scheme}_LSTM", f"dataset_{dataset}"] = LSTM_std_f1s[i]
        mean_df.loc[f"{scheme}_XGBoost", f"dataset_{dataset}"] = XG_mean_f1s[i]
        std_df.loc[f"{scheme}_XGBoost", f"dataset_{dataset}"] = XG_std_f1s[i]
        
    for dataset in [1, 2]:
        mean_df.loc[f"raw_ESM2", f"dataset_{dataset}"] = esm2LSTM_mean_f1s[dataset-1]
        std_df.loc[f"raw_ESM2", f"dataset_{dataset}"] = esm2LSTM_std_f1s[dataset-1]
        
    print(mean_df)
    print(std_df)
    # Plot multi-histogram plot with error bars for standard deviation
    fig, axes = plt.subplots(1, 3, figsize=(9, 4), sharey=True)

    # pick 6 warm but distinct colors
    colors = ['skyblue', 'salmon', 'lightgreen', 'orange', 'pink', 'purple']

    # replace nans with 0s
    mean_df = mean_df.fillna(0)
    std_df = std_df.fillna(0)

    for i, dataset in enumerate(mean_df.columns):
        dataset_mean = mean_df[dataset].astype(float)
        dataset_std = std_df[dataset].astype(float)

        # Use Matplotlib's bar for error bar control
        axes[i].bar(dataset_mean.index, dataset_mean.values, yerr=dataset_std.values, capsize=3, color=colors)
        axes[i].set_title(f"Dataset {dataset.split('_')[1]}")
        axes[i].set_ylim(0, 1)
        axes[i].set_ylabel("F1 Score")
        axes[i].set_xlabel("Scheme")
        
        # Set y-ticks at 0.1 intervals
        axes[i].set_yticks(np.arange(0, 1.1, 0.1))
        
        # Rotate the x-axis labels to 45 degrees
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        # Add horizontal grid lines
        axes[i].yaxis.grid(True, linestyle='--', alpha=0.5)
        
        # put the df in a csv
        mean_df.to_csv(f"{args.out}/mean_f1_scores.csv")

    plt.tight_layout()
    plt.savefig(f"{args.out}/f1_scores")


if __name__ == "__main__": 
    main()