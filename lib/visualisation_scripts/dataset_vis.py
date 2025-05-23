# python visualisation_scripts/dataset_vis.py --dataset /scratch/project/squid/Thesis_datasets/dataset_1.tsv --fasta_redundancy_reduced /scratch/project/squid/Thesis_datasets/dataset_1.fasta --output /scratch/project/squid/Thesis_datasets/vis/dataset_1
# python visualisation_scripts/dataset_vis.py --dataset /scratch/project/squid/Thesis_datasets/dataset_2.tsv --fasta_redundancy_reduced /scratch/project/squid/Thesis_datasets/dataset_2.fasta --output /scratch/project/squid/Thesis_datasets/vis/dataset_2
# python visualisation_scripts/dataset_vis.py --dataset /scratch/project/squid/Thesis_datasets/dataset_3.tsv --fasta_redundancy_reduced /scratch/project/squid/Thesis_datasets/dataset_3.fasta --output /scratch/project/squid/Thesis_datasets/vis/dataset_3
# for dir in /scratch/project/squid/Thesis_datasets/vis/dataset_{1,2,3}; do for f in "$dir"/*; do mv "$f" "$dir/$(basename "$dir")_$(basename "$f")"; done; done

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
import os
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'font.size': 18})  # Applies to everything

def argparser():
    parser = argparse.ArgumentParser(description='Visualise dataset')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--fasta_redundancy_reduced', required=True, help='Path to fasta file with redundancy reduced sequences')
    #parser.add_argument('--validation_set', required=False, help='Path to validation set')
    parser.add_argument('--output', required=True, help='Path to output directory')
    args = parser.parse_args()
    return args


def get_AS_pos_from_uniprot(df):
    active_sites = []
    active_site_AA = []
    # iterate through the df and get the active sites
    for index, row in df.iterrows():
        active_site_string = row["Active site"]
        active_site_list = []
        intermediate_list = active_site_string.split(";")
        #iterate through the intermediate list and get the active sites
        for item in intermediate_list:
            if item.startswith("ACT_SITE") or item.startswith(" ACT_SITE"):
                active_site_list.append(int(item.split("ACT_SITE ")[1])-1)
        active_sites.append(active_site_list)
        
        # go through the active site list and extract the AA at the active site position
        AAs_in_active_site = []
        for active_site in active_site_list:
            AA = row["Sequence"][active_site]
            AAs_in_active_site.append(AA)
        active_site_AA.append(AAs_in_active_site)
        
    return active_sites, active_site_AA


def get_EC_TX_from_uniprot(df, tier = 2):
    EC_TX = []
    # iterate through the df and get the ECs up to the 2nd tier
    for index, row in df.iterrows():
        EC_string = row["EC number"]
        EC_list = []
        intermediate_list = EC_string.split(";")
        intermediate_list = [x.strip() for x in intermediate_list]
        intermediate_list = ['.'.join(x.split(".")[0:tier]) for x in intermediate_list]     # get the first 2 tiers of the EC number
        EC_list = list(dict.fromkeys(intermediate_list))    # remove duplicate strings in the list
        EC_TX.append(EC_list)
    return EC_TX


def gather_validation_set_data(df, output, dataset = 1, reproduction_run_dir="/scratch/project/squid/code_modular/reproduction_runs/small"):
    # get all the validation sets within the dataset of interest
    # /scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_1_2024-11-14/
    validation_sets = glob.glob(f"{reproduction_run_dir}/reproducing_squidly_scheme_*_dataset_{dataset}_*/*/Low30_mmseq_ID_250_exp_subset.txt", recursive=True)
    
    print(validation_sets)
    
    # open the validation sets and get the entries then count their EC X.X numbers for later counting
    validation_set_ECs = []
    for validation_set in validation_sets:
        with open(validation_set, "r") as f:
            lines = f.readlines()
        # each line is an entry
        entries = [line.strip() for line in lines]
        # get the EC numbers
        df_subset = df[df["Entry"].isin(entries)]
        EC_TX = get_EC_TX_from_uniprot(df_subset, tier = 1)
        validation_set_ECs.append(EC_TX)
        
    # now do the value counts of ECs, and divide it by the number of validation sets to get the average validation distribution
    validation_set_ECs = [item for sublist in validation_set_ECs for item in sublist]
    # remove any sublists [2, 3], [1, 3, 6] by making it so we can count them all
    validation_set_ECs = [item for sublist in validation_set_ECs for item in sublist]
    validation_set_ECs = pd.Series(validation_set_ECs)
    validation_set_ECs_counts = validation_set_ECs.value_counts()
    validation_set_ECs_counts = validation_set_ECs_counts/len(validation_sets)
    
    print(validation_set_ECs_counts)
    
    # plot the distribution of EC_TX numbers in the dataset as a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(validation_set_ECs_counts, labels=validation_set_ECs_counts.index, autopct='%1.1f%%', textprops={'fontsize': 14})
    plt.title(f'Dataset {dataset}')
    plt.savefig(os.path.join(output, f'dataset{dataset}_validation_set_EC_TX_pie_distribution.svg'), format='svg')
    
def main():
    dataset = 2
    
    args = argparser()
    
    # load the dataset
    df = pd.read_csv(args.dataset, sep='\t')
    
    # load the fasta
    with open(args.fasta_redundancy_reduced, "r") as fasta:
        fasta_lines = fasta.readlines()
    
    # get all the Entry names from the fasta file
    entry_names = []
    for line in fasta_lines:
        if line.startswith(">"):
            entry_names.append(line.split("|")[1])
    
    # subset the df to only include the rows that are in the fasta file
    df = df[df["Entry"].isin(entry_names)]

    # drop df if nan in EC
    df = df.dropna(subset=["EC number"])
    # drop df if nan in Active site
    df = df.dropna(subset=["Active site"])
    
    df["EC_TX"] = get_EC_TX_from_uniprot(df, tier = 1)
    
    sites, AAs = get_AS_pos_from_uniprot(df)
    df["Active_sites"] = sites
    df["Active_site_AAs"] = AAs
    
    # plot the distribution of EC_TX numbers in the dataset
    EC_TX = [item for sublist in df["EC_TX"] for item in sublist]
    EC_TX = pd.Series(EC_TX)
    EC_TX_counts = EC_TX.value_counts()
    # colour the plot with a unique colour for each EC number
    colours = sns.color_palette("tab10", len(EC_TX_counts))

    # use this pallette
    colours = ['#FFC440', '#D7572B', '#3A53A4', '#AFC6CF', '#895981', '#937A64', '#A4C8E1']
    order = ['1', 
            '2',
            '3',
            '4',
            '5',
            '6',
            '7'
            ]


    plt.figure(figsize=(8, 5))
    sns.barplot(x=EC_TX_counts.index, y=EC_TX_counts.values, palette=colours)
    plt.xticks(rotation=90)
    plt.xlabel('Enzyme Class - Tier 1')
    plt.ylabel('Frequency')
    plt.title(f'Dataset {dataset}: Distribution of EC numbers (tier 1) in the dataset')
    plt.savefig(os.path.join(args.output, 'EC_TX_distribution.svg'), format='svg')
    plt.savefig(os.path.join(args.output, 'EC_TX_distribution.png'), format='png')
    
    
    
    # get the 
    df["EC_TX"] = get_EC_TX_from_uniprot(df)
    
    sites, AAs = get_AS_pos_from_uniprot(df)
    df["Active_sites"] = sites
    df["Active_site_AAs"] = AAs
    
    # plot the distribution of EC_TX numbers in the dataset
    EC_TX = [item for sublist in df["EC_TX"] for item in sublist]
    EC_TX = pd.Series(EC_TX)
    EC_TX_X_counts = EC_TX.value_counts()
    
    # now make a colour palette for each individual EC tier two number based on its tier 1 number
    tier_2colours = []
    for i in range(len(EC_TX_X_counts)):
        tier_1 = EC_TX_X_counts.index[i].split(".")[0]
        tier_1_index = EC_TX_counts.index.get_loc(tier_1)
        tier_2colours.append(colours[tier_1_index])
        
    print(tier_2colours)
    
    plt.figure(figsize=(20, 9))
    sns.barplot(x=EC_TX_X_counts.index, y=EC_TX_X_counts.values, palette=tier_2colours)
    # add a legend that corresponds to the tier 1 EC number colours
    legend = []
    for i in range(len(EC_TX_counts)):
        legend.append(plt.Line2D([0], [0], color=colours[i], lw=4))
    plt.legend(legend, EC_TX_counts.index, title='Tier 1 EC number')
    plt.xticks(rotation=90)
    plt.xlabel('Enzyme Class - Tier 2')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of EC numbers (tier 2) in the dataset {dataset}')
    plt.savefig(os.path.join(args.output, 'EC_TX.X_distribution.svg'), format='svg')
    plt.savefig(os.path.join(args.output, 'EC_TX.X_distribution.png'), format='png')
    plt.close()

    plt.rcParams.update({'font.size': 16})  # Applies to everything

    # plot the distribution of active site AAs in the dataset
    AAs = [item for sublist in df["Active_site_AAs"] for item in sublist]
    AAs = pd.Series(AAs)
    AAs_counts = AAs.value_counts()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=AAs_counts.index, y=AAs_counts.values, palette=['#895981'])
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Catalytic Amino Acids in the dataset {dataset}')
    plt.savefig(os.path.join(args.output, 'Active_site_AAs_distribution.svg'), format='svg')
    plt.savefig(os.path.join(args.output, 'Active_site_AAs_distribution.png'), format='png')
    plt.close()

    total_AAs = len(AAs)
    print(f"Total number of Catalytic AAs: {total_AAs}")

    #gather_validation_set_data(df, args.output, dataset = dataset)
    
    return

if __name__ == "__main__":
    main()