import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os


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

 
    
def main():
    # Load the data
    path = "/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230_unduplicated.tsv"
    df = pd.read_csv(path, sep="\t")
    
    # read in the squidly benchmark test list and find how many overlap with squidly, then save the ones that don't overlap
    squidly_benchmark = "/scratch/project/squid/code_modular/squidly_benchmark/Structurally_filtered_eval_benchmark/0.3_Structural_ID_benchmark.txt"
    
    # load the list
    with open(squidly_benchmark, "r") as f:
        squidly_benchmark_list = f.readlines()
    squidly_benchmark_list = [x.strip() for x in squidly_benchmark_list]
    
    # find the overlap
    overlap = df[df["Entry"].isin(squidly_benchmark_list)]
    overlap_entries = overlap['Entry'].tolist()
    
    # now save the entries that don't overlap
    squidly_non_overlap = [x for x in squidly_benchmark_list if x not in overlap_entries]
    
    # print the number of entries that don't overlap
    print("benchmark overlap with squidly")
    print(len(squidly_non_overlap))
    print(squidly_non_overlap)
    
    # save the non-overlapping entries as a txt file
    non_overlap_df = df[df["Entry"].isin(squidly_non_overlap)]
    with open("/scratch/project/squid/code_modular/squidly_benchmark/Structurally_filtered_eval_benchmark/0.3_Structural_ID_benchmark_non_overlap_with_AEGAN.txt", "w") as f:
        for item in squidly_non_overlap:
            f.write("%s\n" % item)
    
    # get the seqs from this df
    path = "/scratch/project/squid/code_modular/datasets/dataset_2.tsv"
    squid_df = pd.read_csv(path, sep="\t")
    squid_df = squid_df[squid_df["Entry"].isin(squidly_non_overlap)]
    
    # save
    squid_df.to_csv("/scratch/project/squid/code_modular/squidly_benchmark/Structurally_filtered_eval_benchmark/0.3_Structural_ID_benchmark_non_overlap_with_AEGAN_seqs.tsv", sep="\t", index=False)
    
    df = squid_df
    
    output = "/scratch/project/squid/AEGAN_extracted_sequences/uni3175/"
    output = "/scratch/project/squid/code_modular/squidly_benchmark/Structurally_filtered_eval_benchmark/"
    
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
    # plot the distribution of EC_TX numbers in the dataset as a pie chart
    plt.figure(figsize=(6, 6))
    plt.pie(EC_TX_counts, labels=EC_TX_counts.index, autopct='%1.1f%%', textprops={'fontsize': 14})
    plt.title(f'Uni3175 EC Distribution')
    plt.savefig(os.path.join(output, f'AEGAN_ECX_piechart.png'), format="png")
    
    # plot the distribution of active site AAs in the dataset
    AAs = [item for sublist in df["Active_site_AAs"] for item in sublist]
    AAs = pd.Series(AAs)
    AAs_counts = AAs.value_counts()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=AAs_counts.index, y=AAs_counts.values)
    plt.xlabel('Amino Acid')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Catalytic Amino Acids in Uni3175')
    plt.savefig(os.path.join(output, 'AEGAN_Active_site_AAs_distribution.png'), format="png")
    
    return

if __name__ == "__main__":
    main()