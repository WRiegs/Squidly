import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
import subprocess
import glob
import json

MMSEQS_BIN = "mmseqs"

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

 
# Function to create a FASTA file from a TSV file
def create_fasta_from_tsv(tsv_file, fasta_file, entry_col, seq_col):
    df = pd.read_csv(tsv_file, sep='\t')
    with open(fasta_file, 'w') as fasta:
        for _, row in df.iterrows():
            fasta.write(f">{row[entry_col]}\n{row[seq_col]}\n")

# Function to run MMseqs2 clustering
def run_mmseqs_search(train_fasta, test_fasta, output_dir, identity_threshold):
    train_db = os.path.join(output_dir, "train_db")
    test_db = os.path.join(output_dir, "test_db")
    result_file = os.path.join(output_dir, f"results_{identity_threshold}.tsv")
    tmp_dir = os.path.join(output_dir, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # Create MMseqs2 databases for training and test sets
    subprocess.run([MMSEQS_BIN, "createdb", train_fasta, train_db], check=True)
    subprocess.run([MMSEQS_BIN, "createdb", test_fasta, test_db], check=True)

    # Run search to compare test sequences against training sequences
    subprocess.run([
        MMSEQS_BIN, "search", test_db, train_db, result_file, tmp_dir,
        "--min-seq-id", str(identity_threshold / 100),
        "-c", "0.8"
    ], check=True)
    
    return result_file


def run_mmseqs_similarity_search(set1_fasta, set2_fasta, output_dir):
    """Run MMseqs2 search to find similarities between two sets of sequences."""
    result_file = os.path.join(output_dir, "similarity_results.tsv")
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"mmseqs easy-search {set1_fasta} {set2_fasta} {result_file} tmp"
    os.system(cmd)
    return result_file

def run_mmseqs_similarity_search_max_identity(set1_fasta, set2_fasta, output_dir):
    """Run MMseqs2 search and filter sequences based on maximum sequence identity."""
    result_file = os.path.join(output_dir, "similarity_results.tsv")
    os.makedirs(output_dir, exist_ok=True)

    # Run MMseqs2 easy-search
    cmd = f"mmseqs easy-search {set1_fasta} {set2_fasta} {result_file} tmp"
    os.system(cmd)

    # Read the results and calculate maximum identity per query
    result_df = pd.read_csv(result_file, sep='\t', header=None)
    result_df.columns = ["Query", "Target", "Identity", "Alignment Length", "Mismatches", 
                         "Gap Opens", "Query Start", "Query End", "Target Start", 
                         "Target End", "E-value", "Bit Score"]

    # Group by Query and calculate the maximum identity
    max_identity_df = result_df.groupby("Query")["Identity"].max().reset_index()
        
    return max_identity_df

# Function to label test sequences based on identity ranges
def label_test_sequences(train_tsv, test_tsv, output_file):
    train_fasta = "train_sequences.fasta"
    test_fasta = "test_sequences.fasta"
    
    # Create FASTA files for training and test sets
    create_fasta_from_tsv(train_tsv, train_fasta, "Entry", "Sequence")
    create_fasta_from_tsv(test_tsv, test_fasta, "Entry", "Sequence")

    test_df = pd.read_csv(test_tsv, sep='\t')

    output_dir = f"mmseqs_output"
    os.makedirs(output_dir, exist_ok=True)

    # Run MMseqs2 search
    result_df = run_mmseqs_similarity_search_max_identity(test_fasta, train_fasta, output_dir)

    # merge the test_df with the result_df, by the Entry column and only include the Identity column. There will be some entries in the test-DF with no hits... keep them and label them as 0 in Identity col
    merged_df = test_df.merge(result_df, left_on="Entry", right_on="Query", how="left")
    merged_df["Identity"] = merged_df["Identity"].fillna(0)
    
    # convert the Identity column to percentage
    merged_df["Identity"] *= 100
    
    # now create new cols for 30,50,80 and label the sequences
    merged_df["Identity Range"] = "Unlabeled"
    merged_df.loc[merged_df["Identity"] >= 80, "Identity Range"] = "80-100"
    merged_df.loc[merged_df["Identity"] < 80, "Identity Range"] = "50-80"
    merged_df.loc[merged_df["Identity"] < 50, "Identity Range"] = "30-50"
    merged_df.loc[merged_df["Identity"] < 30, "Identity Range"] = "0-30"
    
    # get counts of each range
    counts = merged_df["Identity Range"].value_counts()
    print(counts)
    
    # save the counts to a file for later use
    counts.to_csv(output_dir + "/identity_counts.tsv", sep="\t")
    
    # save the merged_df to a file
    merged_df.to_csv(output_file, sep="\t", index=False)
    
    
    
def main():
    # Load the data
    path = "/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175_unduplicated.tsv"
    df = pd.read_csv(path, sep="\t")
    output = "/scratch/project/squid/AEGAN_extracted_sequences/uni3175/"
    
    df["EC_TX"] = get_EC_TX_from_uniprot(df, tier = 1)
    
    sites, AAs = get_AS_pos_from_uniprot(df)
    df["Active_sites"] = sites
    df["Active_site_AAs"] = AAs
    
    # Input files
    train_tsv = "/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230_unduplicated.tsv"
    test_tsv = "/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175_unduplicated.tsv"
    output_tsv = "/scratch/project/squid/AEGAN_extracted_sequences/uni3175/labeled_test_sequences.tsv"
    
    # Label test sequences
    label_test_sequences(train_tsv, test_tsv, output_tsv)
    
    return

if __name__ == "__main__":
    main()