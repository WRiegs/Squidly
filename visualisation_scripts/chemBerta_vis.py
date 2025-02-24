import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
import os
import pickle

from sklearn.decomposition import PCA


def argparser():
    parser = argparse.ArgumentParser(description='Visualise dataset')
    parser.add_argument('--ChemBerta', required=True, help='Path to dataset')
    parser.add_argument('--dataset', required=True, help='Path to dataset')
    parser.add_argument('--fasta_redundancy_reduced', required=True, help='Path to fasta file with redundancy reduced sequences')
    parser.add_argument('--validation_set', required=False, help='Path to validation set')
    parser.add_argument('--output', required=True, help='Path to output directory')
    args = parser.parse_args()
    return args


def main():
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
    
    dataset = 1
    reproduction_run_dir="/scratch/project/squid/code_modular/reproduction_runs/small"
    validation_sets = glob.glob(f"{reproduction_run_dir}/reproducing_squidly_scheme_*_dataset_{dataset}_*/*/Low30_mmseq_ID_250_exp_subset.txt", recursive=True)
    # get the first validation_set
    validation_set = validation_sets[0]
    
    # load the validation set
    with open(validation_set, "r") as f:
        lines = f.readlines()
    # each line is an entry
    val_entries = [line.strip() for line in lines]
    
    # load the chemberta pkl
    with open(args.ChemBerta, "rb") as f:
        chemBerta = pickle.load(f)
    
    print(chemBerta)
    
    print(chemBerta.columns)
    
    print(chemBerta['uid'])
    print(chemBerta['reaction'])
    print(chemBerta['id'])
    
    # make a PCA of the chemBerta['chemberta'] embeddings
    
    # prep the embeddings into a np matrix for PCA
    embeddings = np.array(chemBerta['chemberta'].tolist())
    print(embeddings)
    
    
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(embeddings)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
    principalDf['uid'] = chemBerta['uid']
    principalDf['reaction'] = chemBerta['reaction']
    principalDf['id'] = chemBerta['id']
    
    # make a new column with just the first digit from the id
    principalDf['id_first_digit'] = principalDf['id'].str[0]
    
    print(principalDf)
    
    # randomise the order of the df
    principalDf = principalDf.sample(frac=1).reset_index(drop=True)
    
    # plot the PCA and colour by first digit of id
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='PC1', y='PC2', data=principalDf, hue='id_first_digit', alpha = 0.4)
    plt.xlabel('PC1 (explained variance: {})'.format(pca.explained_variance_ratio_[0]))
    plt.ylabel('PC2 (explained variance: {})'.format(pca.explained_variance_ratio_[1]))
    plt.title(f'Dataset {dataset}: PCA of ChemBerta embeddings')
    plt.savefig(os.path.join(args.output, 'PCA_ChemBerta.png'))
    
    
    
        
    
    
    
    return

if __name__ == '__main__':
    main()

