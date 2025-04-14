import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA

import torch

from tqdm import tqdm

plt.rcParams['svg.fonttype'] = 'none'

def get_catalytic_residues(metadata, EC_TX, AA_specific=None, ratio_non=5):
    # subset the df to all that have EC_TX in the EC_TX column where EC_TX column contains lists of EC numbers as strings
    cp_metadata = metadata.copy()
    
    # process the Active sites column so that it's not a string that looks like "[10,25]"
    if cp_metadata['AS'].dtype == "object":
        sites = list(cp_metadata["AS"])
        sites = [site.strip("[]").split(",") for site in sites]
        sites = [[int(x) for x in site] for site in sites]
        cp_metadata["AS"] = sites
    
    cr_entries = []
    non_cr_entries = []
    for i, row in tqdm(cp_metadata.iterrows()):
        if EC_TX in row["EC_TX"]:
            if AA_specific:
                # get catalytic residues which are the AA_specific
                crs = [pos for pos in row['AS'] if row['Sequence'][pos] == AA_specific]
                if len(crs) == 0:
                    continue
                num_cr = len(crs)
                # Get ratio_non number of non-catalytic residues that are the AA_specific
                seq = row['Sequence']
                seq_len = len(seq)
                cr_positions = row['AS']
                non_cr_positions = list(set(range(seq_len)) - set(cr_positions))
                non_cr_positions = [pos for pos in non_cr_positions if seq[pos] == AA_specific]
                if len(non_cr_positions)==0:
                    continue
                elif len(non_cr_positions) < ratio_non*num_cr:
                    non_cr_entries.append((row['Entry'], non_cr_positions, [seq[pos] for pos in non_cr_positions], len(row['Sequence'])))
                    cr_entries.append((row['Entry'], crs, [AA_specific]*len(crs), len(row['Sequence'])))
                else:
                    non_cr_positions = np.random.choice(non_cr_positions, ratio_non*num_cr, replace=False)
                    non_cr_AAs = [seq[pos] for pos in non_cr_positions]
                    non_cr_entries.append((row['Entry'], non_cr_positions, non_cr_AAs, len(row['Sequence'])))
                    cr_entries.append((row['Entry'], crs, [AA_specific]*len(crs), len(row['Sequence'])))
            else:            
                cr_AAs = [row['Sequence'][pos] for pos in row['AS']]
                cr_entries.append((row['Entry'], row['AS'], cr_AAs, len(row['Sequence'])))
                num_cr = len(row['AS'])
                seq = row['Sequence']
                seq_len = len(seq)
                cr_positions = row['AS']
                non_cr_positions = list(set(range(seq_len)) - set(cr_positions))
                non_cr_positions = np.random.choice(non_cr_positions, ratio_non*num_cr, replace=False)
                non_cr_AAs = [seq[pos] for pos in non_cr_positions]
                non_cr_entries.append((row['Entry'], non_cr_positions, non_cr_AAs, len(row['Sequence'])))            
            
            
            
    # check how many cr and entries we have
    print(f"Number of entries: {len(cr_entries)}")
    # num ca residues
    num_cr = sum([len(x[1]) for x in cr_entries])
    num_non_cr = sum([len(x[1]) for x in non_cr_entries])
    print(f"Number of catalytic residues: {num_cr}")
    print(f"Number of non-catalytic residues: {num_non_cr}")
    
    return cr_entries, non_cr_entries
    

def main():
    # import metadata from all data
    path = "/scratch/project/squid/code_modular/reproducing_squidly_scheme_2_dataset_2_esm2_t36_3B_UR50D_2025-01-10/Scheme2_4000/metadata_paired.tsv"
    metadata = pd.read_csv(path, sep="\t")
    print(f"Length of metadata: {len(metadata)}")
    
    # use this fasta file to drop any entries which are not in the fasta file
    fasta_path = "/scratch/project/squid/code_modular/reproducing_squidly_scheme_2_dataset_2_esm2_t36_3B_UR50D_2025-01-10/Scheme2_4000/0.9filtered_for_redundancy.fasta"
    with open(fasta_path, "r") as f:
        fasta = f.readlines()
    # get the entries
    entries = [line.strip().strip(">") for line in fasta if line.startswith(">")]
    entries = [entry.split("|")[1] for entry in entries]
    # get the entries in the metadata
    metadata = metadata[metadata["Entry"].isin(entries)]
    print(f"Length of metadata after redundancy filtering: {len(metadata)}")
    
    
    # loop through 3 EC numbers and every AA
    
    AA_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    
    for EC in ["2.7", "3.1", "1.3"]:
        for AA_specific in AA_alphabet:
            c_residues, non_cr_entries = get_catalytic_residues(metadata, EC, AA_specific=AA_specific, ratio_non=1)
            if len(non_cr_entries) == 0 or len(c_residues) == 0:
                continue
            AA_all = []
            cr_embeddings = []
            cr_positions_all = []
            cr_lengths_all = []
            for i, (entry, cr_positions, AAs, length) in tqdm(enumerate(c_residues)):
                # load the embedding for the entry
                emb_path = f"/scratch/project/squid/code_modular/reproducing_squidly_scheme_2_dataset_2_esm2_t36_3B_UR50D_2025-01-10/embeddings/sp|{entry}|esm.pt"
                emb = torch.load(emb_path)["representations"][36]
                # extract the embeddings for the catalytic residues
                cr_emb = emb[cr_positions]
                cr_embeddings.extend(cr_emb)
                AA_all.extend(AAs)
                cr_lengths_all.append(length)
                cr_positions_all.extend(cr_positions)
            
            non_cr_embeddings = []
            for i, (entry, non_cr_positions, AAs, length) in enumerate(non_cr_entries):
                # load the embedding for the entry
                emb_path = f"/scratch/project/squid/code_modular/reproducing_squidly_scheme_2_dataset_2_esm2_t36_3B_UR50D_2025-01-10/embeddings/sp|{entry}|esm.pt"
                emb = torch.load(emb_path)["representations"][36]
                # extract the embeddings for the catalytic residues
                non_cr_emb = emb[non_cr_positions]
                non_cr_embeddings.extend(non_cr_emb)
                AA_all.extend(AAs)
                cr_lengths_all.append(length)
                cr_positions_all.extend(non_cr_positions)
            
            # do a PCA with all the embeddings, and then colour them by catalytic/non-catalytic
            pca = PCA(n_components=2)
            all_embeddings = cr_embeddings + non_cr_embeddings
            all_embeddings = np.array(all_embeddings)
            
            # normalise the embeddings
            all_embeddings = (all_embeddings - np.mean(all_embeddings, axis=0))/np.std(all_embeddings, axis=0)
            
            pca.fit(all_embeddings)
            all_embeddings = pca.transform(all_embeddings)
            
            # plot the embeddings
            fig, ax = plt.subplots()
            ax.scatter(all_embeddings[:len(cr_embeddings), 0], all_embeddings[:len(cr_embeddings), 1], color="red", label="Catalytic", alpha=0.5)
            ax.scatter(all_embeddings[len(cr_embeddings):, 0], all_embeddings[len(cr_embeddings):, 1], color="blue", label="Non-catalytic", alpha=0.3)
            ax.legend()
            # put the x and y axis labels with % explained variance
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
            
            # save the plot
            plt.savefig((f"PCA_final/AA_specific/PCA_catalytic_non_catalytic_{EC}_{AA_specific}.svg"), format="svg")
            
            # now plot all the embeddings in the PCA and colour by a continuous colour scale for the length of the sequence
            fig, ax = plt.subplots()
            min_length = min(all_embeddings.shape[0], len(cr_lengths_all))
            all_embeddings = all_embeddings[:min_length]
            cr_lengths_all = cr_lengths_all[:min_length]
            sc = ax.scatter(all_embeddings[:, 0], all_embeddings[:, 1], c=cr_lengths_all, cmap="viridis")
            ax.legend(*sc.legend_elements(), title="Sequence Length")  
            # put the x and y axis labels with % explained variance
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
            
            # save the plot
            plt.savefig((f"PCA_final/AA_specific/PCA_sequence_length_{EC}_{AA_specific}_LENGTHS.svg"), format="svg")
            
            # now do the same but plot the colour by position of the residues in the sequence
            fig, ax = plt.subplots()
            min_length = min(all_embeddings.shape[0], len(cr_positions_all))
            all_embeddings = all_embeddings[:min_length]
            cr_positions_all = cr_positions_all[:min_length]
            sc = ax.scatter(all_embeddings[:, 0], all_embeddings[:, 1], c=cr_positions_all, cmap="viridis")
            ax.legend(*sc.legend_elements(), title="Position in Sequence")
            # put the x and y axis labels with % explained variance
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
            # save the plot
            plt.savefig((f"PCA_final/AA_specific/PCA_sequence_length_{EC}_{AA_specific}_POSITIONS.svg"), format="svg")
            
            # now calculate the correlation of PC1 and PC2 with the position in the sequence
            corr_PC1_pos = np.corrcoef(all_embeddings[:, 0], cr_positions_all)[0, 1]
            corr_PC2_pos = np.corrcoef(all_embeddings[:, 1], cr_positions_all)[0, 1]
            
            # and do the same for the length of the sequence
            corr_PC1_length = np.corrcoef(all_embeddings[:, 0], cr_lengths_all)[0, 1]
            corr_PC2_length = np.corrcoef(all_embeddings[:, 1], cr_lengths_all)[0, 1]
            
            # now save the correlation values
            with open(f"PCA_final/AA_specific/PCA_sequence_length_{EC}_{AA_specific}_correlations.txt", "w") as f:
                f.write(f"Correlation of PC1 with position in sequence: {corr_PC1_pos}\n")
                f.write(f"Correlation of PC2 with position in sequence: {corr_PC2_pos}\n")
                f.write(f"Correlation of PC1 with length of sequence: {corr_PC1_length}\n")
                f.write(f"Correlation of PC2 with length of sequence: {corr_PC2_length}\n")
            
            
            
        # now running for all AAs in the ECs
        c_residues, non_cr_entries = get_catalytic_residues(metadata, EC, AA_specific=None, ratio_non=3)
        AA_all = []
        cr_embeddings = []
        for i, (entry, cr_positions, AAs, lengths) in tqdm(enumerate(c_residues)):
            # load the embedding for the entry
            emb_path = f"/scratch/project/squid/code_modular/reproducing_squidly_scheme_2_dataset_2_esm2_t36_3B_UR50D_2025-01-10/embeddings/sp|{entry}|esm.pt"
            emb = torch.load(emb_path)["representations"][36]
            # extract the embeddings for the catalytic residues
            cr_emb = emb[cr_positions]
            cr_embeddings.extend(cr_emb)
            AA_all.extend(AAs)
        
        non_cr_embeddings = []
        for i, (entry, non_cr_positions, AAs, lengths) in enumerate(non_cr_entries):
            # load the embedding for the entry
            emb_path = f"/scratch/project/squid/code_modular/reproducing_squidly_scheme_2_dataset_2_esm2_t36_3B_UR50D_2025-01-10/embeddings/sp|{entry}|esm.pt"
            emb = torch.load(emb_path)["representations"][36]
            # extract the embeddings for the catalytic residues
            non_cr_emb = emb[non_cr_positions]
            non_cr_embeddings.extend(non_cr_emb)
            AA_all.extend(AAs)
        
        # do a PCA with all the embeddings, and then colour them by catalytic/non-catalytic
        pca = PCA(n_components=2)
        all_embeddings = cr_embeddings + non_cr_embeddings
        all_embeddings = np.array(all_embeddings)
        
        # normalise the embeddings
        all_embeddings = (all_embeddings - np.mean(all_embeddings, axis=0))/np.std(all_embeddings, axis=0)
        
        pca.fit(all_embeddings)
        all_embeddings = pca.transform(all_embeddings)
        
        # plot the embeddings
        fig, ax = plt.subplots()
        ax.scatter(all_embeddings[:len(cr_embeddings), 0], all_embeddings[:len(cr_embeddings), 1], color="red", label="Catalytic", alpha=0.5)
        ax.scatter(all_embeddings[len(cr_embeddings):, 0], all_embeddings[len(cr_embeddings):, 1], color="blue", label="Non-catalytic", alpha=0.3)
        ax.legend()
        # put the x and y axis labels with % explained variance
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        
        # save the plot
        plt.savefig((f"PCA_final/PCA_catalytic_non_catalytic_{EC}.svg"), format="svg")
        
        # now plot and colour the embeddings by Amino Acid type -- i.e. physiochemical properties
        # convert the AAs to physiochemical properties
        
        # physiochemical properties of AAs
        AA_properties = {
            "A": "Aliphatic Hydrophobic",
            "C": "Polar",
            "D": "Negative",
            "E": "Negative",
            "F": "Aromatic Hydrophobic",
            "G": "Aliphatic Hydrophobic",
            "H": "Polar",
            "I": "Aliphatic Hydrophobic",
            "K": "Positive",
            "L": "Aliphatic Hydrophobic",
            "M": "Aliphatic Hydrophobic",
            "N": "Polar",
            "P": "Aliphatic Hydrophobic",
            "Q": "Polar",
            "R": "Positive",
            "S": "Polar",
            "T": "Polar",
            "V": "Aliphatic Hydrophobic",
            "W": "Aromatic Hydrophobic",
            "Y": "Polar",
            "U": "Polar" 
        }
        
        AA_all = [AA_properties[AA] for AA in AA_all]

        df = pd.DataFrame(all_embeddings, columns=["PC1", "PC2"], index=[f"Entry_{i}" for i in range(len(all_embeddings))])
        # add a column for the AAs
        df["AA"] = [AA_all[i] for i in range(len(all_embeddings))]
        
        # plot the embeddings with colours by AA
        fig, ax = plt.subplots()
        for AA in set(AA_all):
            df_AA = df[df["AA"] == AA]
            ax.scatter(df_AA["PC1"], df_AA["PC2"], label=AA, alpha=0.5)
        ax.legend()
        # put the x and y axis labels with % explained variance
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
        
        # save the plot
        plt.savefig((f"PCA_final/PCA_AAs_{EC}.svg"), format="svg")
        
    
    
    
    

if __name__ == "__main__":
    main()