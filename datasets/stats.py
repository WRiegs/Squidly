import pandas
import torch
from torch.utils.data import Dataset, DataLoader

class PointerPairedDataset(Dataset):
    def __init__(self, main_store, pair_indices, labels):
        self.main_store = main_store     # Dict - e.g.  {i: torch.rand(512) for i in range(1000)}
        self.pair_indices = pair_indices # Each pair is a tuple of keys from `main_store`, like (0, 5) or (2, 8)
        self.labels = labels             # List of labels for each pair

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        # Retrieve indices for the current pair
        idx1, idx2 = self.pair_indices[idx]
        # Look up the actual data in the main store using these indices
        item1, item2 = self.main_store[idx1], self.main_store[idx2]
        label = self.labels[idx]
        return item1, item2, label  # Return the pair for contrastive training

def main():
    datasets = ["/scratch/project/squid/code_modular/datasets/dataset_1.tsv", "/scratch/project/squid/code_modular/datasets/dataset_2.tsv", "/scratch/project/squid/code_modular/datasets/dataset_3.tsv"]
    
    redundancy_sets = ["/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_1_2024-11-14/Scheme1_2300000_1/0.9filtered_for_redundancy.fasta",
                       "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_2_2024-11-14/Scheme1_2300000_1/0.9filtered_for_redundancy.fasta",
                       "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_3_2024-11-14/Scheme1_2300000_2/0.9filtered_for_redundancy.fasta"]
    
    validation_sets = ["/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_1_2024-11-14/Scheme1_2300000_1/Low30_mmseq_ID_250_exp_subset.txt",
                       "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_2_2024-11-14/Scheme1_2300000_1/Low30_mmseq_ID_250_exp_subset.txt",
                       "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_3_2024-11-14/Scheme1_2300000_2/Low30_mmseq_ID_250_exp_subset.txt"]
    
    for i in range(3):
        dataset = datasets[i]
        redundancy_set = redundancy_sets[i]
        validation_set = validation_sets[i]
        
        df = pandas.read_csv(dataset, sep="\t")
        
        with open(redundancy_set, "r") as fasta:
            fasta_lines = fasta.readlines()
        reduced_entry_names = []
        for line in fasta_lines:
            if line.startswith(">"):
                reduced_entry_names.append(line.split("|")[1])
            
        with open(validation_set, "r") as f:
            lines = f.readlines()
        # each line is an entry
        val_entries = [line.strip() for line in lines]
        
        redundancy = df[df["Entry"].isin(reduced_entry_names)]
        validation = df[df["Entry"].isin(val_entries)]
        
        print(f"Dataset {i+1}")
        print(f"Number of sequences: {len(df)}")
        print(f"Number of redundant sequences: {len(redundancy)}")
        
        # get the average length of the sequences in the redundancy set
        avg_len = redundancy["Sequence"].apply(len).mean()
        print(f"Average length of redundant sequences: {avg_len}")
        
        print(f"Number of validation sequences: {len(validation)}")
        print("")
        
        # get the number of catalytic residues in the redundancy set
        all_catalytic_residues = redundancy["Active site"]
        all_catalytic_residues = all_catalytic_residues.dropna()
        substring = "ACT_SITE"
        all_catalytic_residues = all_catalytic_residues.str.count(substring)
        all_catalytic_residues = all_catalytic_residues.sum()
        print(f"Number of catalytic residues in the redundancy set: {all_catalytic_residues}")
        validation_catalytic_residues = validation["Active site"]
        validation_catalytic_residues = validation_catalytic_residues.dropna()
        validation_catalytic_residues = validation_catalytic_residues.str.count(substring)
        validation_catalytic_residues = validation_catalytic_residues.sum()
        print(f"Number of catalytic residues in the validation set: {validation_catalytic_residues}")
        
        # write it to a stats file
        with open(f"/scratch/project/squid/code_modular/datasets/dataset_stats.txt", "w") as f:
            f.write(f"Dataset {i+1}\n")
            f.write(f"Number of sequences: {len(df)}\n")
            f.write(f"Number of redundant sequences: {len(redundancy)}\n")
            f.write(f"Average length of redundant sequences: {avg_len}\n")
            f.write(f"Number of catalytic residues in the redundancy set: {all_catalytic_residues}\n")
            f.write(f"Number of validation sequences: {len(validation)}\n")
            f.write(f"Number of catalytic residues in the validation set: {validation}\n")
            f.write("\n")
            
    # load the contrastive datasets of
    s1_ds1 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_1_2024-11-14/Scheme1_2300000_1/paired_embeddings_dataset.pt"
    s1_ds2 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_2_2024-11-14/Scheme1_2300000_1/paired_embeddings_dataset.pt"
    s1_ds3 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_1_dataset_3_2024-11-14/Scheme1_2300000_2/paired_embeddings_dataset.pt"
    s2_ds1 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_2_dataset_1_2024-11-14/Scheme2_3100_2/paired_embeddings_dataset.pt"
    s2_ds2 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_2_dataset_2_2024-11-14/Scheme2_3100_1/paired_embeddings_dataset.pt"
    s2_ds3 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_2_dataset_3_2024-11-14/Scheme2_3100_2/paired_embeddings_dataset.pt"
    s3_ds1 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_3_dataset_1_2024-11-14/Scheme3_2500_2/paired_embeddings_dataset.pt"
    s3_ds2 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_3_dataset_2_2024-11-14/Scheme3_2500_1/paired_embeddings_dataset.pt"
    s3_ds3 = "/scratch/project/squid/code_modular/reproduction_runs/small/reproducing_squidly_scheme_3_dataset_3_2024-11-14/Scheme3_2500_2/paired_embeddings_dataset.pt"
    
    paths = [s1_ds1, s1_ds2, s1_ds3, s2_ds1, s2_ds2, s2_ds3, s3_ds1, s3_ds2, s3_ds3]
    
    for file in paths:
        # load the dataset
        dataset = torch.load(file)
        # print the length of the dataset.pair_indices
        
        print(f"Dataset: {file}")
        print(f"Number of pairs: {len(dataset.pair_indices)}")
        print("")
        
    

if __name__ == '__main__':
    main()