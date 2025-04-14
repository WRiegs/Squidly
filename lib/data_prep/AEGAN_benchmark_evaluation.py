import pandas as pd
import os
import json


def main():
    base_dir = "/scratch/project/squid/AEGAN_extracted_sequences/"
    EF_family = base_dir+"EF_family/EF_family.fasta"
    EF_fold = base_dir+"EF_fold/EF_fold.fasta"
    EF_superfamily = base_dir+"EF_superfamily/EF_superfamily.fasta"
    HA_superfamily = base_dir+"HA_superfamily/HA_superfamily.fasta"
    NN = base_dir+"NN/NN.fasta"
    PC = base_dir+"PC/PC.fasta"
    
    # get all the entries from the 6 family files then put them in one set
    entries = []
    for file in [EF_family, EF_fold, EF_superfamily, HA_superfamily, NN, PC]:
        with open(file, "r") as f:
            for line in f:
                if line.startswith(">"):
                    entries.append(line.strip())
                    
    # double check if any in training data:
    training_entries = "/scratch/project/squid/AEGAN_extracted_sequences/train_test/uni3175_uni14230_unduplicated_entries.txt"
    
    # open and load
    with open(training_entries, "r") as f:
        training_entries = f.readlines()
    
    #Check for any duplicates
    duplicates = list(set(entries).intersection(training_entries))
    print(f"Training leakage: {len(duplicates)}")
                         
    entries_unique = list(set(entries))
    
    # load the all_fams.tsv
    all_fams = "/scratch/project/squid/AEGAN_extracted_sequences/family_specific/all_fams.tsv"
    all_fams_df = pd.read_csv(all_fams, sep="\t")
    
    # deduplicate rows in the df
    all_fams_df = all_fams_df.drop_duplicates(subset="Entry")
    
    # now make a new deduplicated fasta file with the unique entries
    all_fams_dedup = "/scratch/project/squid/AEGAN_extracted_sequences/family_specific/all_fams_dedup.fasta"
    with open(all_fams_dedup, "w") as f:
        for row in all_fams_df.iterrows():
            entry = row[1]["Entry"]
            f.write('>' + entry + "\n")
            f.write(row[1]["Sequence"] + "\n")
            
    extracting_esm = f"python lib/extract_esm2.py esm2_t36_3B_UR50D {all_fams_dedup} /scratch/project/squid/code_modular/reproducing_AEGAN_benchmark_squidly_scheme_2_esm2_t36_3B_UR50D_2025-01-09/family_specific_embeddings --toks_per_batch 1000 --include per_tok"
    #os.system(extracting_esm)
    # now make a new fast file with the unique entries
    
    dirty_training_data = "/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230.json"
    
    # load the json file and get the entries from there
    with open(dirty_training_data, "r") as f:
        data = json.load(f)
    # get the entries which are the keys of the dictionary
    entries_dirty = list(data.keys())
    duplicates = list(set(entries).intersection(entries_dirty))
    print(f"Training leakage in the dirty set: {len(duplicates)}")
    # now compare 
    
    
    
if __name__ == "__main__":
    main()