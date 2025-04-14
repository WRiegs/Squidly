
path = "/scratch/project/squid/AEGAN_extracted_sequences/train_test/uni3175_uni14230.fasta"
#path = "/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230.fasta"


# get the entries from the fasta file
entries = []
with open(path, "r") as f:
    for line in f:
        if line.startswith(">"):
            entries.append(line.strip())

# now find any duplicates and print them
duplicates = []
for entry in entries:
    if entries.count(entry) > 1:
        duplicates.append(entry)

print(duplicates)

print(len(duplicates))

print(len(set(entries)))

# reload the whole fasta file, including the sequence, but if the entry is a duplicate, skip it
# then write a new fastafile /scratch/project/squid/AEGAN_extracted_sequences/train_test/uni3175_uni14230_unduplicated.fasta

# read the whole fasta file
entries = []
seqs = []
with open(path, "r") as f:
    for line in f:
        if line.startswith(">"):
            entries.append(line.strip())
        else:
            seqs.append(line.strip())
            
# now drop the duplicate entries
entries_unique = list(set(entries))


import json
# now get the json files entries

json_3175 = "/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175.json"
json_14230 = "/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230.json"

with open(json_3175, "r") as f:
    data_3175 = json.load(f)
# get the entries which are the keys of the dictionary
entries_3175 = list(data_3175.keys())
print(len(entries_3175))

with open(json_14230, "r") as f:
    data_14230 = json.load(f)
# get the entries which are the keys of the dictionary
entries_14230 = list(data_14230.keys())
print(len(entries_14230))

# now get the overlapping entries in the json files
overlapping_entries = list(set(entries_3175).intersection(entries_14230))
print(len(overlapping_entries))

print(overlapping_entries)

# save the overlapping_entries in a txt file
with open("/scratch/project/squid/AEGAN_extracted_sequences/train_test/overlapping_entries.txt", "w") as f:
    for entry in overlapping_entries:
        f.write(entry + "\n")

# drop the overlapping entries from the entries_3175 list
#entries_3175 = list(set(entries_3175) - set(overlapping_entries))

entries_14230 = list(set(entries_14230) - set(overlapping_entries))


# save the list as the deduplicated entries in a txt file
with open("/scratch/project/squid/AEGAN_extracted_sequences/train_test/uni3175_uni14230_unduplicated_entries.txt", "w") as f:
    for entry in entries_3175:
        f.write(entry + "\n")
    for entry in entries_14230:
        f.write(entry + "\n")

# now for uni14230 and uni3175 alone
with open("/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175_unduplicated_entries.txt", "w") as f:
    for entry in entries_3175:
        f.write(entry + "\n")
        
with open("/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230_unduplicated_entries.txt", "w") as f:
    for entry in entries_14230:
        f.write(entry + "\n")

import pandas as pd

uni3175 = pd.read_csv("/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175.tsv", sep="\t")

# drop the Length, Protein families, Fragment and Subunit structure columns from the dataframe
uni3175 = uni3175.drop(columns=["Fragment", "Subunit structure"])

uni14230 = pd.read_csv("/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230.tsv", sep="\t")

uni14230 = uni14230.drop(columns=["Length", "Protein families", "Fragment", "Subunit structure"])

# drop rows that aren't in the entries_3175 list and entries_14230 list
uni3175 = uni3175[uni3175["Entry"].isin(entries_3175)]
uni14230 = uni14230[uni14230["Entry"].isin(entries_14230)]

# check if the dfs have any nans in Active site column

print("NAN hunting")
print(uni3175["Active site"].isnull().sum())
print(uni14230["Active site"].isnull().sum())
print()

# drop the rows with nans in Active site column
uni3175 = uni3175.dropna(subset=["Active site"])
uni14230 = uni14230.dropna(subset=["Active site"])


# check lengths
print(len(uni3175))
print(len(uni14230))

# save the dataframes as tsv files with unduplicated
uni3175.to_csv("/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175_unduplicated.tsv", sep="\t", index=False)
uni14230.to_csv("/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230_unduplicated.tsv", sep="\t", index=False)

# now make new fasta files for both using the Sequence column in the dataframes and the Entry column
with open("/scratch/project/squid/AEGAN_extracted_sequences/train_test/uni3175_uni14230_unduplicated.fasta", "w") as f:
    for entry, seq in zip(uni3175["Entry"], uni3175["Sequence"]):
        f.write('>' + entry + "\n" + seq + "\n")
    for entry, seq in zip(uni14230["Entry"], uni14230["Sequence"]):
        f.write('>' + entry + "\n" + seq + "\n")
        
with open("/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175_unduplicated.fasta", "w") as f:
    for entry, seq in zip(uni3175["Entry"], uni3175["Sequence"]):
        f.write('>' + entry + "\n" + seq + "\n")
        
with open("/scratch/project/squid/AEGAN_extracted_sequences/uni14230/uni14230_unduplicated.fasta", "w") as f:
    for entry, seq in zip(uni14230["Entry"], uni14230["Sequence"]):
        f.write('>' + entry + "\n" + seq + "\n")
