import glob
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


input_fasta = "/scratch/project/squid/code_modular/squidly_benchmark/0.9filtered_for_redundancy.fasta"

family_specific_embeddings = "/scratch/project/squid/code_modular/benchmark_data/family_specific_embeddings_esm2_t48_15B_UR50D"

# extract all the entry names in the family specific embeddings
filenames = glob.glob(f"{family_specific_embeddings}/*esm.pt")

# extract the entry names from the filenames
family_specific_entry_names = []
for filename in filenames:
    entry_name = filename.split("/")[-1].split("|")[1]
    family_specific_entry_names.append(entry_name)
    

# load the set of test data
eval_set = "/scratch/project/squid/code_modular/squidly_benchmark/Structurally_filtered_eval_benchmark/0.3_Structural_ID_benchmark.txt"
with open(eval_set, "r") as f:
    eval_set_entry_names = [line.strip() for line in f]


# now find any entry names that are in both lists and remove them from the original input fasta
# make a new fasta file with the remaining entries
# call it 0.9reduced_training_set_no_family_specific.fasta
count = 0
output_fasta = "/scratch/project/squid/code_modular/squidly_benchmark/0.9reduced_set_no_family_specific.fasta"
with open(output_fasta, "w") as out:
    for record in SeqIO.parse(input_fasta, "fasta"):
        if record.id.split("|")[1] not in family_specific_entry_names:
            SeqIO.write(record, out, "fasta")
        elif record.id.split("|")[1] in eval_set_entry_names:
            SeqIO.write(record, out, "fasta")
        else:
            count += 1

print(f"Number of entries removed from training data: {count}")