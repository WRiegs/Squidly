import pandas
import numpy
from Bio import SeqIO
from Bio.Seq import Seq
import os
import random
import argparse
import subprocess

def argparser():
    parser = argparse.ArgumentParser(description='Generate evaluation data for TRAST')
    parser.add_argument('--fasta', type=str, help='Path to the fasta file')
    parser.add_argument('--out', type=str, help='Path to the output directory')
    parser.add_argument('--num_samples', type=int, help='Number of samples to take from the full set')
    parser.add_argument('--experimental_data', type=str, help='Path to the experimental data file')
    parser.add_argument('--redundancy_threshold', type=float, help='Threshold for redundancy filtering on all data')
    return parser.parse_args()


def load_fasta(fasta_file):
    seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs.append(record)
    return seqs


def load_cdhit(clstr_file):
    clusters = []
    with open(clstr_file) as f:
        cluster = []
        for line in f:
            if line.startswith(">"):
                if cluster:
                    clusters.append(cluster)
                cluster = []
            else:
                cluster.append(line.strip())
        clusters.append(cluster)
    return clusters


def get_cluster_seqs(cluster):
    redundant_seqs = []
    for seq in cluster:
        redundant_seqs.append(seq.split(">")[1].split("|")[1])
    return redundant_seqs


def filter_low_identity_sequences(input_fasta, similarity_threshold=0.9):
    # Define temporary directories
    tmp_dir = "mmseqs_tmp"
    db_dir = os.path.join(tmp_dir, "db")
    clustered_dir = os.path.join(tmp_dir, "clustered")
    repseqs_fasta = os.path.join(tmp_dir, "repseqs")
    output_fasta = os.path.join(tmp_dir, "output")
    
    if similarity_threshold < 0.4:
        cov = 0.5
    elif similarity_threshold > 0.8:
        cov = 0.9
    else:
        cov = 0.7
    
    # Create temporary directories
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # Convert FASTA to MMseqs2 database
        cmd_convert = [
            "mmseqs", "createdb", input_fasta, db_dir
        ]
        subprocess.run(cmd_convert, check=True)

        # Perform clustering with the specified similarity threshold
        cmd_cluster = [
            "mmseqs", "cluster", db_dir, clustered_dir, tmp_dir,
            "--min-seq-id", str(similarity_threshold),
            "-c", str(cov), # Coverage threshold
            "--cov-mode", "0", # Use the full alignment for coverage calculation
            "--threads", "1" # Number of threads to use
        ]
        subprocess.run(cmd_cluster, check=True)

        # Get representative sequences (these will be the representatives of the clusters)
        cmd_repseq = [
            "mmseqs", "result2repseq", db_dir, clustered_dir, repseqs_fasta
        ]
        subprocess.run(cmd_repseq, check=True)

        # Extract sequences that were not part of any cluster (low identity sequences)
        cmd_extract = [
            "mmseqs", "createsubdb", repseqs_fasta, db_dir, output_fasta
        ]
        subprocess.run(cmd_extract, check=True)

        # Convert the final database back to a FASTA file
        cmd_export = [
            "mmseqs", "convert2fasta", output_fasta, output_fasta + ".fasta"
        ]
        subprocess.run(cmd_export, check=True)
        
        # Covert thre representative sequences to a fasta file
        cmd_export_rep = [
            "mmseqs", "convert2fasta", repseqs_fasta, repseqs_fasta + ".fasta"
        ]
        subprocess.run(cmd_export_rep, check=True)

        low_identity_sequences = []
        with open(output_fasta + ".fasta", "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                low_identity_sequences.append(record.id)
                
        rep_sequences = []
        with open(repseqs_fasta + ".fasta", "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                rep_sequences.append(record.id)
        
        low_identity_sequences = low_identity_sequences + rep_sequences
                

    finally:
        # Clean up temporary directories
        if os.path.exists(tmp_dir):
            subprocess.run(["rm", "-r", tmp_dir])

    return low_identity_sequences


def filter_singleton_sequences(input_fasta, similarity_threshold=0.4):
    # Define temporary directories
    tmp_dir = "mmseqs_tmp"
    db_dir = os.path.join(tmp_dir, "db")
    clustered_dir = os.path.join(tmp_dir, "clustered")
    repseqs_fasta = os.path.join(tmp_dir, "repseqs")
    output_fasta = os.path.join(tmp_dir, "output")
    
    if similarity_threshold < 0.4:
        cov = 0.5
    elif similarity_threshold > 0.8:
        cov = 0.9
    else:
        cov = 0.7
    
    # Create temporary directories
    os.makedirs(tmp_dir, exist_ok=True)
    
    try:
        # Convert FASTA to MMseqs2 database
        cmd_convert = [
            "mmseqs", "createdb", input_fasta, db_dir
        ]
        subprocess.run(cmd_convert, check=True)

        # Perform clustering with the specified similarity threshold
        cmd_cluster = [
            "mmseqs", "cluster", db_dir, clustered_dir, tmp_dir,
            "--min-seq-id", str(similarity_threshold),
            "-c", str(cov), # Coverage threshold
            "--cov-mode", "0", # Use the full alignment for coverage calculation
            "--threads", "1" # Number of threads to use
        ]
        subprocess.run(cmd_cluster, check=True)

        # Get the list of representative sequences (these include all singletons and representatives)
        cmd_repseq = [
            "mmseqs", "result2repseq", db_dir, clustered_dir, repseqs_fasta
        ]
        subprocess.run(cmd_repseq, check=True)

        # Find singleton sequences by identifying those that are their own representative
        cmd_singletons = [
            "mmseqs", "createsubdb", repseqs_fasta, db_dir, output_fasta
        ]
        subprocess.run(cmd_singletons, check=True)

        # Convert the final singleton database back to a FASTA file
        cmd_export_singletons = [
            "mmseqs", "convert2fasta", output_fasta, output_fasta + ".fasta"
        ]
        subprocess.run(cmd_export_singletons, check=True)

        singleton_sequences = []
        with open(output_fasta + ".fasta", "r") as fasta_file:
            for record in SeqIO.parse(fasta_file, "fasta"):
                singleton_sequences.append(record.id)

    finally:
        # Clean up temporary directories
        if os.path.exists(tmp_dir):
            subprocess.run(["rm", "-r", tmp_dir])

    return singleton_sequences
    

def main():
    args = argparser()
    
    # if the fasta input file is a tsv, make fastafile from tsv
    if args.fasta.endswith('.tsv'):
        df = pandas.read_csv(args.fasta, sep='\t')
        with open(f'{args.out}/input_fasta.fasta', "w") as f:
            for index, row in df.iterrows():
                f.write(f">{row['Entry']}\n{row['Sequence']}\n")
        args.fasta = f'{args.out}/input_fasta.fasta'

    low_identity = filter_low_identity_sequences(args.fasta, similarity_threshold=args.redundancy_threshold)
    
    # create a new fastafile with the low identity sequences
    # load the fasta file
    seqs = load_fasta(args.fasta)
    low_identity_seqs = []
    for seq in seqs:
        if seq.id in low_identity:
            low_identity_seqs.append(seq)
            
    with open(f'{args.out}/{args.redundancy_threshold}filtered_for_redundancy.fasta', "w") as f:
        SeqIO.write(low_identity_seqs, f, "fasta")
        
    # also make a list of the ids of the low identity sequences, and ensure list is just the entry name
    low_identity = [x.id for x in low_identity_seqs]
    # check '|' in the ids, if so, split on '|' and take the second element
    if '|' in low_identity[0]:
        low_identity = [x.split('|')[1] for x in low_identity]
    # now write to text file
            
    # training ID set
    training_set = low_identity
    
                    
    # take a subset of the low identity sequences for use as a test set, take 5% of sequences
    test_set = random.sample(low_identity, int(len(low_identity) * 0.05))
    
    with open(f'{args.out}/filt90_TEST_set.txt', "w") as f:
        for i in test_set:
            f.write(i + "\n")
    
    exp_df = pandas.read_csv(args.experimental_data, sep='\t')
    exp_entries = exp_df['Entry'].tolist() # These are not low id but it shouldn't affect the results of following code.
    
    args.fasta = f'{args.out}/{args.redundancy_threshold}filtered_for_redundancy.fasta'
    
    low_identity = filter_singleton_sequences(args.fasta, similarity_threshold=0.3)
    
    if '|' in low_identity[0]:
        low_identity = [x.split('|')[1] for x in low_identity]
    
    mmseq_list_exp = [x for x in low_identity if x in exp_entries]
    
    # remove any seqs that are in test_set from the list of experimental low id
    mmseq_list_exp = [x for x in mmseq_list_exp if x not in test_set]
    
    print('num of experimental low id in mmseqs2 0.3 reduced set: ', len(mmseq_list_exp))
    
    with open(f'{args.out}/Low30_mmseq_ID_exp_subset.txt', "w") as f:
        for i in mmseq_list_exp:
            f.write(i + "\n")
            
    # now find the overlap in the mmseq_list_exp with the training_set, and remove those from the training set
    training_set = [x for x in training_set if x not in mmseq_list_exp]
    with open(f'{args.out}/training_IDs.txt', "w") as f:
        for i in training_set:
            f.write(i + "\n")
    
    
            
    
    
            
            
if __name__ == "__main__":
    main()