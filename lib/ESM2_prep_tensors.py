import pandas as pd
import glob
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
import os
from Bio import SeqIO

# Set up the parser
parser = argparse.ArgumentParser(description='Prepare tensors for ASTformer')
parser.add_argument('--metadata', type=str, help='Metadata tsv file from uniprot, containing act sites')
parser.add_argument('--emb_dir', type=str, help='Directory containing embeddings', required=True)
parser.add_argument('--emb_layer', type=int, help='Layer of embeddings to use', required=True)
parser.add_argument('--emb_size', type=int, help='Size of embeddings', required=True)
parser.add_argument('--max_len', type=int, help='Max length of the sequences', required=True)
parser.add_argument('--out', type=str, help='Output_tag_file_identifier', required=True)

args = parser.parse_args()

def load_fasta(fasta_file):
    seqs = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        seqs.append(record)
    return seqs

def manual_pad_sequence_tensors(tensors, target_length, padding_value=0):
    padded_tensors = []
    for tensor in tensors:
        # Check if padding is needed along the first dimension
        if tensor.size(0) < target_length:
            pad_size = target_length - tensor.size(0)
            # Create a padding tensor with the specified value
            padding_tensor = torch.full((pad_size, tensor.size(1)), padding_value, dtype=tensor.dtype, device=tensor.device)
            # Concatenate the padding tensor to the original tensor along the first dimension
            padded_tensor = torch.cat([tensor, padding_tensor])
        # If the tensor is longer than the target length, trim it along the first dimension
        else:
            padded_tensor = tensor[:target_length, :]
        padded_tensors.append(padded_tensor)
    return padded_tensors

def main(args):
    df = pd.read_csv(args.metadata, sep = "\t")	
    
    ids = df["Entry"]

    # drop all rows that don't have Entry in ids
    df = df[df["Entry"].isin(ids)]
    
    print(len(df))
    
    # get the active sites of each sequence in the dataset
    active_sites = []

    # iterate through the df and get the active sites
    for index, row in df.iterrows():
        active_site_string = row["Active site"]
        active_site_list = []
            
        intermediate_list = active_site_string.split(";")
        # iterate through the intermediate list and get the active sites
        for item in intermediate_list:
            if item.startswith("ACT_SITE") or item.startswith(" ACT_SITE"):
                active_site_list.append(item.split("ACT_SITE ")[1]) # we will correct indexing from 1 later
        active_sites.append(active_site_list)
            
    df["Active sites"] = active_sites
    
    # Convert the df to a dictionary, with Entry as its key
    dict_df = df.set_index("Entry").T.to_dict()
    
    # Now get all the embedding files
    files = []

    for file in glob.glob(args.emb_dir + "/**/*.pt", recursive=True):
        files.append(file)
        
    # get all entries in the dict_df
    entries = dict_df.keys()

    embeddings_tensors = []
    embeddings_labels = []
    for file in files:
        embedding_file = torch.load(file, map_location=torch.device('cpu'))
        tensor = embedding_file['representations'][args.emb_layer] # have to get the last layer of the embeddings...
        label = embedding_file['label']
        if label.split('|')[1] not in entries:
            continue
        embeddings_tensors.append(tensor)
        embeddings_labels.append(label)

    print(len(embeddings_tensors))
    print(len(embeddings_labels))

    paired_active_sites = []
    paired_labels = []
    for label in embeddings_labels:
        label = label.split("|")[1]
        if label not in entries:
            continue
        paired_active_sites.append(dict_df[label]["Active sites"])
        paired_labels.append(label)
    
    # Convert all paired_active_sites to a list lists of integers
    paired_active_sites = [[int(i) for i in sublist] for sublist in paired_active_sites]
    
    # Reorder the metadata df to match the order of the embeddings paired_labels
    # reorder them by the paired_labels
    df_paired = df.set_index("Entry")
    df_paired = df_paired.reindex(paired_labels)
    df_paired = df_paired.reset_index()
    
    tensors = embeddings_tensors
    active_sites = paired_active_sites
    meta_data = df_paired


    # NOTE: the max length should always be consistent! The model will only accept a specific length
    max_length = max(tensor.shape[0] for tensor in tensors)
    
    if max_length != args.max_len:
        # override and manually pad them
        max_length = args.max_len
        padded_tensor = manual_pad_sequence_tensors(tensors, max_length)
        padded_tensor = torch.stack(padded_tensor)
    else:
        print("Max length of the input tensors:", max_length)
        padded_tensor = pad_sequence(tensors, batch_first=True, padding_value=0.0)

    print("Shape of the padded input tensor:", padded_tensor.shape)  # (number of samples, max_length, embedding dimension)

    max_length = padded_tensor.shape[1]  # Maximum sequence length from the padded input tensor
    num_sequences = len(active_sites)

    # Initialize the target tensor with zeros
    targets = torch.zeros(num_sequences, max_length, dtype=torch.long)

    # Populate the target tensor
    for i, sites in enumerate(active_sites):
        #convert the strings in the list to integers
        sites = [int(site)-1 for site in sites] # take -1 from the site to get the correct index (index from 1 in uniprot)
        targets[i, sites] = 1  # Set active site positions to 1

    print("Shape of the targets tensor:", targets.shape)  # (number of samples, max_length)

    # free some space up by removing the now supurfluous variables
    del tensors
    del active_sites
    
    # Save the padded tensors and metadata to a very useful datalocation

    data_dir = args.out + "/"
    
    # make the directory if doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    print(df_paired)
    torch.save(padded_tensor, data_dir + "padded_tensor.pt")
    torch.save(targets, data_dir + "padded_targets_index0.pt")
    df_paired.to_csv(data_dir + "metadata_paired.tsv", sep='\t',index=False)


if __name__ == "__main__":
    main(args)