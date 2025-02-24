'''
The purpose of this script is to create the pair scheme used to train the contrastive embedding model for sequence token classification

Inputs: per token embeddings of sequences, metadata for each sequence (Active site positions, sequence, entry name)

Outputs: Torch datasets containing pairs of embeddings and a label (pos or neg) for each pair. Will be split into training and testing.


Rules:
    - pairs must always be the same amino acid (i.e. His paired only with His)
    - pairs must only be paired if part of the same EC class up to the 2nd tier
    - For now we will exclude sequences that are part of the validation set. We will only train on the training set.

The pairs are generated as follows:
    - For each sequence, generate pairs of active site positions and non active site positions WITHIN the same sequence
    - For each active site position, generate positive pairs of active site positions
    - For each non active site position, generate positive pairs of non active site positions
    - For each active site position, generate negative pairs of non active site positions
'''

import warnings
import pandas as pd
import glob
import pickle
import argparse
import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm
import math
import os
import time

import matplotlib.pyplot as plt
import seaborn as sns

import collections

# Suppress the PerformanceWarning from pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# set the seed
seed = 420
random.seed(seed)


def arg_parser():
    parser = argparse.ArgumentParser(description='Train a model on the ASTformer dataset')
    parser.add_argument('--embedding_dir', 
                        type=str, 
                        help='Path to the directory containing preprocessed (padded) tensor of embeddings data')

    parser.add_argument('--metadata', 
                        type=str,
                        help='Path to the metadata for each sequences... must be in same order as embeddings/targets')

    parser.add_argument('--sample_limit',
                        type=int,
                        default=600,
                        help='The maximum number of samples to take from each EC class/ amino acid combo')
    
    parser.add_argument('--scheme',
                        type=int,
                        choices=[1,2,3,4,5,6],
                        help="select the pairscheme protocol used. See readme for more info")
    
    parser.add_argument('--eval',
                        type=str,
                        help='Path to the evaluation set of sequences')
    
    parser.add_argument('--test',
                        type=str,
                        help='Path to the test set of sequences')
    
    parser.add_argument('--create_torch',
                        action='store_true',
                        help='If true, the code will save the pair scheme as a torch dataset with the embeddings. Else, it will save only the location of the embeddings and labels.')

    parser.add_argument('--out', 
                        type=str, 
                        help='Path to the directory to save the pair scheme.')
    
    parser.add_argument('--layer',
                        type=int,
                        default=36,
                        help='The layer of the embeddings to use for the pairs')
    
    parser.add_argument('--BSvAS', 
                        type=str, 
                        choices=['BS', 'AS'],
                        help='If BS, the script will generate the binding site positions from the uniprot tsv file. If AS, the script will generate the active site positions from the uniprot tsv file.')
    
    parser.add_argument('--leaked_entries',
                        type=str,
                        required=False,
                        default=None,
                        help='Path to the file containing the entries that are in the training set that I want to test with')
    
    args = parser.parse_args()
    return args


def split_entries(dict_df, train = 0.8, test = 0.2, val = 0):
    entries = list(dict_df.keys())

    # shuffle the entries
    random.shuffle(entries)

    # split the entries into training, test and validation sets
    train_entries = entries[:int(len(entries)*train)]
    test_entries = entries[int(len(train_entries)):int(len(entries)*(test+train))]
    if val != 0:
        val_entries = entries[int(len(entries)*test+train):]
        return train_entries, test_entries, val_entries
    else:
        return train_entries, test_entries


def get_embeddings(dir,layer=-1):
    embeddings_tensors = []
    embeddings_labels = []
    count = 0
    files = glob.glob(dir + '/*.pt')
    for file in files:
        embedding_file = torch.load(file) 
        tensor = embedding_file['representations'][layer] # have to get the last layer (48) of the embeddings...
        label = embedding_file['label']
        embeddings_tensors.append(tensor)
        embeddings_labels.append(label.split('|')[1]) # file dependant TODO: change this
    return embeddings_tensors, embeddings_labels
               
               
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


def get_embedding_token(pos, entry, dir, layer=48):
    file = dir + '/sp|' + entry + '|esm' + '.pt'
    with open(file, 'rb') as f:
        embedding_file = torch.load(f)
    return (embedding_file['representations'][layer])[pos,:]


def get_AS_pos_from_uniprot(df):
    active_sites = []
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
    return active_sites


# I specifically had the data in the format already ... TODO: change this to be more general or make new function for others to use	
def get_BS_pos_from_uniprot_processed_tsv(df):
    binding_sites = []
    #count how often there are duplicates in the binding sites list
    count = 0
    for index, row in df.iterrows():
        binding_site_string = row["Cofactor sites"]
        binding_site_string = binding_site_string.replace("[", "").replace("]", "").replace(" ", "").replace("(", "").replace(")", "")
        binding_site_string = binding_site_string.replace("np.int64", "")
        binding_site_string = [int(x) for x in binding_site_string.split(",")]
        if len(binding_site_string) != len(set(binding_site_string)):
            count +=1
        binding_sites.append(binding_site_string)
    print("Number of sequences that have duplicate bs positions: ", count)
    
    binding_sites_ids = []
    for index, row in df.iterrows():
        binding_id_string = row["Cofactor IDs"]
        binding_id_string = binding_id_string.replace("[", "").replace("]", "").replace(" ", "").replace("(", "").replace(")", "")
        binding_id_string = binding_id_string.replace("np.int64", "")
        binding_id_string = [int(x) for x in binding_id_string.split(",")]
        binding_sites_ids.append(binding_id_string)
        
        
    # figure out which positions in the binding site string have duplicates and then get the corresponding ID from the ID string
    for i in range(len(binding_sites)):
        binding_site = binding_sites[i]
        binding_site_ids_IDC = binding_sites_ids[i]
        duplicates = [item for item, count in collections.Counter(binding_site).items() if count > 1]
        for duplicate in duplicates:
            idx = [i for i, x in enumerate(binding_site) if x == duplicate]
            for y in idx:
                # print the duplicate position and the corresponding ID
                print(f"{i}: Duplicate position: {duplicate} with ID: {binding_site_ids_IDC[y]}")
        
    return binding_sites, binding_sites_ids


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


def get_BS_idx_from_uniprot(df):
    binding_sites = []
    # iterate through the df and get the binding sites
    for index, row in df.iterrows():
        binding_site_string = row["Binding site"]
        binding_site_list = []
        intermediate_list = binding_site_string.split(";")
        #iterate through the intermediate list and get the binding sites
        for item in intermediate_list:
            if item.startswith("BINDING") or item.startswith(" BINDING"):
                binding_site_list.append(int(item.split("BINDING ")[1])-1)
        binding_sites.append(binding_site_list)
    return binding_sites


# completely binary -- all AS vs all NAS etc with no consideration for EC class or residue.
def scheme_1(df, embeddings_dir, train_entries, non_AS_samplerate = 0.05, EC = False, sample_limit = 1000000, ASvBS = 'AS', layer = 36):
    
    all_paired_embeddings = []
    all_labels = []
    
    all_AS_positions = []
    non_AS_positions = []
    for i, row in df.iterrows():
        if row["Entry"] in train_entries:
            active_sites = row[ASvBS]
            entry = row["Entry"]
            
            for site in active_sites:
                pos = (site, entry, embeddings_dir)
                all_AS_positions.append(pos)
            # Take a random sample of 0.X of the non active site embeddings and add them to non_AS_embeddings
            non_AS_sites = random.sample(range(0, len(row["Sequence"])), int(len(row["Sequence"])*non_AS_samplerate))
            for site in non_AS_sites:
                if site not in active_sites:
                    pos = (site, entry, embeddings_dir)
                    non_AS_positions.append(pos)
    
    # subset the positions to a random set to control the explosion of possible combinations - improves speed and doesn't impact final number of pairs - shouldn't at least using the sqrt
    if len(all_AS_positions) > math.ceil(math.sqrt(sample_limit/2)):
        all_AS_positions = random.sample(all_AS_positions, math.ceil(math.sqrt(sample_limit/2)))
    if len(non_AS_positions) > math.ceil(math.sqrt(sample_limit/2)):
        non_AS_positions = random.sample(non_AS_positions, math.ceil(math.sqrt(sample_limit/2)))
    
    rows = []
    for pos in all_AS_positions:
        rows.append([True, pos])
    for pos in non_AS_positions:
        rows.append([False, pos])
        
    
    site_embedding_pool = []
    site_embedding_pool_labels = []
    # create a df from the rows
    df = pd.DataFrame(rows, columns=['AS', 'pos'])
    
    # take the pos column and find the set of all unique pos
    unique_pos = list(df['pos'].unique())
    
    print(unique_pos[0:10])
    
    # create a new df from the unique pos, with site, entry, embeddings_dir as columns and the index as the pos
    unique_pos_df = pd.DataFrame(unique_pos, columns=['site', 'entry', 'embeddings_dir'])
    
    # now iterate through each unique entry name and extract all the sites from the corresponding file and put them in a list
    unique_entries = unique_pos_df['entry'].unique()
    for entry in tqdm(unique_entries):
        # subset the unique_pos_df to only the rows where the entry is the same as the current entry
        unique_entry_df = unique_pos_df[unique_pos_df['entry'] == entry]
        # get all the sites in a list
        sites = unique_entry_df['site'].tolist()
        
        emb_dir = unique_entry_df['embeddings_dir'].iloc[0]
        embedding = torch.load(emb_dir + '/sp|' + entry + '|esm' + '.pt')
        embedding = embedding['representations'][layer]
        site_embeddings = embedding[sites]
        site_embedding_pool.extend(site_embeddings)
        
        # make a list of the labels for the site embeddings
        sites = [entry+'_'+str(site) for site in sites]
        site_embedding_pool_labels.extend(sites)
    
    # create a dictionary with the site embeddings and the corresponding labels
    site_embedding_dict = dict(zip(site_embedding_pool_labels, site_embedding_pool))
    
    # now iterate through the df and replace the pos with the key for looking up the embedding in the dictionary
    for i, row in df.iterrows():
        entry = row['pos'][1]
        site = row['pos'][0]
        key = entry + '_' + str(site)
        df.at[i, 'pos'] = key
        
    #split the df into AS and NAS
    AS_df = df[df['AS'] == True]
    NAS_df = df[df['AS'] == False]
        
    all_AS_positions = [pos for pos in AS_df['pos']]
    non_AS_positions = [pos for pos in NAS_df['pos']]
        
    ASvAS_paired_positions = []
    NASvNAS_paired_positions = []
    ASvNAS_paired_positions = []
        
    if len(non_AS_positions) != 0:
        #do the same for the non_AS_embeddings
        for i in range(len(non_AS_positions)):
            for j in range(i+1, len(non_AS_positions)):
                ASvAS_paired_positions.append((non_AS_positions[i], non_AS_positions[j]))
    # control the sample rate
    if len(ASvAS_paired_positions) > sample_limit:
        ASvAS_paired_positions = random.sample(ASvAS_paired_positions, sample_limit)
    ASvAS_labels = [1]*len(ASvAS_paired_positions)
            
    if len(all_AS_positions) != 0:
        for i in range(len(all_AS_positions)):
            for j in range(i+1, len(all_AS_positions)):
                NASvNAS_paired_positions.append((all_AS_positions[i], all_AS_positions[j]))
    # control the sample rate
    if len(NASvNAS_paired_positions) > sample_limit:
        NASvNAS_paired_positions = random.sample(NASvNAS_paired_positions, sample_limit)
    NASvNAS_labels = [1]*len(NASvNAS_paired_positions)
            
    if len(all_AS_positions) != 0 or len(non_AS_positions) != 0:
        # Now make random pairs between the AS and non AS embeddings
        for i in range(len(all_AS_positions)):
            for j in range(len(non_AS_positions)):
                ASvNAS_paired_positions.append((all_AS_positions[i], non_AS_positions[j]))
    # control the sample rate
    if len(ASvNAS_paired_positions) > 2*sample_limit:
        ASvNAS_paired_positions = random.sample(ASvNAS_paired_positions, 2*sample_limit)
    ASvNAS_labels = [-1]*len(ASvNAS_paired_positions)
        
    all_paired_embeddings.extend(ASvAS_paired_positions)
    all_labels.extend(ASvAS_labels)
    all_paired_embeddings.extend(NASvNAS_paired_positions)
    all_labels.extend(NASvNAS_labels)
    all_paired_embeddings.extend(ASvNAS_paired_positions)
    all_labels.extend(ASvNAS_labels)

    return all_paired_embeddings, all_labels, site_embedding_dict


# Loosely created EC scheme, where AS/NAS contrastive pairs are made only within EC number/Residue groups. No negative contrasts included for seperating the EC classes further from each other...
def scheme_2(df, embeddings_dir, train_entries, non_AS_samplerate = 0.02, EC = True, sample_limit = 600, ASvBS = 'AS', layer=36):
    
    # get all unique EC numbers in EC_TX
    unique_EC_TX = df['EC_TX'].explode().unique()
    print("Unique EC numbers: ")
    print(unique_EC_TX)
    
    # all possible amino acids in a set
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    all_paired_embeddings = []
    all_labels = []
    
    rows = []
    for EC in tqdm(unique_EC_TX):
        EC_df = df[df['EC_TX'].apply(lambda x: EC in x)]
        if len(EC_df) < 10:
            non_AS_samplerate = 1
        elif len(EC_df) < 100:
            non_AS_samplerate = 0.3
        elif len(EC_df) < 500:
            non_AS_samplerate = 0.1
        else:
            non_AS_samplerate = 0.05
        for AA in amino_acids:
            all_AS_positions = []
            non_AS_positions = []
            for i, row in EC_df.iterrows():
                if row["Entry"] in train_entries:
                    active_sites = row[ASvBS]
                    entry = row["Entry"]
                    
                    # get all positions with AA in the sequence
                    AA_positions = [pos for pos, char in enumerate(row["Sequence"]) if char == AA]
                    
                    for site in active_sites:
                        if site in AA_positions:
                            pos = (site, entry, embeddings_dir)
                            all_AS_positions.append(pos)
                    # Take a random sample of 0.1 of the non active site embeddings and add them to non_AS_embeddings
                    non_AS_sites = random.sample(range(0, len(row["Sequence"])), int(len(row["Sequence"])*non_AS_samplerate))
                    for site in non_AS_sites:
                        if site not in active_sites:
                            if site in AA_positions:
                                pos = (site, entry, embeddings_dir)
                                non_AS_positions.append(pos)
            
            # subset the positions to a random set of 300 to control the explosion of possible combinations - improves speed and doesn't impact final number of pairs
            if len(all_AS_positions) > 600:
                all_AS_positions = random.sample(all_AS_positions, 600)
            if len(non_AS_positions) > 600:
                non_AS_positions = random.sample(non_AS_positions, 600)
            
            for pos in all_AS_positions:
                rows.append([EC, AA, True, pos])
            for pos in non_AS_positions:
                rows.append([EC, AA, False, pos])
    
    site_embedding_pool = []
    site_embedding_pool_labels = []
    # create a df from the rows
    df = pd.DataFrame(rows, columns=['EC', 'AA', 'AS', 'pos'])
    
    # take the pos column and find the set of all unique pos
    unique_pos = list(df['pos'].unique())
    
    print(unique_pos[0:10])
    
    # create a new df from the unique pos, with site, entry, embeddings_dir as columns and the index as the pos
    unique_pos_df = pd.DataFrame(unique_pos, columns=['site', 'entry', 'embeddings_dir'])
    
    # now iterate through each unique entry name and extract all the sites from the corresponding file and put them in a list
    unique_entries = unique_pos_df['entry'].unique()
    for entry in tqdm(unique_entries):
        # subset the unique_pos_df to only the rows where the entry is the same as the current entry
        unique_entry_df = unique_pos_df[unique_pos_df['entry'] == entry]
        # get all the sites in a list
        sites = unique_entry_df['site'].tolist()
        
        emb_dir = unique_entry_df['embeddings_dir'].iloc[0]
        embedding = torch.load(emb_dir + '/sp|' + entry + '|esm' + '.pt')
        embedding = embedding['representations'][layer]
        site_embeddings = embedding[sites]
        site_embedding_pool.extend(site_embeddings)
        
        # make a list of the labels for the site embeddings
        sites = [entry+'_'+str(site) for site in sites]
        site_embedding_pool_labels.extend(sites)
    
    # create a dictionary with the site embeddings and the corresponding labels
    site_embedding_dict = dict(zip(site_embedding_pool_labels, site_embedding_pool))
    
    # now iterate through the df and replace the pos with the key for looking up the embedding in the dictionary
    for i, row in df.iterrows():
        entry = row['pos'][1]
        site = row['pos'][0]
        key = entry + '_' + str(site)
        df.at[i, 'pos'] = key
    
    
    for EC in tqdm(unique_EC_TX):
        for AA in amino_acids:
            # subset the df to df where all EC AA and AS are same
            same_EC = df[df['EC'] == EC]
            same_EC_AA = same_EC[same_EC['AA'] == AA]
            same_EC_AA_AS = same_EC_AA[same_EC_AA['AS'] == True]
            same_EC_AA_NAS = same_EC_AA[same_EC_AA['AS'] == False]
            
            if len(same_EC_AA_AS) < 1 or len(same_EC_AA_NAS) < 1:
                continue
            
            all_AS_positions = [pos for pos in same_EC_AA_AS['pos']]
            non_AS_positions = [pos for pos in same_EC_AA_NAS['pos']]
                        
            # Take all possible unique pairs of the AS embeddings and save them in a list
            ASvAS_paired_positions = []
            NASvNAS_paired_positions = []
            ASvNAS_paired_positions = []
            
            if len(non_AS_positions) != 0:
                #do the same for the non_AS_embeddings
                for i in range(len(non_AS_positions)):
                    for j in range(i+1, len(non_AS_positions)):
                        ASvAS_paired_positions.append((non_AS_positions[i], non_AS_positions[j]))
            # control the sample rate
            if len(ASvAS_paired_positions) > sample_limit:
                ASvAS_paired_positions = random.sample(ASvAS_paired_positions, sample_limit)
            ASvAS_labels = [1]*len(ASvAS_paired_positions)
            
            if len(all_AS_positions) != 0:
                for i in range(len(all_AS_positions)):
                    for j in range(i+1, len(all_AS_positions)):
                        NASvNAS_paired_positions.append((all_AS_positions[i], all_AS_positions[j]))
            # control the sample rate
            if len(NASvNAS_paired_positions) > sample_limit:
                NASvNAS_paired_positions = random.sample(NASvNAS_paired_positions, sample_limit)
            NASvNAS_labels = [1]*len(NASvNAS_paired_positions)
            
            if len(all_AS_positions) != 0 or len(non_AS_positions) != 0:
                # Now make random pairs between the AS and non AS embeddings
                for i in range(len(all_AS_positions)):
                    for j in range(len(non_AS_positions)):
                        ASvNAS_paired_positions.append((all_AS_positions[i], non_AS_positions[j]))
            # control the sample rate
            if len(ASvNAS_paired_positions) > sample_limit:
                ASvNAS_paired_positions = random.sample(ASvNAS_paired_positions, sample_limit)
            ASvNAS_labels = [-1]*len(ASvNAS_paired_positions)

            all_paired_embeddings.extend(ASvAS_paired_positions)
            all_labels.extend(ASvAS_labels)
            all_paired_embeddings.extend(NASvNAS_paired_positions)
            all_labels.extend(NASvNAS_labels)
            all_paired_embeddings.extend(ASvNAS_paired_positions)
            all_labels.extend(ASvNAS_labels)
            
            
    return all_paired_embeddings, all_labels, site_embedding_dict


# Scheme 2, but adding the EC class negative pairs, so that we contrast the EC numbers within pairs of AS/AS and NAS/NAS of the same residue.
def scheme_3(df, embeddings_dir, train_entries, non_AS_samplerate = 0.02, EC = True, sample_limit = 600, ASvBS = 'AS', layer=36):
    
    # get all unique EC numbers in EC_TX
    unique_EC_TX = df['EC_TX'].explode().unique()
    print("Unique EC numbers: ")
    print(unique_EC_TX)
    
    # all possible amino acids in a set
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    all_paired_embeddings = []
    all_labels = []
    
    rows = []
    for EC in tqdm(unique_EC_TX):
        EC_df = df[df['EC_TX'].apply(lambda x: EC in x)]
        if len(EC_df) < 10:
            non_AS_samplerate = 1
        elif len(EC_df) < 100:
            non_AS_samplerate = 0.3
        elif len(EC_df) < 500:
            non_AS_samplerate = 0.1
        else:
            non_AS_samplerate = 0.05
        for AA in amino_acids:
            all_AS_positions = []
            non_AS_positions = []
            for i, row in EC_df.iterrows():
                if row["Entry"] in train_entries:
                    active_sites = row[ASvBS]
                    entry = row["Entry"]
                    
                    # get all positions with AA in the sequence
                    AA_positions = [pos for pos, char in enumerate(row["Sequence"]) if char == AA]
                    
                    for site in active_sites:
                        if site in AA_positions:
                            pos = (site, entry, embeddings_dir)
                            all_AS_positions.append(pos)
                    # Take a random sample of 0.1 of the non active site embeddings and add them to non_AS_embeddings
                    non_AS_sites = random.sample(range(0, len(row["Sequence"])), int(len(row["Sequence"])*non_AS_samplerate))
                    for site in non_AS_sites:
                        if site not in active_sites:
                            if site in AA_positions:
                                pos = (site, entry, embeddings_dir)
                                non_AS_positions.append(pos)
            
            # subset the positions to a random set of 300 to control the explosion of possible combinations - improves speed and doesn't impact final number of pairs
            if len(all_AS_positions) > 200:
                all_AS_positions = random.sample(all_AS_positions, 200)
            if len(non_AS_positions) > 200:
                non_AS_positions = random.sample(non_AS_positions, 200)
            
            for pos in all_AS_positions:
                rows.append([EC, AA, True, pos])
            for pos in non_AS_positions:
                rows.append([EC, AA, False, pos])
    
    site_embedding_pool = []
    site_embedding_pool_labels = []
    # create a df from the rows
    df = pd.DataFrame(rows, columns=['EC', 'AA', 'AS', 'pos'])
    
    # take the pos column and find the set of all unique pos
    unique_pos = list(df['pos'].unique())
    
    print(unique_pos[0:10])
    
    # create a new df from the unique pos, with site, entry, embeddings_dir as columns and the index as the pos
    unique_pos_df = pd.DataFrame(unique_pos, columns=['site', 'entry', 'embeddings_dir'])
    
    # now iterate through each unique entry name and extract all the sites from the corresponding file and put them in a list
    unique_entries = unique_pos_df['entry'].unique()
    for entry in tqdm(unique_entries):
        # subset the unique_pos_df to only the rows where the entry is the same as the current entry
        unique_entry_df = unique_pos_df[unique_pos_df['entry'] == entry]
        # get all the sites in a list
        sites = unique_entry_df['site'].tolist()
        
        emb_dir = unique_entry_df['embeddings_dir'].iloc[0]
        embedding = torch.load(emb_dir + '/sp|' + entry + '|esm' + '.pt')
        embedding = embedding['representations'][layer]
        site_embeddings = embedding[sites]
        site_embedding_pool.extend(site_embeddings)
        
        # make a list of the labels for the site embeddings
        sites = [entry+'_'+str(site) for site in sites]
        site_embedding_pool_labels.extend(sites)
    
    # create a dictionary with the site embeddings and the corresponding labels
    site_embedding_dict = dict(zip(site_embedding_pool_labels, site_embedding_pool))
    
    # now iterate through the df and replace the pos with the key for looking up the embedding in the dictionary
    for i, row in df.iterrows():
        entry = row['pos'][1]
        site = row['pos'][0]
        key = entry + '_' + str(site)
        df.at[i, 'pos'] = key
    
    
    for EC in tqdm(unique_EC_TX):
        for AA in amino_acids:
            # subset the df to df where all EC AA and AS are same
            same_EC = df[df['EC'] == EC]
            same_EC_AA = same_EC[same_EC['AA'] == AA]
            same_EC_AA_AS = same_EC_AA[same_EC_AA['AS'] == True]
            same_EC_AA_NAS = same_EC_AA[same_EC_AA['AS'] == False]
            
            if len(same_EC_AA_AS) < 1 or len(same_EC_AA_NAS) < 1:
                continue
            
            all_AS_positions = [pos for pos in same_EC_AA_AS['pos']]
            non_AS_positions = [pos for pos in same_EC_AA_NAS['pos']]
                        
            # Take all possible unique pairs of the AS embeddings and save them in a list
            ASvAS_paired_positions = []
            NASvNAS_paired_positions = []
            ASvNAS_paired_positions = []
            
            if len(non_AS_positions) != 0:
                #do the same for the non_AS_embeddings
                for i in range(len(non_AS_positions)):
                    for j in range(i+1, len(non_AS_positions)):
                        ASvAS_paired_positions.append((non_AS_positions[i], non_AS_positions[j]))
            # control the sample rate
            if len(ASvAS_paired_positions) > sample_limit:
                ASvAS_paired_positions = random.sample(ASvAS_paired_positions, sample_limit)
            ASvAS_labels = [1]*len(ASvAS_paired_positions)
            
            if len(all_AS_positions) != 0:
                for i in range(len(all_AS_positions)):
                    for j in range(i+1, len(all_AS_positions)):
                        NASvNAS_paired_positions.append((all_AS_positions[i], all_AS_positions[j]))
            # control the sample rate
            if len(NASvNAS_paired_positions) > sample_limit:
                NASvNAS_paired_positions = random.sample(NASvNAS_paired_positions, sample_limit)
            NASvNAS_labels = [1]*len(NASvNAS_paired_positions)
            
            if len(all_AS_positions) != 0 or len(non_AS_positions) != 0:
                # Now make random pairs between the AS and non AS embeddings
                for i in range(len(all_AS_positions)):
                    for j in range(len(non_AS_positions)):
                        ASvNAS_paired_positions.append((all_AS_positions[i], non_AS_positions[j]))
            # control the sample rate
            if len(ASvNAS_paired_positions) > sample_limit:
                ASvNAS_paired_positions = random.sample(ASvNAS_paired_positions, sample_limit)
            ASvNAS_labels = [-1]*len(ASvNAS_paired_positions)

            all_paired_embeddings.extend(ASvAS_paired_positions)
            all_labels.extend(ASvAS_labels)
            all_paired_embeddings.extend(NASvNAS_paired_positions)
            all_labels.extend(NASvNAS_labels)
            all_paired_embeddings.extend(ASvNAS_paired_positions)
            all_labels.extend(ASvNAS_labels)
            
            
            AS_EC_negative_pairs=[]
            NAS_EC_negative_pairs=[]
            
            # subset the df to df wher EC not the same but AA and AS yes
            same_AA = df[df['AA'] == AA]
            same_AA = same_AA[same_AA['EC'] != EC]
            same_AA_AS = same_AA[same_AA['AS'] == True]
            same_AA_NAS = same_AA[same_AA['AS'] == False]
                
            # append a random sample of pos from same_AA_AS to all_AS_positions - less than 300
            if len(same_AA_AS) > 200:
                same_AA_AS = same_AA_AS.sample(n=200)
            same_AA_AS_positions = [pos for pos in same_AA_AS['pos']]
            
            if len(all_AS_positions) != 0 or len(same_AA_AS_positions) != 0:
                # Now make random pairs between the AS and non AS embeddings
                for i in range(len(all_AS_positions)):
                    for j in range(len(same_AA_AS_positions)):
                        AS_EC_negative_pairs.append((all_AS_positions[i], same_AA_AS_positions[j]))
            # control the sample rate
            if len(AS_EC_negative_pairs) > sample_limit:
                AS_EC_negative_pairs = random.sample(AS_EC_negative_pairs, sample_limit)
            same_AA_AS_labels = [-1]*len(AS_EC_negative_pairs)
            
            if len(same_AA_NAS) > 200:
                same_AA_NAS = same_AA_NAS.sample(n=200)
            same_AA_NAS_positions = [pos for pos in same_AA_NAS['pos']]
            
            if len(all_AS_positions) != 0 or len(same_AA_NAS_positions) != 0:
                # Now make random pairs between the AS and non AS embeddings
                for i in range(len(all_AS_positions)):
                    for j in range(len(same_AA_NAS_positions)):
                        NAS_EC_negative_pairs.append((all_AS_positions[i], same_AA_NAS_positions[j]))
            # control the sample rate
            if len(NAS_EC_negative_pairs) > sample_limit:
                NAS_EC_negative_pairs = random.sample(NAS_EC_negative_pairs, sample_limit)
            same_AA_NAS_labels = [-1]*len(NAS_EC_negative_pairs)
            
            all_paired_embeddings.extend(AS_EC_negative_pairs)
            all_labels.extend(same_AA_AS_labels)
            all_paired_embeddings.extend(NAS_EC_negative_pairs)
            all_labels.extend(same_AA_NAS_labels)

    return all_paired_embeddings, all_labels, site_embedding_dict


# Important cols : Cofactor sites	Cofactor IDs	EC_tier1	EC_tier2	EC_tier3	EC_tier4
def Flexible_EC_labeling(df, EC_threshold = 50, out_dir = ''):
    # for each EC_tier col, fix the fact that they are stored as a string, not a list of strings:
    for i, row in df.iterrows():
        for col in ['EC_tier1', 'EC_tier2', 'EC_tier3', 'EC_tier4']:
            ECs = row[col]
            ECs = ECs.replace("[", "").replace("]", "").replace(" ", "").replace("(", "").replace(")", "")
            ECs = ECs.replace("'", "")
            ECs = ECs.split(",")
            ECs = [str(x.strip()) for x in ECs]
            df.at[i, col] = ECs
    
    df['EC_T-filtered'] = df['EC_tier1']
    
    EC_T4_counts = df['EC_tier4'].explode().value_counts()
    EC_T4_counts = EC_T4_counts[EC_T4_counts > EC_threshold]
    for EC in EC_T4_counts.index:
        if '-' in EC:
            EC_T4_counts = EC_T4_counts.drop(EC)
    EC_T4 = EC_T4_counts.index
    for i, row in df.iterrows():
        new_ECs = []
        for EC4 in row['EC_tier4']:
            if EC4 in EC_T4:
                new_ECs.append(EC4)
        if new_ECs != []:
            df.at[i, 'EC_T-filtered'] = new_ECs
            df.at[i, 'EC_tier3'] = []
            
            
    EC_T3_counts = df['EC_tier3'].explode().value_counts()
    EC_T3_counts = EC_T3_counts[EC_T3_counts > EC_threshold]
    for EC in EC_T3_counts.index:
        if '-' in EC:
            EC_T3_counts = EC_T3_counts.drop(EC)
    EC_T3 = EC_T3_counts.index
    for i, row in df.iterrows():
        new_ECs = []
        if len(row['EC_tier3']) == 0:
            continue
        for EC3 in row['EC_tier3']:
            if EC3 in EC_T3:
                new_ECs.append(EC3)
        if new_ECs != []:
            df.at[i, 'EC_T-filtered'] = new_ECs
            df.at[i, 'EC_tier2'] = []
    
    EC_T2_counts = df['EC_tier2'].explode().value_counts()
    EC_T2_counts = EC_T2_counts[EC_T2_counts > EC_threshold]
    for EC in EC_T2_counts.index:
        if '-' in EC:
            EC_T2_counts = EC_T2_counts.drop(EC)
    EC_T2 = EC_T2_counts.index
    for i, row in df.iterrows():
        new_ECs = []
        if len(row['EC_tier2']) == 0:
            continue
        for EC2 in row['EC_tier2']:
            if EC2 in EC_T2:
                new_ECs.append(EC2)
        if new_ECs != []:
            df.at[i, 'EC_T-filtered'] = new_ECs     
                    
    EC_T_filtered_counts = df['EC_T-filtered'].explode().value_counts()
    print("EC_T_filtered value counts: ")
    print(EC_T_filtered_counts)
    
    # How many of the sequences in the df have an EC in EC_T-filtered
    count = 0
    for i, row in df.iterrows():
        if len(row['EC_T-filtered']) > 0:
            count+=1
    print("Number of sequences with an EC in EC_T-filtered: ", count, "out of ", len(df))
    
    # now go through the df and pick one of the EC numbers if there are multiple in the row's EC_T-filtered
    # if one of the EC numbers is higher tier over others, take that one
    # otherwise, do it randomly
    count = 0
    for i, row in df.iterrows():
        ECs = row['EC_T-filtered']
        if len(row['EC_T-filtered']) > 1:
            tiers = [len(EC.split('.')) for EC in ECs]
            highest_tier = max(tiers)
            ECs = [EC for EC in ECs if len(EC.split('.')) == highest_tier]
            df.at[i, 'EC_T-filtered'] = random.choice(ECs)
            count += 1
        else:
            df.at[i, 'EC_T-filtered'] = ECs[0]
    
    print("Number of sequences that had multiple ECs in EC_T-filtered: ", count, "out of ", len(df))
            
                
    # print unique vals in EC_T-filtered
    unique_EC_TX = df['EC_T-filtered'].unique()
    print("Unique EC numbers: ", unique_EC_TX)
    
    #TODO fix this it looks like shite
    # plot a distribution of the value counts for the EC_T-filtered
    EC_T_filtered_counts = df['EC_T-filtered'].explode().value_counts()
    print("EC_T_filtered value counts: ")
    print(EC_T_filtered_counts)

    plt.figure(figsize=(26,10))
    plt.bar(EC_T_filtered_counts.index, EC_T_filtered_counts.values)
    # set the x-axis to be rotated 45 degrees
    plt.xticks(rotation=60)
    # add a title
    plt.title('Distribution of EC_T-filtered')
    # add x and y labels
    plt.xlabel('EC number')
    plt.ylabel('Frequency')
    # save the plot
    plt.savefig(out_dir + 'EC_T_filtered_distribution.png')
    
    return df
    
    
def BS_site_ID_column_splitting(df):
    # quick check to see how many sequences have a list of BS IDs that is not the same length as the list of BS sites
    count = 0
    for i, row in df.iterrows():
        if len(row['BS']) != len(row['Cofactor IDs']):
            #print("Sequence: ", row['Entry'])
            #print("Cofactor sites: ", row['BS'])
            #print("Cofactor IDs: ", row['Cofactor IDs'])
            count+=1
    print("Number of sequences with different lengths of BS sites and BS IDs: ", count, "out of ", len(df))
    # get the unique BS IDs
    unique_BS_IDs = df['Cofactor IDs'].explode().unique()
    # Create a dictionary to initialize columns for each unique BS_ID with empty lists
    BS_ID_dict = {BS_ID: [[] for _ in range(len(df))] for BS_ID in unique_BS_IDs}
    for i, row in df.iterrows():
        BS_positions = row['BS']
        Cofactor_IDs = row['Cofactor IDs']
        
        # Check if the length of Cofactor_IDs matches BS_positions to prevent out-of-range errors
        if len(Cofactor_IDs) != len(BS_positions):
            print(f"Row {i} has a mismatch between BS positions and Cofactor IDs lengths")
            raise ValueError

        for y, pos in enumerate(BS_positions):
            BS_ID = Cofactor_IDs[y]
            if BS_ID in BS_ID_dict:  # Ensure the BS_ID exists in the dictionary
                if i < len(BS_ID_dict[BS_ID]):  # Check to avoid index out of range
                    BS_ID_dict[BS_ID][i].append(pos)
                else:
                    print(f"Index {i} out of range for BS_ID {BS_ID}")
                    raise ValueError

    # Add the BS_ID columns to the original dataframe
    for BS_ID, positions_list in BS_ID_dict.items():
        df[BS_ID] = positions_list

    # Now go through the rows and find any duplicate positions in the BS columns
    for i, row in df.iterrows():
        positions_all = row['BS']
        if len(positions_all) != len(set(positions_all)):
            for pos in set(positions_all):
                if positions_all.count(pos) > 1:
                    indices = [j for j, x in enumerate(positions_all) if x == pos]
                    BS_IDs_with_dup = [row['Cofactor IDs'][j] for j in indices]
                    drop = []
                    # Select a random BS_ID that is not 0 and drop the rest
                    # First remove the 0s from the list
                    BS_IDs_with_dup_no_zero = [x for x in BS_IDs_with_dup if x != 0]
                    if len(BS_IDs_with_dup_no_zero) < len(BS_IDs_with_dup):
                        drop.append(0)
                    # if there are still duplicates, drop all but one
                    if len(BS_IDs_with_dup_no_zero) > 1:
                        # if the duplicate positions have different IDs, remove all the positions because it's cursed and not usable position
                        if len(set(BS_IDs_with_dup_no_zero)) > 1:
                            drop.extend(BS_IDs_with_dup_no_zero)
                        # Otherwise just keep the duplicates because they are the exact same position with the same IDs - probs coordinate 2 of the same cofactor
                    
                    for BS_ID in drop:
                        if BS_ID in df.columns:
                            df.at[i, BS_ID] = [x for x in df.at[i, BS_ID] if x != pos]
                            
    # double check there aren't any positions existing in multiple BS_ID columns - it will really break the model in training.
    for i, row in df.iterrows():
        positions_all = row['BS']
        for pos in positions_all:
            if sum([1 for BS_ID in unique_BS_IDs if pos in df.at[i, BS_ID]]) > 1:
                print("Position ", pos, " is in multiple BS_ID columns for sequence ", row['Entry'])
    
    df = df.copy()
    
    return df


# Binding site specific!
def scheme_4(df, embeddings_dir, train_entries, non_AS_samplerate = 0.02, EC = "EC_T-filtered", sample_limit = 600):
     # get all unique EC numbers in EC_TX
    unique_EC_TX = df[EC].explode().unique()
    print("Unique EC numbers: ")
    print(unique_EC_TX)
    
    all_paired_embeddings = []
    all_labels = []
    
    # get the unique set of Cofactor IDs in the df
    unique_BS_IDs = df['Cofactor IDs'].explode().unique()
    rows = []
    for EC_num in tqdm(unique_EC_TX):
        EC_df = df[df[EC].apply(lambda x: EC_num in x)] #OMG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG
        if len(EC_df) < 50:
            non_AS_samplerate = 1
        elif len(EC_df) < 100:
            non_AS_samplerate = 0.5
        elif len(EC_df) < 500:
            non_AS_samplerate = 0.1
        else:
            non_AS_samplerate = 0.05
        for BS_ID in unique_BS_IDs:
            all_BS_positions = []
            non_BS_positions = []
            for i, row in EC_df.iterrows():
                if row["Entry"] in train_entries:
                    binding_sites = row[BS_ID]
                    entry = row["Entry"]
                    for site in binding_sites:
                        pos = (site, entry, embeddings_dir)
                        all_BS_positions.append(pos)
                    # Take a random sample of 0.1 of the non active site embeddings and add them to non_AS_embeddings
                    non_AS_sites = random.sample(range(0, len(row["Sequence"])), int(len(row["Sequence"])*non_AS_samplerate))
                    for site in non_AS_sites:
                        if site not in binding_sites:
                            pos = (site, entry, embeddings_dir)
                            non_BS_positions.append(pos)
                            
        # subset the positions to a random set of 300 to control the explosion of possible combinations - improves speed and doesn't impact final number of pairs
            if len(all_BS_positions) > 200:
                all_BS_positions = random.sample(all_BS_positions, 200)
            if len(non_BS_positions) > 200:
                non_BS_positions = random.sample(non_BS_positions, 200)
            
            for pos in all_BS_positions:
                rows.append([EC_num, BS_ID, True, pos])
            for pos in non_BS_positions:
                rows.append([EC_num, BS_ID, False, pos])
        
    # create a df from the rows
    df = pd.DataFrame(rows, columns=['EC', 'BS_ID', 'BS', 'pos'])
    
    for EC in tqdm(unique_EC_TX):
        for BS_ID in unique_BS_IDs:
            # subset the df to df w BS_ID and BS are same
            same_EC = df[df['EC'] == EC]
            same_EC_BS_ID = same_EC[same_EC['BS_ID'] == BS_ID]
            same_EC_BS_ID_BS = same_EC_BS_ID[same_EC_BS_ID['BS'] == True]
            same_EC_BS_ID_NBS = same_EC_BS_ID[same_EC_BS_ID['BS'] == False]
            
            if len(same_EC_BS_ID_BS) < 1 or len(same_EC_BS_ID_NBS) < 1:
                continue
            
            all_BS_positions = [pos for pos in same_EC_BS_ID_BS['pos']]
            non_BS_positions = [pos for pos in same_EC_BS_ID_NBS['pos']]
                        
            # Take all possible unique pairs of the AS embeddings and save them in a list
            BSvBS_paired_positions = []
            NBSvNBS_paired_positions = []
            BSvNBS_paired_positions = []
            
            if len(non_BS_positions) != 0:
                #do the same for the non_BS_embeddings
                for i in range(len(non_BS_positions)):
                    for j in range(i+1, len(non_BS_positions)):
                        BSvBS_paired_positions.append((non_BS_positions[i], non_BS_positions[j]))
            # control the sample rate
            if len(BSvBS_paired_positions) > sample_limit:
                BSvBS_paired_positions = random.sample(BSvBS_paired_positions, sample_limit)
            BSvBS_labels = [1]*len(BSvBS_paired_positions)
            
            if len(all_BS_positions) != 0:
                for i in range(len(all_BS_positions)):
                    for j in range(i+1, len(all_BS_positions)):
                        NBSvNBS_paired_positions.append((all_BS_positions[i], all_BS_positions[j]))
            # control the sample rate
            if len(NBSvNBS_paired_positions) > sample_limit:
                NBSvNBS_paired_positions = random.sample(NBSvNBS_paired_positions, sample_limit)
            NBSvNBS_labels = [1]*len(NBSvNBS_paired_positions)
            
            if len(all_BS_positions) != 0 or len(non_BS_positions) != 0:
                # Now make random pairs between the AS and non AS embeddings
                for i in range(len(all_BS_positions)):
                    for j in range(len(non_BS_positions)):
                        BSvNBS_paired_positions.append((all_BS_positions[i], non_BS_positions[j]))
            # control the sample rate
            if len(BSvNBS_paired_positions) > sample_limit:
                BSvNBS_paired_positions = random.sample(BSvNBS_paired_positions, sample_limit)
            BSvNBS_labels = [-1]*len(BSvNBS_paired_positions)
            
            all_paired_embeddings.extend(BSvBS_paired_positions)
            all_labels.extend(BSvBS_labels)
            all_paired_embeddings.extend(NBSvNBS_paired_positions)
            all_labels.extend(NBSvNBS_labels)
            all_paired_embeddings.extend(BSvNBS_paired_positions)
            all_labels.extend(BSvNBS_labels)
            
            BS_EC_negative_pairs=[]
            NBS_EC_negative_pairs=[]
            
            # subset the df to df where EC not the same but BS_ID and BS yes
            same_BSID = df[df['BS_ID'] == BS_ID]
            # TODO: some argument could be made that contrasting negatively in the EC space on binding sites could be detrimental... need to consider further
            same_BSID = same_BSID[same_BSID['EC'] != EC]
            same_BSID_BS = same_BSID[same_BSID['BS'] == True]
            same_BSID_NBS = same_BSID[same_BSID['BS'] == False]
            
            # append a random sample of pos from same_BSID_BS to all_BS_positions - less than 200
            if len(same_BSID_BS) > 200:
                same_BSID_BS = same_BSID_BS.sample(n=200)
            same_BSID_BS_positions = [pos for pos in same_BSID_BS['pos']]
            
            if len(all_BS_positions) != 0 or len(same_BSID_BS_positions) != 0:
                for i in range(len(all_BS_positions)):
                    for j in range(len(same_BSID_BS_positions)):
                        BS_EC_negative_pairs.append((all_BS_positions[i], same_BSID_BS_positions[j]))
            # control the sample rate
            if len(BS_EC_negative_pairs) > sample_limit:
                BS_EC_negative_pairs = random.sample(BS_EC_negative_pairs, sample_limit)
            same_BSID_BS_labels = [-1]*len(BS_EC_negative_pairs)
            
            if len(same_BSID_NBS) > 200:
                same_BSID_NBS = same_BSID_NBS.sample(n=200)
            same_BSID_NBS_positions = [pos for pos in same_BSID_NBS['pos']]
            
            if len(all_BS_positions) != 0 or len(same_BSID_NBS_positions) != 0:
                for i in range(len(all_BS_positions)):
                    for j in range(len(same_BSID_NBS_positions)):
                        NBS_EC_negative_pairs.append((all_BS_positions[i], same_BSID_NBS_positions[j]))
            # control the sample rate
            if len(NBS_EC_negative_pairs) > sample_limit:
                NBS_EC_negative_pairs = random.sample(NBS_EC_negative_pairs, sample_limit)
            same_BSID_NBS_labels = [-1]*len(NBS_EC_negative_pairs)
            
            all_paired_embeddings.extend(BS_EC_negative_pairs)
            all_labels.extend(same_BSID_BS_labels)
            all_paired_embeddings.extend(NBS_EC_negative_pairs)
            all_labels.extend(same_BSID_NBS_labels)
            
            contrBS_EC_negative_pairs = []
            
            # Now need to contrast the BS IDs which are different but have the same EC number
            same_EC = df[df['EC'] == EC]
            same_EC = same_EC[same_EC['BS_ID'] != BS_ID]
            same_EC_BS = same_EC[same_EC['BS'] == True]
            
            if len(same_EC_BS) > 200:
                same_EC_BS = same_EC_BS.sample(n=200)
            same_EC_BS_positions = [pos for pos in same_EC_BS['pos']]
            
            if len(all_BS_positions) != 0 or len(same_EC_BS_positions) != 0:
                for i in range(len(all_BS_positions)):
                    for j in range(len(same_EC_BS_positions)):
                        contrBS_EC_negative_pairs.append((all_BS_positions[i], same_EC_BS_positions[j]))
            # control the sample rate
            if len(contrBS_EC_negative_pairs) > sample_limit:
                contrBS_EC_negative_pairs = random.sample(contrBS_EC_negative_pairs, sample_limit)
            same_EC_BS_labels = [-1]*len(contrBS_EC_negative_pairs)
            
            all_paired_embeddings.extend(contrBS_EC_negative_pairs)
            all_labels.extend(same_EC_BS_labels)
            
    return all_paired_embeddings, all_labels


# Binding site specific!
# Scheme 4 but without NBS v NBS pairs
def scheme_5(df, embeddings_dir, train_entries, non_AS_samplerate = 0.02, EC = "EC_T-filtered", sample_limit = 600):
     # get all unique EC numbers in EC_TX
    unique_EC_TX = df[EC].explode().unique()
    print("Unique EC numbers: ")
    print(unique_EC_TX)
    
    all_paired_embeddings = []
    all_labels = []
    
    # get the unique set of Cofactor IDs in the df
    unique_BS_IDs = df['Cofactor IDs'].explode().unique()
    rows = []
    for EC in tqdm(unique_EC_TX):
        EC_df = df[df['EC_TX'].apply(lambda x: EC in x)]
        if len(EC_df) < 50:
            non_AS_samplerate = 1
        elif len(EC_df) < 100:
            non_AS_samplerate = 0.5
        elif len(EC_df) < 500:
            non_AS_samplerate = 0.1
        else:
            non_AS_samplerate = 0.05
        for BS_ID in unique_BS_IDs:
            all_BS_positions = []
            non_BS_positions = []
            for i, row in EC_df.iterrows():
                if row["Entry"] in train_entries:
                    binding_sites = row[BS_ID]
                    entry = row["Entry"]
                    for site in binding_sites:
                        pos = (site, entry, embeddings_dir)
                        all_BS_positions.append(pos)
                    # Take a random sample of 0.1 of the non active site embeddings and add them to non_AS_embeddings
                    non_AS_sites = random.sample(range(0, len(row["Sequence"])), int(len(row["Sequence"])*non_AS_samplerate))
                    for site in non_AS_sites:
                        if site not in binding_sites:
                            pos = (site, entry, embeddings_dir)
                            non_BS_positions.append(pos)
                            
        # subset the positions to a random set of 300 to control the explosion of possible combinations - improves speed and doesn't impact final number of pairs
            if len(all_BS_positions) > 200:
                all_BS_positions = random.sample(all_BS_positions, 200)
            if len(non_BS_positions) > 200:
                non_BS_positions = random.sample(non_BS_positions, 200)
            
            for pos in all_BS_positions:
                rows.append([EC, BS_ID, True, pos])
            for pos in non_BS_positions:
                rows.append([EC, BS_ID, False, pos])
        
    # create a df from the rows
    df = pd.DataFrame(rows, columns=['EC', 'BS_ID', 'BS', 'pos'])
    
    for EC in tqdm(unique_EC_TX):
        for BS_ID in unique_BS_IDs:
            # subset the df to df where all EC BS_ID and BS are same
            same_EC = df[df['EC'] == EC]
            same_EC_BS_ID = same_EC[same_EC['BS_ID'] == BS_ID]
            same_EC_BS_ID_BS = same_EC_BS_ID[same_EC_BS_ID['BS'] == True]
            same_EC_BS_ID_NBS = same_EC_BS_ID[same_EC_BS_ID['BS'] == False]
            
            if len(same_EC_BS_ID_BS) < 1 or len(same_EC_BS_ID_NBS) < 1:
                continue
            
            all_BS_positions = [pos for pos in same_EC_BS_ID_BS['pos']]
            non_BS_positions = [pos for pos in same_EC_BS_ID_NBS['pos']]
                        
            # Take all possible unique pairs of the AS embeddings and save them in a list
            BSvBS_paired_positions = []
            #NBSvNBS_paired_positions = []   # Not needed for this scheme
            BSvNBS_paired_positions = []
            
            if len(non_BS_positions) != 0:
                #do the same for the non_BS_embeddings
                for i in range(len(non_BS_positions)):
                    for j in range(i+1, len(non_BS_positions)):
                        BSvBS_paired_positions.append((non_BS_positions[i], non_BS_positions[j]))
            # control the sample rate
            if len(BSvBS_paired_positions) > sample_limit:
                BSvBS_paired_positions = random.sample(BSvBS_paired_positions, sample_limit)
            BSvBS_labels = [1]*len(BSvBS_paired_positions)
            
            #if len(all_BS_positions) != 0:
            #    for i in range(len(all_BS_positions)):
            #        for j in range(i+1, len(all_BS_positions)):
            #            NBSvNBS_paired_positions.append((all_BS_positions[i], all_BS_positions[j]))
            # control the sample rate
            #if len(NBSvNBS_paired_positions) > sample_limit:
            #    NBSvNBS_paired_positions = random.sample(NBSvNBS_paired_positions, sample_limit)
            #NBSvNBS_labels = [1]*len(NBSvNBS_paired_positions)
            
            if len(all_BS_positions) != 0 or len(non_BS_positions) != 0:
                # Now make random pairs between the AS and non AS embeddings
                for i in range(len(all_BS_positions)):
                    for j in range(len(non_BS_positions)):
                        BSvNBS_paired_positions.append((all_BS_positions[i], non_BS_positions[j]))
            # control the sample rate
            if len(BSvNBS_paired_positions) > sample_limit:
                BSvNBS_paired_positions = random.sample(BSvNBS_paired_positions, sample_limit)
            BSvNBS_labels = [-1]*len(BSvNBS_paired_positions)
            
            all_paired_embeddings.extend(BSvBS_paired_positions)
            all_labels.extend(BSvBS_labels)
            #all_paired_embeddings.extend(NBSvNBS_paired_positions)
            #all_labels.extend(NBSvNBS_labels)
            all_paired_embeddings.extend(BSvNBS_paired_positions)
            all_labels.extend(BSvNBS_labels)
            
            BS_EC_negative_pairs=[]
            #NBS_EC_negative_pairs=[]  # Not needed for this scheme
            
            # subset the df to df where EC not the same but BS_ID and BS yes
            same_BSID = df[df['BS_ID'] == BS_ID]
            # TODO: some argument could be made that contrasting negatively in the EC space on binding sites could be detrimental... need to consider further
            same_BSID = same_BSID[same_BSID['EC'] != EC]
            same_BSID_BS = same_BSID[same_BSID['BS'] == True]
            same_BSID_NBS = same_BSID[same_BSID['BS'] == False]
            
            # append a random sample of pos from same_BSID_BS to all_BS_positions - less than 200
            if len(same_BSID_BS) > 200:
                same_BSID_BS = same_BSID_BS.sample(n=200)
            same_BSID_BS_positions = [pos for pos in same_BSID_BS['pos']]
            
            if len(all_BS_positions) != 0 or len(same_BSID_BS_positions) != 0:
                for i in range(len(all_BS_positions)):
                    for j in range(len(same_BSID_BS_positions)):
                        BS_EC_negative_pairs.append((all_BS_positions[i], same_BSID_BS_positions[j]))
            # control the sample rate
            if len(BS_EC_negative_pairs) > sample_limit:
                BS_EC_negative_pairs = random.sample(BS_EC_negative_pairs, sample_limit)
            same_BSID_BS_labels = [-1]*len(BS_EC_negative_pairs)
            
            if len(same_BSID_NBS) > 200:
                same_BSID_NBS = same_BSID_NBS.sample(n=200)
            same_BSID_NBS_positions = [pos for pos in same_BSID_NBS['pos']]
            
            #if len(all_BS_positions) != 0 or len(same_BSID_NBS_positions) != 0:
            #    for i in range(len(all_BS_positions)):
            #        for j in range(len(same_BSID_NBS_positions)):
            #            NBS_EC_negative_pairs.append((all_BS_positions[i], same_BSID_NBS_positions[j]))
            # control the sample rate
            #if len(NBS_EC_negative_pairs) > sample_limit:
            #    NBS_EC_negative_pairs = random.sample(NBS_EC_negative_pairs, sample_limit)
            #same_BSID_NBS_labels = [-1]*len(NBS_EC_negative_pairs)
            
            all_paired_embeddings.extend(BS_EC_negative_pairs)
            all_labels.extend(same_BSID_BS_labels)
            #all_paired_embeddings.extend(NBS_EC_negative_pairs)
            #all_labels.extend(same_BSID_NBS_labels)
            
            contrBS_EC_negative_pairs = []
            
            # Now need to contrast the BS IDs which are different but have the same EC number
            # same_EC = df[df['EC'] == EC]
            # same_EC = same_EC[same_EC['BS_ID'] != BS_ID]
            # same_EC_BS = same_EC[same_EC['BS'] == True]
            
            #if len(same_EC_BS) > 200:
            #    same_EC_BS = same_EC_BS.sample(n=200)
            #same_EC_BS_positions = [pos for pos in same_EC_BS['pos']]
            
            #if len(all_BS_positions) != 0 or len(same_EC_BS_positions) != 0:
            #    for i in range(len(all_BS_positions)):
            #        for j in range(len(same_EC_BS_positions)):
            #            contrBS_EC_negative_pairs.append((all_BS_positions[i], same_EC_BS_positions[j]))
            # control the sample rate
            #if len(contrBS_EC_negative_pairs) > sample_limit:
            #    contrBS_EC_negative_pairs = random.sample(contrBS_EC_negative_pairs, sample_limit)
            # same_EC_BS_labels = [-1]*len(contrBS_EC_negative_pairs)
            
            # all_paired_embeddings.extend(contrBS_EC_negative_pairs)
            # all_labels.extend(same_EC_BS_labels)
            
    return all_paired_embeddings, all_labels


# Binding site specific!
# Scheme 4 but without NBS v NBS pairs and no BS_EC_negative_pairs
def scheme_6(df, embeddings_dir, train_entries, non_AS_samplerate = 0.02, EC = "EC_T-filtered", sample_limit = 600):
     # get all unique EC numbers in EC_TX
    unique_EC_TX = df[EC].explode().unique()
    print("Unique EC numbers: ")
    print(unique_EC_TX)
    
    all_paired_embeddings = []
    all_labels = []
    
    # get the unique set of Cofactor IDs in the df
    unique_BS_IDs = df['Cofactor IDs'].explode().unique()
    rows = []
    for EC in tqdm(unique_EC_TX):
        EC_df = df[df['EC_TX'].apply(lambda x: EC in x)]
        if len(EC_df) < 50:
            non_AS_samplerate = 1
        elif len(EC_df) < 100:
            non_AS_samplerate = 0.5
        elif len(EC_df) < 500:
            non_AS_samplerate = 0.1
        else:
            non_AS_samplerate = 0.05
        for BS_ID in unique_BS_IDs:
            all_BS_positions = []
            non_BS_positions = []
            for i, row in EC_df.iterrows():
                if row["Entry"] in train_entries:
                    binding_sites = row[BS_ID]
                    entry = row["Entry"]
                    for site in binding_sites:
                        pos = (site, entry, embeddings_dir)
                        all_BS_positions.append(pos)
                    # Take a random sample of 0.1 of the non active site embeddings and add them to non_AS_embeddings
                    non_AS_sites = random.sample(range(0, len(row["Sequence"])), int(len(row["Sequence"])*non_AS_samplerate))
                    for site in non_AS_sites:
                        if site not in binding_sites:
                            pos = (site, entry, embeddings_dir)
                            non_BS_positions.append(pos)
                            
        # subset the positions to a random set of 300 to control the explosion of possible combinations - improves speed and doesn't impact final number of pairs
            if len(all_BS_positions) > 200:
                all_BS_positions = random.sample(all_BS_positions, 200)
            if len(non_BS_positions) > 200:
                non_BS_positions = random.sample(non_BS_positions, 200)
            
            for pos in all_BS_positions:
                rows.append([EC, BS_ID, True, pos])
            for pos in non_BS_positions:
                rows.append([EC, BS_ID, False, pos])
        
    # create a df from the rows
    df = pd.DataFrame(rows, columns=['EC', 'BS_ID', 'BS', 'pos'])
    
    for EC in tqdm(unique_EC_TX):
        for BS_ID in unique_BS_IDs:
            # subset the df to df where all EC BS_ID and BS are same
            same_EC = df[df['EC'] == EC]
            same_EC_BS_ID = same_EC[same_EC['BS_ID'] == BS_ID]
            same_EC_BS_ID_BS = same_EC_BS_ID[same_EC_BS_ID['BS'] == True]
            same_EC_BS_ID_NBS = same_EC_BS_ID[same_EC_BS_ID['BS'] == False]
            
            if len(same_EC_BS_ID_BS) < 1 or len(same_EC_BS_ID_NBS) < 1:
                continue
            
            all_BS_positions = [pos for pos in same_EC_BS_ID_BS['pos']]
            non_BS_positions = [pos for pos in same_EC_BS_ID_NBS['pos']]
                        
            # Take all possible unique pairs of the AS embeddings and save them in a list
            BSvBS_paired_positions = []
            #NBSvNBS_paired_positions = []   # Not needed for this scheme
            BSvNBS_paired_positions = []
            
            if len(non_BS_positions) != 0:
                #do the same for the non_BS_embeddings
                for i in range(len(non_BS_positions)):
                    for j in range(i+1, len(non_BS_positions)):
                        BSvBS_paired_positions.append((non_BS_positions[i], non_BS_positions[j]))
            # control the sample rate
            if len(BSvBS_paired_positions) > sample_limit:
                BSvBS_paired_positions = random.sample(BSvBS_paired_positions, sample_limit)
            BSvBS_labels = [1]*len(BSvBS_paired_positions)
            
            #if len(all_BS_positions) != 0:
            #    for i in range(len(all_BS_positions)):
            #        for j in range(i+1, len(all_BS_positions)):
            #            NBSvNBS_paired_positions.append((all_BS_positions[i], all_BS_positions[j]))
            # control the sample rate
            #if len(NBSvNBS_paired_positions) > sample_limit:
            #    NBSvNBS_paired_positions = random.sample(NBSvNBS_paired_positions, sample_limit)
            #NBSvNBS_labels = [1]*len(NBSvNBS_paired_positions)
            
            if len(all_BS_positions) != 0 or len(non_BS_positions) != 0:
                # Now make random pairs between the AS and non AS embeddings
                for i in range(len(all_BS_positions)):
                    for j in range(len(non_BS_positions)):
                        BSvNBS_paired_positions.append((all_BS_positions[i], non_BS_positions[j]))
            # control the sample rate
            if len(BSvNBS_paired_positions) > sample_limit:
                BSvNBS_paired_positions = random.sample(BSvNBS_paired_positions, sample_limit)
            BSvNBS_labels = [-1]*len(BSvNBS_paired_positions)
            
            all_paired_embeddings.extend(BSvBS_paired_positions)
            all_labels.extend(BSvBS_labels)
            #all_paired_embeddings.extend(NBSvNBS_paired_positions)
            #all_labels.extend(NBSvNBS_labels)
            all_paired_embeddings.extend(BSvNBS_paired_positions)
            all_labels.extend(BSvNBS_labels)
            
            BS_EC_negative_pairs=[]
            #NBS_EC_negative_pairs=[]  # Not needed for this scheme
            
            # subset the df to df where EC not the same but BS_ID and BS yes
            same_BSID = df[df['BS_ID'] == BS_ID]
            # TODO: some argument could be made that contrasting negatively in the EC space on binding sites could be detrimental... need to consider further
            same_BSID = same_BSID[same_BSID['EC'] != EC]
            same_BSID_BS = same_BSID[same_BSID['BS'] == True]
            same_BSID_NBS = same_BSID[same_BSID['BS'] == False]
            
            # append a random sample of pos from same_BSID_BS to all_BS_positions - less than 200
            #if len(same_BSID_BS) > 200:
            #    same_BSID_BS = same_BSID_BS.sample(n=200)
            #same_BSID_BS_positions = [pos for pos in same_BSID_BS['pos']]
            
            #if len(all_BS_positions) != 0 or len(same_BSID_BS_positions) != 0:
            #    for i in range(len(all_BS_positions)):
            #        for j in range(len(same_BSID_BS_positions)):
            #            BS_EC_negative_pairs.append((all_BS_positions[i], same_BSID_BS_positions[j]))
            # control the sample rate
            #if len(BS_EC_negative_pairs) > sample_limit:
            #    BS_EC_negative_pairs = random.sample(BS_EC_negative_pairs, sample_limit)
            #same_BSID_BS_labels = [-1]*len(BS_EC_negative_pairs)
            
            #if len(same_BSID_NBS) > 200:
            #    same_BSID_NBS = same_BSID_NBS.sample(n=200)
            #same_BSID_NBS_positions = [pos for pos in same_BSID_NBS['pos']]
            
            #if len(all_BS_positions) != 0 or len(same_BSID_NBS_positions) != 0:
            #    for i in range(len(all_BS_positions)):
            #        for j in range(len(same_BSID_NBS_positions)):
            #            NBS_EC_negative_pairs.append((all_BS_positions[i], same_BSID_NBS_positions[j]))
            # control the sample rate
            #if len(NBS_EC_negative_pairs) > sample_limit:
            #    NBS_EC_negative_pairs = random.sample(NBS_EC_negative_pairs, sample_limit)
            #same_BSID_NBS_labels = [-1]*len(NBS_EC_negative_pairs)
            
            #all_paired_embeddings.extend(BS_EC_negative_pairs)
            #all_labels.extend(same_BSID_BS_labels)
            #all_paired_embeddings.extend(NBS_EC_negative_pairs)
            #all_labels.extend(same_BSID_NBS_labels)
            
            #contrBS_EC_negative_pairs = []
            
            # Now need to contrast the BS IDs which are different but have the same EC number
            # same_EC = df[df['EC'] == EC]
            # same_EC = same_EC[same_EC['BS_ID'] != BS_ID]
            # same_EC_BS = same_EC[same_EC['BS'] == True]
            
            #if len(same_EC_BS) > 200:
            #    same_EC_BS = same_EC_BS.sample(n=200)
            #same_EC_BS_positions = [pos for pos in same_EC_BS['pos']]
            
            #if len(all_BS_positions) != 0 or len(same_EC_BS_positions) != 0:
            #    for i in range(len(all_BS_positions)):
            #        for j in range(len(same_EC_BS_positions)):
            #            contrBS_EC_negative_pairs.append((all_BS_positions[i], same_EC_BS_positions[j]))
            # control the sample rate
            #if len(contrBS_EC_negative_pairs) > sample_limit:
            #    contrBS_EC_negative_pairs = random.sample(contrBS_EC_negative_pairs, sample_limit)
            # same_EC_BS_labels = [-1]*len(contrBS_EC_negative_pairs)
            
            # all_paired_embeddings.extend(contrBS_EC_negative_pairs)
            # all_labels.extend(same_EC_BS_labels)
            
    return all_paired_embeddings, all_labels


def main():
    args = arg_parser()
    # Make a base directory
    os.makedirs(args.out, exist_ok=True)
        
    # Load the metadata
    df = pd.read_csv(args.metadata, sep='\t')
    
    print("Finding Nans in the metadata: ")
    print(len(df))
    # check if any rows in the csv have a NAN in the EC or Active site columns
    df = df.dropna(subset=["Active site", "EC number"])
    print(len(df))
        
    # remove any rows with missing values in the EC_TX column
    
    # get all entries in the embedding directory
    filenames = glob.glob(args.embedding_dir + '/*.pt')
    train_entries = [x.split('|')[1] for x in filenames]
        
    # load the eval set .txt file
    with open(args.eval, 'r') as f:
        eval_entries = f.readlines()
    eval_entries = [x.strip() for x in eval_entries]
    
    with open(args.test, 'r') as f:
        test_entries = f.readlines()
    test_entries = [x.strip() for x in test_entries]
    train_entries = [x for x in train_entries if x not in eval_entries]
    train_entries = [x for x in train_entries if x not in test_entries]
    
    if args.leaked_entries:
        with open(args.leaked_entries, 'r') as f:
            leaked_entries = f.readlines()
        leaked_entries = [x.strip() for x in leaked_entries]
        # append the leaked entries to the train entries
        train_entries.extend(leaked_entries)
    
    if args.BSvAS == 'BS':
        pos, ids = get_BS_pos_from_uniprot_processed_tsv(df)
        df[args.BSvAS] = pos
        df['Cofactor IDs'] = ids
        df['Cofactor IDs'] = df['Cofactor IDs'].apply(lambda x: x if isinstance(x, list) else [])
    elif args.BSvAS == 'AS':
        df[args.BSvAS] = get_AS_pos_from_uniprot(df)
    
    if args.BSvAS == 'BS':
        df = BS_site_ID_column_splitting(df)
    df["EC_TX"] = get_EC_TX_from_uniprot(df)
    df.to_csv(args.out + '/metadata_paired.tsv', sep='\t', index=False)
    
    # subset the df to the emb_entries in training set
    df = df[df["Entry"].isin(train_entries)]
        
    # reset the index
    df = df.reset_index(drop=True)
        
    print("Collecting contrastive pair scheme: ")
    
    # add a switch statement for the args.scheme used
    if args.scheme == 1:
        all_paired_positions, all_labels, site_embeddings_dict = scheme_1(df, args.embedding_dir, train_entries, sample_limit=2000000, ASvBS=args.BSvAS, layer=args.layer)
        print("Number of pairs | Number of labels | Number of embeddings in embedding pool")
        print(len(all_paired_positions), len(all_labels), len(site_embeddings_dict))
        dataset = PointerPairedDataset(site_embeddings_dict, all_paired_positions, all_labels)
        torch.save(dataset, args.out + '/paired_embeddings_dataset.pt')
    elif args.scheme == 2:
        all_paired_positions, all_labels, site_embeddings_dict = scheme_2(df, args.embedding_dir, train_entries, sample_limit=args.sample_limit, ASvBS=args.BSvAS, layer=args.layer)
        print("Number of pairs | Number of labels | Number of embeddings in embedding pool")
        print(len(all_paired_positions), len(all_labels), len(site_embeddings_dict))
        dataset = PointerPairedDataset(site_embeddings_dict, all_paired_positions, all_labels)
        torch.save(dataset, args.out + '/paired_embeddings_dataset.pt')
    elif args.scheme == 3:
        all_paired_positions, all_labels, site_embeddings_dict = scheme_3(df, args.embedding_dir, train_entries, sample_limit=args.sample_limit, ASvBS=args.BSvAS, layer=args.layer)
        print("Number of pairs | Number of labels | Number of embeddings in embedding pool")
        print(len(all_paired_positions), len(all_labels), len(site_embeddings_dict))
        dataset = PointerPairedDataset(site_embeddings_dict, all_paired_positions, all_labels)
        torch.save(dataset, args.out + '/paired_embeddings_dataset.pt')
        
        
    elif args.scheme == 4:
        unique_EC_TX = df['EC_T-filtered'].explode().unique()
        unique_BS_IDs = df['Cofactor IDs'].explode().unique()
        num_pair_arguments = 6
        num_clusters = len(unique_BS_IDs)*len(unique_EC_TX)
        max_pairs = num_pair_arguments*args.sample_limit*num_clusters
        print("Unique EC numbers: ", len(unique_EC_TX))
        print("Unique BS IDs: ", len(unique_BS_IDs))
        print("Max # pairs per cluster: ", num_pair_arguments*args.sample_limit)
        print("Number of clusters: ", num_clusters)
        print("Therefore, the Approx. # for Max total pairs: ", max_pairs)
        all_paired_positions, all_labels = scheme_4(df, args.embedding_dir, train_entries, sample_limit=args.sample_limit)
        print(len(all_paired_positions), len(all_labels))
    elif args.scheme == 5:
        unique_EC_TX = df['EC_T-filtered'].explode().unique()
        unique_BS_IDs = df['Cofactor IDs'].explode().unique()
        num_pair_arguments = 3
        num_clusters = len(unique_BS_IDs)*len(unique_EC_TX)
        max_pairs = num_pair_arguments*args.sample_limit*num_clusters
        print("Unique EC numbers: ", len(unique_EC_TX))
        print("Unique BS IDs: ", len(unique_BS_IDs))
        print("Max # pairs per cluster: ", num_pair_arguments*args.sample_limit)
        print("Number of clusters: ", num_clusters)
        print("Therefore, the Approx. # for Max total pairs: ", max_pairs)
        all_paired_positions, all_labels = scheme_5(df, args.embedding_dir, train_entries, sample_limit=args.sample_limit)
        print(len(all_paired_positions), len(all_labels))
    elif args.scheme == 6:
        unique_EC_TX = df['EC_T-filtered'].explode().unique()
        unique_BS_IDs = df['Cofactor IDs'].explode().unique()
        num_pair_arguments = 3
        num_clusters = len(unique_BS_IDs)*len(unique_EC_TX)
        max_pairs = num_pair_arguments*args.sample_limit*num_clusters
        print("Unique EC numbers: ", len(unique_EC_TX))
        print("Unique BS IDs: ", len(unique_BS_IDs))
        print("Max # pairs per cluster: ", num_pair_arguments*args.sample_limit)
        print("Number of clusters: ", num_clusters)
        print("Therefore, the Approx. # for Max total pairs: ", max_pairs)
        all_paired_positions, all_labels = scheme_6(df, args.embedding_dir, train_entries, sample_limit=args.sample_limit)
        print(len(all_paired_positions), len(all_labels))

    

if __name__ == "__main__":
    main()
