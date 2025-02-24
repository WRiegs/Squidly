
import warnings
import pandas as pd
import numpy as np
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

from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import davies_bouldin_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import BatchSampler
import torch.nn.functional as F

from tqdm import tqdm

# Suppress the PerformanceWarning from pandas
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# set the seed
seed = 420
random.seed(seed)


chebi_numbers = [
   17579, 23357, 15636, 1989, 57453, 15635, 83088, 30314, 16509, 80214, 
   28889, 17679, 60494, 17601, 58201, 52020, 40260, 60470, 17401, 58130, 
   16680, 15414, 59789, 38290, 29073, 16027, 456215, 28938, 15422, 57299, 
   30616, 61338, 37136, 15956, 57586, 30402, 48775, 29108, 48782, 17996, 
   18230, 58416, 27888, 61721, 15982, 60488, 16304, 30411, 48828, 49415, 
   18408, 28265, 60540, 23378, 49552, 29036, 33221, 33913, 72953, 17803, 
   35169, 17694, 42121, 60342, 73113, 73095, 73115, 73096, 4746, 16238, 
   57692, 60454, 68438, 60519, 36183, 60532, 17627, 60344, 61717, 60562, 
   24040, 24041, 17621, 58210, 16048, 57618, 57925, 16856, 62811, 62814, 
   24480, 16240, 17544, 43474, 17594, 24875, 29033, 29034, 60504, 30409, 
   60357, 49701, 49807, 18420, 29035, 44245, 28115, 25372, 16768, 57540, 
   15846, 16908, 57945, 58349, 18009, 16474, 57783, 137373, 137399, 25516, 
   49786, 17154, 47739, 177874, 16858, 47942, 25848, 18067, 26116, 36079, 
   26214, 29103, 150862, 87749, 87746, 87531, 17310, 18405, 597326, 131529, 
   16709, 26461, 18315, 58442, 15361, 32816, 17015, 57986, 59560, 28599, 
   60052, 18023, 58351, 29101, 35104, 16189, 49883, 71177, 9532, 58937, 
   35172, 36970, 27547, 29105, 23354, 26348, 26672
]

ATP_related = [
    "456215", # AMP conj base
    "30616", # ATP (4-)
    "16027", #  adenosine 5'-monophosphate
    "15422", # ATP
    "57299" # ATP (3-)
]

NAD_related = [
    "13389",  # NAD
    "13390",  # NAD(P)
    "13392",  # NAD(P)H
    "57945",  # NADH(2-)
    "16908", # NADH
    "58349",  # NADP(3-)
    "57783",  # NADPH
    "143948", # 5'-end NADH(2−)
    "18009",   # NADP
    "57540", # NAD(1-)
    "17154" # Nicotinamide - pretty small, might want to remove
]

FAD_related = [
    "57692",  # FAD (3-)
    "16238",  # FAD
    "17877", # FADH2
    "58307",   # FADH2 (2-)
    "60470" # 6-hydroxy-FAD(3−)
    
]

HEM_related = [
    "30413",  # Heme
    "35241",  # Heme c
    "24479",  # Heme a
    "26355", # heme b
    "36144", # ferriheme b
    "26621",
    "60562", # heme c
    "60344", # ferroheme b(2-)
    "17627", # ferroheme b
    "24480", # Heme o
    "62811", # heme d cis-diol(2-)
    "62814", # heme d cis-diol
    "61717" # ferroheme c(2−)
]

COA_related = [
    "15346",  # Coenzyme A
    "57287",  # coenzyme A(4−)
    "57288",   # acetyl-CoA(4−)
    "15351", # Acetyl-CoA
    "57392" # propionyl-CoA(4−)
]

FMN_related = [
    "58210",  # FMN(3−)
    "17621",   # FMN
    "87746", # prenyl-FMN(2−)
    "87749", # prenyl-FMN
    "16048", # FMNH2
    "57618", # FMNH2(2−)
    "50528", # FMNH
    "140311", # FMNH(2-)
    "133886", # FMNH2(3-)
    "87531", # prenyl-FMNH2
    "87467" # prenyl-FMNH2(2-)
]

Looks_like_phosphate = [
    "35169", # dihydrogenvanadate
    "17544", # hydrogen carbonate -- sketchy... but I mean...
    "43474", # hydrogen phosphate
]

Coenzyme_F_related = [
    "28265", # Coenzyme F430
    "60540", # coenzyme F430(5−)
]

cobalamin_related = [
    "28115", # methylcobalamin
    "16304", # cob(II)alamin
    "18408", # cobamamide
]

one_anion = [
    "17996", # Chloride ion (Cl⁻)
]

one_Cation = [
    "29101",  # Sodium ion (Na⁺)
    "29103",   # Potassium ion (K⁺)
    "30150", # al+1
    "39099", # Calcium (1+)
    "49713", # lithium (1+)
    "49847", # Rubidium (1+)
    "49547", # Cesium (1+)
    "49552" # Copper (1+)
]

two_Cation = [
    "18420",  # Magnesium ion (Mg²⁺)
    "29108",  # Calcium ion (Ca²⁺)
    "29105",  # Zinc ion (Zn²⁺)
    "29033",  # Ferrous ion (Fe²⁺)
    "29036",   # Copper ion (Cu²⁺)
    "29035",   # Manganese ion (Mn²⁺)
    "24875", # iron cation
    "23378", # copper cation
    "48828", # cobalt(2+)
    "48775", # Cadmium (2+)
    "49786", # Nickel (2+)
]

three_Cation = [
    "29034",  # Ferric ion (Fe³⁺)
    "29039",   # Aluminum ion (Al³⁺)
    "49544",  # Chromium (3+)
    "49415",   # Cobalt (3+)
    "49701" # Lanthanum (3+)
]

Iron_sulfur_clusters = [
    "30409", # iron-sulfur-molybdenum cofactor
    "30408",  # Iron-sulfur cluster   
    "60357", # iron-sulfur-vanadium cofactor
    "60504", # iron-sulfur-iron cofactor
    "48796", # iron-sulfur-molybdenum cluster 
    "49883",
    "177874", # NiFe4S5 cluster
    "47739", # NiFe4S4 cluster
    "60519", # Fe4S2O2 iron-sulfur-oxygen cluster
]

ORPHANS = ['60494', '60342', '73095', '137373', '58201', '60540', '17310', '59560', '28115', '60357']

# make a dictionary for chebi_groupings
chebi_groupings = {}
chebi_groupings["ATP_related"] = ATP_related
chebi_groupings["NAD_related"] = NAD_related
chebi_groupings["FAD_related"] = FAD_related
chebi_groupings["HEM_related"] = HEM_related
chebi_groupings["COA_related"] = COA_related
chebi_groupings["FMN_related"] = FMN_related
#chebi_groupings["PLP_related"] = PLP_related
chebi_groupings["Looks_like_phosphate"] = Looks_like_phosphate
chebi_groupings["cobalamin_related"] = cobalamin_related
chebi_groupings["Coenzyme_F_related"] = Coenzyme_F_related
chebi_groupings["one_anion"] = one_anion
chebi_groupings["one_Cation"] = one_Cation
chebi_groupings["two_Cation"] = two_Cation
chebi_groupings["three_Cation"] = three_Cation
chebi_groupings["Iron_sulfur_clusters"] = Iron_sulfur_clusters
chebi_groupings["Orphans"] = ORPHANS


# Define the dataset class
class PairedEmbeddingsDataset(Dataset):
    def __init__(self, tensor, labels):
        assert tensor.shape[0] == len(labels), "The number of labels must match the number of rows in the tensors"
        
        self.tensor = tensor
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.tensor[idx], dtype=torch.float32), self.labels[idx]


def get_cofactor_sites(df):
    # remove any binding that has /ligand_id=ChEBI:XXXXX/ where XXXXX not in chebi_numbers
    cofactor_sites_per_seq = []
    cofactor_IDs_per_seq = []
    for i in range(len(df)):
        cofactor_sites = []
        cofactor_IDs = []
        binding = df['Binding site'][i]
        # if nan, skip
        if pd.isna(binding):
            cofactor_sites_per_seq.append(None)
            cofactor_IDs_per_seq.append(None)
            continue
        binding = binding.replace('\"', '')
        binding = binding.replace(' ', '')
        binding = binding.replace('\'', '')
        binding = binding.split(";")
        for j in binding:
            if j.startswith("BINDING"):
                sites = (j.split('BINDING')[1]).strip()
                sites = sites.split('..')
                if len(sites) == 2:
                    try:
                        sites = np.arange(int(sites[0])-1,int(sites[1]))
                        found = True
                    except:
                        found = False
                else:
                    try:
                        sites = [int(sites[0])-1]
                        found = True
                    except:
                        found = False
            if found is False:
                continue
            if j.startswith("/ligand_id=ChEBI:CHEBI:") or j.startswith("/ligand_id=ChEBI:"): 
                j = j.replace("ChEBI:", "")
                j = j.replace("CHEBI:", "")
                chebi_id = int(j.split("=")[1])
                if chebi_id not in chebi_numbers:
                    found = False
                    continue
                cofactor_sites.extend(sites)
                cofactor_IDs.extend([chebi_id]*len(sites))
                found = False
            # look for substrate binding sites
            if j.startswith("/ligand=substrate") or j.startswith("/ligand=Substrate"):
                cofactor_sites.extend(sites)
                cofactor_IDs.extend([00000]*len(sites))
                found = False
        if len(cofactor_sites) == 0:
            cofactor_sites_per_seq.append(None)
            cofactor_IDs_per_seq.append(None)
        else:
            cofactor_sites_per_seq.append(cofactor_sites)
            cofactor_IDs_per_seq.append(cofactor_IDs)
    
    assert len(cofactor_sites_per_seq) == len(df), "Length of cofactor sites and master_df do not match"
    assert len(cofactor_IDs_per_seq) == len(df), "Length of cofactor IDs and master_df do not match"
    
    df['Cofactor sites'] = cofactor_sites_per_seq
    df['Cofactor IDs'] = cofactor_IDs_per_seq
    
    return df


def get_EC_tiers(df):
    # for both df, get the EC number for each tier, [0:4] and make a column for each
    ECs = [str(x) for x in df['EC number']]
    ECs = [x.replace(' ','') for x in ECs]
    ECs = [x.split(';') for x in ECs]
    ECs = [[x.split('.') for x in y] for y in ECs]
    ECs_tier1 = [['.'.join(x[0:1]) for x in y] for y in ECs]
    ECs_tier2 = [['.'.join(x[0:2]) for x in y] for y in ECs]
    ECs_tier3 = [['.'.join(x[0:3]) for x in y] for y in ECs]
    ECs_tier4 = [['.'.join(x[0:4]) for x in y] for y in ECs]
    df['EC_tier1'] = ECs_tier1
    df['EC_tier2'] = ECs_tier2
    df['EC_tier3'] = ECs_tier3
    df['EC_tier4'] = ECs_tier4
    return df


def get_embedding(pos, entry, dir, layer=48):
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
        if len(binding_site_string) != len(set(binding_site_string)):
            count +=1
        binding_sites.append(binding_site_string)
    print("Number of sequences that have duplicate bs positions: ", count)
    
    binding_sites_ids = []
    for index, row in df.iterrows():
        binding_id_string = row["Cofactor IDs"]
        binding_sites_ids.append(binding_id_string)
        
        
    # figure out which positions in the binding site string have duplicates and then get the corresponding ID from the ID string
    # for i in range(len(binding_sites)):
    #     binding_site = binding_sites[i]
    #     binding_site_ids_IDC = binding_sites_ids[i]
    #     duplicates = [item for item, count in collections.Counter(binding_site).items() if count > 1]
    #     for duplicate in duplicates:
    #         idx = [i for i, x in enumerate(binding_site) if x == duplicate]
    #         for y in idx:
    #             # print the duplicate position and the corresponding ID
    #             #print(f"{i}: Duplicate position: {duplicate} with ID: {binding_site_ids_IDC[y]}")
                
    dict_IDs = {}
    # find the IDs that seem to overlap the most with others
    overlaps = []
    for i in range(len(binding_sites)):
        binding_site = binding_sites[i]
        binding_site_ids_IDC = binding_sites_ids[i]
        duplicates = [item for item, count in collections.Counter(binding_site).items() if count > 1]

        for duplicate in duplicates:
            # get all the unique IDs in the duplicate positions
            idx = [i for i, x in enumerate(binding_site) if x == duplicate]
            # now add 1 to the counter in dict_IDs for each ID in the idx list
            if len (idx) == 2:
                # if any of idx is 0, continue
                if 0 in idx:
                    continue
            else:
                # save the positions that have duplicates
                overlaps.append(duplicate)
                for y in idx:
                    if binding_site_ids_IDC[y] in dict_IDs:
                        dict_IDs[binding_site_ids_IDC[y]] += 1
                    else:
                        dict_IDs[binding_site_ids_IDC[y]] = 1
    
    # pairs_dict = {}
    # TODO: go through the overlaps, and find the most common pairings of IDs
                        
    # print the dict_IDs
    print("Dict of IDs: ", dict_IDs)
    
    # make a histogram of the dict_IDs
    plt.figure(figsize=(8,10))
    plt.bar(dict_IDs.keys(), dict_IDs.values())
    # set the x-axis to be rotated 45 degrees
    plt.xticks(rotation=60)
    # add a title
    plt.title('Distribution of Binding Site ID Overlaps')
    # add x and y labels
    plt.xlabel('Binding Site ID')
    plt.ylabel('Frequency of Overlap')
    
    # save the plot
    plt.savefig('Binding_site_ID_overlaps.png')
    
    df['BS'] = binding_sites
    df['Cofactor IDs'] = binding_sites_ids
        
    return df


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
                if '..' in item.split("BINDING ")[1]:
                    # get the range between the two numbers
                    sites = (item.split('BINDING ')[1]).strip()
                    sites = sites.split('..')
                    if len(sites) == 2:
                        try:
                            sites = np.arange(int(sites[0])-1,int(sites[1])) 
                            binding_site_list.extend(sites)
                            found = True
                        except:
                            found = False
                else:
                    binding_site_list.append(int(item.split("BINDING ")[1])-1)
        binding_sites.append(binding_site_list)
    return binding_sites


# Important cols : Cofactor sites	Cofactor IDs	EC_tier1	EC_tier2	EC_tier3	EC_tier4
def Flexible_EC_labeling(df, EC_threshold = 50, out_dir = ''):
    # for each EC_tier col, fix the fact that they are stored as a string, not a list of strings:
    for i, row in df.iterrows():
        for col in ['EC_tier1', 'EC_tier2', 'EC_tier3', 'EC_tier4']:
            ECs = row[col]
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


# sample without AA specificity
def sample_residue_embeddings(df, embeddings_dir, EC = "EC_T-filtered", sample_limit=200):
     # get all unique EC numbers in EC_TX
    unique_EC_TX = df[EC].explode().unique()
    print("Unique EC numbers: ")
    print(unique_EC_TX)
    
    all_paired_embeddings = []
    all_labels = []
    
    # get the unique set of Cofactor IDs in the df
    unique_BS_IDs = df['Cofactor IDs'].explode().unique()
    print("Unique BS IDs: ")
    print(unique_BS_IDs)
    rows = []
    for EC_num in tqdm(unique_EC_TX):
        EC_df = df[df[EC].apply(lambda x: EC_num == x)] #OMG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG BUG
        if len(EC_df) < 50:
            non_AS_samplerate = 1
        elif len(EC_df) < 100:
            non_AS_samplerate = 0.5
        elif len(EC_df) < 500:
            non_AS_samplerate = 0.1
        else:
            non_AS_samplerate = 0.05
            
        # GETTING EC and CHEBI CONSTRAINED BS POSITIONS
        for BS_ID in unique_BS_IDs:
            for AA in "ACDEFGHIKLMNPQRSTVWY":
                all_BS_positions = []
                for i, row in EC_df.iterrows():
                    binding_sites = row[BS_ID]
                    entry = row["Entry"]
                    for site in binding_sites:
                        if row["Sequence"][site] == AA:
                            pos = (site, entry, embeddings_dir)
                            all_BS_positions.append(pos)
                # subset the positions to a random set of X positions
                if len(all_BS_positions) > 5*sample_limit:
                    all_BS_positions = random.sample(all_BS_positions, 5*sample_limit)
                for pos in all_BS_positions:
                    rows.append([EC_num, BS_ID, AA, True, pos])   
            
        # GETTING EC CONSTRAINED NON-BS POSITIONS
        non_BS_positions = []
        for i, row in EC_df.iterrows():
            for AA in "ACDEFGHIKLMNPQRSTVWY":
                collated_all_BS_positions = row["collated_all_BS"]
                # Take a random sample of X_sample_rate of the non active site embeddings and add them to non_AS_embeddings
                non_AS_sites = random.sample(range(0, len(row["Sequence"])), int(len(row["Sequence"])*non_AS_samplerate))
                entry = row["Entry"]
                for site in non_AS_sites:
                    if site not in collated_all_BS_positions:
                        if row["Sequence"][site] == AA:
                            pos = (site, entry, embeddings_dir)
                            non_BS_positions.append(pos)
            if len(non_BS_positions) > int(sample_limit/3):
                non_BS_positions = random.sample(non_BS_positions, int(sample_limit/3))
            for pos in non_BS_positions:
                rows.append([EC_num, "non-BS", AA, False, pos])

    df = pd.DataFrame(rows, columns=['EC', 'BS_ID', 'AA', 'BS', 'pos'])
    
    # split into BS True and BS False
    df_BS = df[df['BS'] == True]
    df_non_BS = df[df['BS'] == False]
    print("Number of BS samples: ", len(df_BS))
    print("Number of non-BS samples: ", len(df_non_BS))
    print(df_BS.head())
        
    return df_BS, df_non_BS
    
        
# TODO Sample with AA specificity amino_acids = set("ACDEFGHIKLMNPQRSTVWY")


# hierarchy clustering
def h_cluster(embeddings, labels):

    # Calculate centroids for each unique label
    centroids = {}
    for i, label in enumerate(labels):
        if label in centroids:
            centroids[label].append(embeddings[i])
        else:
            centroids[label] = [embeddings[i]]

    # Convert list of embeddings to centroid by taking the mean
    for label in centroids:
        centroids[label] = np.mean(centroids[label], axis=0)
    
    # Calculate variance for each label's points relative to the centroid
    variances = {}
    for i, label in enumerate(labels):
        if label in variances:
            variances[label].append(np.linalg.norm(embeddings[i] - centroids[label]))
        else:
            variances[label] = [np.linalg.norm(embeddings[i] - centroids[label])]
    
    # Prepare the centroids list for hierarchical clustering
    centroid_embeddings = np.array(list(centroids.values()))
    labels_centroid = np.array(list(centroids.keys()))
    
    # Compute the pairwise distances between centroids
    distance_matrix = pdist(centroid_embeddings, metric='euclidean')  # 1D condensed distance matrix
    
    # Perform hierarchical clustering on centroids
    Z = linkage(distance_matrix, 'ward')

    # Calculate the Davies-Bouldin Index using the embeddings and original labels
    DB = davies_bouldin_score(embeddings, labels)
    print("Davies-Bouldin Index: ", DB)

    # Get the top 10 closest pairs of clusters, not including the diagonal
    dist_matrix_square = squareform(distance_matrix)  # Convert 1D distance matrix to 2D square form
    for i in range(len(dist_matrix_square)):
        dist_matrix_square[i][i] = np.inf
    top_10 = np.argsort(dist_matrix_square, axis=None)[:10]
    for i in top_10:
        print("Distance: ", dist_matrix_square[i//len(dist_matrix_square), i%len(dist_matrix_square)])
        print("Labels: ", labels_centroid[i//len(dist_matrix_square)], labels_centroid[i%len(dist_matrix_square)])

    # Plot the dendrogram
    plt.figure(figsize=(60, 20))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster Index')
    plt.ylabel('Distance')
    dendrogram(Z, labels=labels_centroid)
    
    # Save the plot
    plt.savefig('dendrogram.png')

    return DB, dist_matrix_square, centroids, variances


def display_worst_cluster_pairs(distance_matrix, centroids, labels):
    # get the pairs of clusters which have the smallest distances between them and their nearest neighbour
    # print the top 10 and their labels and their distances
    print()
    top_10 = np.argsort(distance_matrix, axis=None)[:10]
    for i in top_10:
        print("Distance: ", distance_matrix[i//len(distance_matrix), i%len(distance_matrix)])
        print("Labels: ", list(centroids.keys())[i//len(distance_matrix)], list(centroids.keys())[i%len(distance_matrix)])
    print()
    return


def merging_protocol(pca_embeddings, labels, previous_DBs = list(), label_tracking = list()):
    # append the labels to the label tracking list
    label_tracking.append(labels)
    
    
    # create 2 sets of labels based on EC or BS_ID
    EC_labels = [label.split('_')[0] for label in labels]
    BS_ID_labels = [label.split('_')[1] for label in labels]
    
    # Now calculate the DB score for the EC_labels, BS_ID_labels, and the labels
    DB_EC, distance_matrix_EC, centroids_EC, variances_EC = h_cluster(pca_embeddings, EC_labels)
    DB_BS_ID, distance_matrix_BS_ID, centroids_BS_ID, variances_BS_ID = h_cluster(pca_embeddings, BS_ID_labels)
    
    # print the DB scores
    print("DB score for EC labels: ", DB_EC)
    print("DB score for BS_ID labels: ", DB_BS_ID)
    
    print("Worst cluster pairs for EC labels: ")
    display_worst_cluster_pairs(distance_matrix_EC, centroids_EC, EC_labels)
    
    print("Worst cluster pairs for BS_ID labels: ")
    display_worst_cluster_pairs(distance_matrix_BS_ID, centroids_BS_ID, BS_ID_labels)
    
    ############################################################
    # Now start the merging protocol using the original labels #
    ############################################################
    
    DB, distance_matrix, centroids, variances = h_cluster(pca_embeddings, labels)
    
    previous_DBs.append(DB)
    
    # Find the top 10 worst cluster pairs
    print("Worst cluster pairs for original labels: ")
    display_worst_cluster_pairs(distance_matrix, centroids, labels)
    
    # Now take the worst cluster, and merge either by EC or BS_ID
    # choosing EC or BS_ID will depend on the distance between the EC and BS_ID labels for these two clusters in their original EC and BS_ID clusters
    # if the distance between the EC labels is smaller than the distance between the BS_ID labels, merge by EC, and vice versa
    # if the distances are the same, merge by EC
    
    # get the worst cluster pair - i.e. pair with shortest distance
    worst_cluster_pair = np.unravel_index(np.argmax(distance_matrix, axis=None), distance_matrix.shape)
    print("Worst cluster pair: ", worst_cluster_pair)
    
    # get the labels for the worst cluster pair
    cluster1 = list(centroids.keys())[worst_cluster_pair[0]]
    cluster2 = list(centroids.keys())[worst_cluster_pair[1]]
    
    # get the EC numbers from the labels
    cluster1_EC = cluster1.split('_')[0]
    cluster2_EC = cluster2.split('_')[0]
    
    # get the BS_IDs from the labels
    cluster1_BS_ID = cluster1.split('_')[1]
    cluster2_BS_ID = cluster2.split('_')[1]
    
    # get the distance between the EC labels
    EC_distance = distance_matrix_EC[np.where(np.array(EC_labels) == cluster1_EC)[0][0], np.where(np.array(EC_labels) == cluster2_EC)[0][0]]
    
    # get the distance between the BS_ID labels
    BS_ID_distance = distance_matrix_BS_ID[np.where(np.array(BS_ID_labels) == cluster1_BS_ID)[0][0], np.where(np.array(BS_ID_labels) == cluster2_BS_ID)[0][0]]
    
    # now merge the clusters based on the distances
    if EC_distance < BS_ID_distance:
        # merge by EC
        new_label = '.'.join(cluster1_EC, cluster2_EC)
        previous_ECs = [cluster1_EC, cluster2_EC]
        EC = True
        BS = False
    elif BS_ID_distance < EC_distance:
        # merge by BS_ID
        new_label = '.'.join(cluster1_BS_ID, cluster2_BS_ID)
        previous_BSs = [cluster1_BS_ID, cluster2_BS_ID]
        EC = False
        BS = True
    else:
        # merge by EC
        new_label = '.'.join(cluster1_EC, cluster2_EC)
        previous_ECs = [cluster1_EC, cluster2_EC]
        EC = True
        BS = False
    
    # update the labels with the new labels
    for i, label in enumerate(labels):
        if EC == True:
            if label.split('_')[0] in previous_ECs:
                labels[i] = new_label + '_' + label.split('_')[1]
        elif BS == True:
            if label.split('_')[1] in previous_BSs:
                labels[i] = label.split('_')[0] + '_' + new_label
                
    if len(previous_DBs) > 1:
        if DB < previous_DBs[-1]:
            # if the DB score has improved, continue the merging protocol
            merging_protocol(pca_embeddings, labels, previous_DBs) 
    elif len(previous_DBs) == 1:
        merging_protocol(pca_embeddings, labels, previous_DBs)
    else:
        return labels, previous_DBs
    
    
def process_pair_data(positions, labels, directory, batch_size=150, cleanup=True):
    # make the directory if not exist
    os.makedirs(directory, exist_ok=True)
    # make a sub directory for batch pkls and one for batch torch sets
    os.makedirs(directory + '/pkl', exist_ok=True)
    os.makedirs(directory + '/torch', exist_ok=True)
    
    positions_batched = [positions[i:i + batch_size] for i in range(0, len(positions), batch_size)]
    batch_labels = [labels[i:i + batch_size] for i in range(0, len(labels), batch_size)]
    
    # for each pair_batch, create a pickle file
    batch_id = 0
    for batch in positions_batched:
        with open(directory + f'/pkl/positions_batched{batch_id}.pkl', 'wb') as f:
            pickle.dump((batch, batch_labels[batch_id]), f)
        batch_id +=1
        
    # for each pickle file, call the slurm process script
    batch_id = 0
    pkl_files = glob.glob(directory + '/pkl/*.pkl')
    for pkl in range(len(pkl_files)):
        os.system(f'bash /scratch/project/squid/code/slurm_TRAST/process_batch_pair_scheme_for_sample_testing.sh Batch{batch_id} 00:30:00 1 {directory}/pkl/positions_batched{batch_id}.pkl {directory}/torch/batch{batch_id}.pt')
        batch_id +=1
    
    total_batches = batch_id
    
    # while the number of files in the torch dir are not equal to total_batches
    while len(glob.glob(directory + '/torch/*.pt')) != total_batches:
        print (f'Waiting for {total_batches - len(glob.glob(directory + "/torch/*.pt"))} batches to finish...')
        time.sleep(120)
    
    print("Batched fetch complete. Merging the torches...")
    
    # once all the torch files are created, load them and concatenate them
    all_tensor = torch.empty((0, 5120))
    all_labels = list()
    batched_torch_files = glob.glob(directory + '/torch/*.pt')
    for batch_id in tqdm(range(len(batched_torch_files))):
        with open(f'{directory}/torch/batch{batch_id}.pt', 'rb') as f:
            batch = torch.load(f)
            all_tensor = torch.cat((all_tensor, batch.tensor), dim=0)
            all_labels.extend(list(batch.labels))
    #if cleanup:
        # remove all directories used for processing
        #os.system(f'rm -r {directory}')
        
    
    return all_tensor, all_labels


def knn_classification(embeddings, clusters, new_data, k=5):
    # Initialize KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    
    # Train KNN with the cluster labels as the target
    knn.fit(embeddings.cpu().numpy(), clusters)
    
    # Predict the cluster for new data points
    predicted_clusters = knn.predict(new_data.cpu().numpy())
    return predicted_clusters


def random_forest_classification(train_embeddings, train_labels, new_data, n_estimators=100):
    # Initialize RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    
    # Train the classifier
    clf.fit(train_embeddings.cpu().detach().numpy(), np.array(train_labels))
    
    # Predict on new data
    predictions = clf.predict(new_data.cpu().detach().numpy())
    
    return clf, predictions


def naive_bayes_classification(train_embeddings, train_labels, new_data):
    # Initialize Naive Bayes Classifier
    clf = GaussianNB()
    
    # Train the classifier
    clf.fit(train_embeddings.cpu().detach().numpy(), np.array(train_labels))
    
    # Predict on new data
    predictions = clf.predict(new_data.cpu().detach().numpy())
    
    return predictions


# Define the contrastive learning model with dropout layers
class ContrastiveModel(nn.Module):
    def __init__(self):
        super(ContrastiveModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(5120, 2048),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
    
    def forward(self, x):
        return self.encoder(x)
            
            
class ClassifierNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = F.dropout(x, p=0.1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        if self.fc3.out_features == 1:
            return torch.sigmoid(self.fc3(x))  # Binary classification
        else:
            return torch.softmax(self.fc3(x), dim=1)  # Multiclass classification


def train_NN_model(train_loader, val_loader, input_size, num_classes, is_binary=True, epochs=20, lr=0.001):
    # Check for CUDA
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = ClassifierNN(input_size, 1 if is_binary else num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Training loop
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # Move data to GPU
            
            if is_binary:
                target = target.unsqueeze(1).float()  # Ensure target shape is [batch_size, 1]
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)  # BCE or CrossEntropy
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)  # Move validation data to GPU
                
                if is_binary:
                    target = target.unsqueeze(1).float()  # Ensure target shape is [batch_size, 1]
                
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                
                # For accuracy calculation
                if is_binary:
                    pred = (output > 0.5).float()
                    correct += (pred == target).sum().item()
                else:
                    _, pred = torch.max(output, 1)
                    correct += (pred == target).sum().item()
        
        # Average loss and accuracy
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        accuracy = correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
    
    return model


# define a Binding_site_dataset class
class Binding_site_dataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
        
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


def pairwise_distance(embeddings):
    """Compute the pairwise distance matrix."""
    dot_product = torch.matmul(embeddings, embeddings.T)
    square_sum = torch.diag(dot_product)
    distances = square_sum.unsqueeze(0) - 2 * dot_product + square_sum.unsqueeze(1)
    return torch.sqrt(torch.relu(distances))


def online_pair_mining(embeddings, labels, margin=1.0):
    """Perform online pair mining by computing distances and selecting pairs."""
    # Step 2: Compute pairwise distances between all embeddings
    distances = pairwise_distance(embeddings)
    
    batch_size = labels.size(0)
    
    # Create masks for positive and negative pairs
    labels = labels.unsqueeze(1)
    positive_mask = torch.eq(labels, labels.T)  # Positive pairs: same label
    negative_mask = torch.ne(labels, labels.T)  # Negative pairs: different label
    
    # Step 3: Mine hardest positive and negative pairs for each anchor
    hardest_positive_dist = torch.max(distances * positive_mask.float(), dim=1)[0]
    hardest_negative_dist = torch.min(distances + 1e6 * (~negative_mask).float(), dim=1)[0]
    
    # Step 4: Apply margin to get the final contrastive loss or triplet loss
    # Example using contrastive loss
    # the loss is minimized when positive pairs have low distances and negative pairs have distances greater than the margin.
    positive_loss = hardest_positive_dist
    negative_loss = F.relu(hardest_negative_dist - margin)
    pos_weight = 0.95  
    neg_weight = 0.05
    loss = pos_weight * positive_loss.mean() + neg_weight * negative_loss.mean()
    return loss.mean()


def supervised_contrastive_loss(embeddings, labels, margin=1.0):
    # Compute pairwise distances
    distances = pairwise_distance(embeddings)
    positive_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0))
    negative_mask = ~positive_mask
    positive_loss = distances[positive_mask]
    negative_loss = F.relu(margin - distances[negative_mask])
    loss = positive_loss.mean() + negative_loss.mean()
    return loss


def get_triplet_mask(labels):
    # Check if labels match for anchor-positive and anchor-negative
    labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
    
    # Create anchor-positive and anchor-negative masks
    mask_anchor_positive = labels_equal.float()
    mask_anchor_negative = 1 - mask_anchor_positive
    
    # Create triplet mask
    triplet_mask = mask_anchor_positive.unsqueeze(2) * mask_anchor_negative.unsqueeze(1)
    return triplet_mask.bool()


def online_triplet_loss(embeddings, labels, margin=1.0, device='cuda'):
    # Get the pairwise distances
    pairwise_distances = pairwise_distance(embeddings)

    # Create a mask for valid triplets
    triplet_mask = get_triplet_mask(labels)
    
    # Reshape distances for broadcasting
    positive_dist = pairwise_distances.unsqueeze(2)  # anchor-positive distances
    negative_dist = pairwise_distances.unsqueeze(1)  # anchor-negative distances
    
    # Compute triplet loss for valid triplets
    triplet_loss = F.relu(positive_dist - negative_dist + margin)

    # Apply triplet mask
    triplet_loss = triplet_loss * triplet_mask.float().to(device)

    # Reduce to mean triplet loss
    valid_triplets = torch.sum(triplet_mask.float())
    triplet_loss = torch.sum(triplet_loss) / (valid_triplets + 1e-16)

    return triplet_loss


def fair_weighted_sampler(dataset, batch_size=150):
    # Count the occurrences of each class
    class_sample_count = np.bincount(dataset.labels)

    # Inverse of class frequency
    class_weights = 1. / class_sample_count

    # Assign weights to each sample based on its class
    sample_weights = class_weights[dataset.labels]
    
    # Create a WeightedRandomSampler to sample indices according to the sample weights
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),  # Number of samples to draw in an epoch
        replacement=True  # Whether to sample with replacement
    )
    
    return sampler


class BalancedBatchSampler(BatchSampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.num_classes = len(np.unique(labels))
        
        # Divide indices by class
        self.class_indices = {cls: np.where(labels == cls)[0] for cls in np.unique(labels)}
        
    def __iter__(self):
        batches = []
        # Sample balanced batches
        for _ in range(len(self.labels) // self.batch_size):
            batch = []
            for cls, indices in self.class_indices.items():
                batch_indices = np.random.choice(indices, self.batch_size // self.num_classes, replace=True)
                batch.extend(batch_indices)
            np.random.shuffle(batch)
            batches.append(batch)
        return iter(batches)
    
    def __len__(self):
        return len(self.labels) // self.batch_size


def validate(model, dataloader, margin=1.0):
    """Validate the model on the validation set."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, labels in dataloader:
            inputs, labels = inputs.cuda(), labels.cuda()
            embeddings = model(inputs)
            loss = online_triplet_loss(embeddings, labels, margin=margin)
            total_loss += loss.item() * inputs.size(0)
    
    return total_loss / len(dataloader.dataset)


def train_with_early_stopping(model, train_loader, val_loader, optimizer, num_epochs=100, margin=2.0, patience=10):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered")
            break
        
        # Training step
        model.train()  # Set model to training mode
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.cuda(), labels.cuda()
            embeddings = model(inputs)
            
            loss = online_triplet_loss(embeddings, labels, margin=margin)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch {epoch}, Batch {batch_idx}, Training Loss: {loss.item()}')

        # Validation step
        val_loss = validate(model, val_loader, margin=margin)
        print(f'Epoch {epoch}, Validation Loss: {val_loss}')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0  # Reset the counter if validation loss improves
            # Optionally, save the best model
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epoch + 1} epochs. Best Validation Loss: {best_val_loss}')
            early_stop = True


def get_validation_ready_for_val_entries(contraster, val_df, embeddings_dir):
    # get entries in val_df
    entries = val_df['Entry']
    
    embeddings = list()
    binary_labels = []
    non_binary_labels = []
    for entry in entries:
        # load the corresponding sequence embedding from the embeddings_dir
        seq_emb = (torch.load(f'{embeddings_dir}/sp|{entry}|esm.pt'))['representations'][48]
        for emb in seq_emb:
            embeddings.append(emb)
        
        # get the labels for the entry - so gross sorry me
        positions = list(val_df[val_df['Entry'] == entry]['collated_all_BS'])[0]
        positions = positions.split(',')
        positions = [x.replace('[', '').replace(']', '').replace(' ', '') for x in positions]
        positions = [int(x) for x in positions]
        cofactor_positions = list(val_df[val_df['Entry'] == entry]['Cofactor sites'])[0]
        cofactor_positions = cofactor_positions.split(',')
        cofactor_positions = [x.replace('[', '').replace(']', '').replace(' ', '') for x in cofactor_positions]
        cofactor_positions = [int(x) for x in cofactor_positions]
        chebi_IDs = list(val_df[val_df['Entry'] == entry]['Cofactor IDs'])[0]
        chebi_IDs = chebi_IDs.split(',')
        chebi_IDs = [x.replace('[', '').replace(']', '').replace(' ', '') for x in chebi_IDs]
        chebi_IDs = [int(x) for x in chebi_IDs if x != '']
        
        cofactor_id_counter = 0
        for i in range(len(seq_emb)):
            if i in positions:
                binary_labels.append("BS")
                if i in cofactor_positions:
                    #get the corresponding chebi ID
                    chebi_ID = chebi_IDs[cofactor_id_counter]
                    non_binary_labels.append(chebi_ID)
                    cofactor_id_counter+=1
                else:
                    non_binary_labels.append("BS_no_label")
            else:
                binary_labels.append("non-BS")
                non_binary_labels.append("non-BS")

    # convert the embeddings to tensors
    embeddings = torch.stack(embeddings)
    
    # normalise the embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    # now validate using the random forest and the contrastive model
    # first run the embeddings through the contrastive model
    contraster.eval()
    with torch.no_grad():
        contraster_embeddings = contraster(embeddings.cuda())
    
    return contraster_embeddings, binary_labels, non_binary_labels


def hyper_contrastive_learning(train_dataset, val_dataset, index_to_label, label_to_index, num_epochs=40, post_shift_margin=1.0, post_shift_patience=5, per_shift_epochs = 5):
    # labels will look like this: EC_Chebi_AA
    # there will be x number of label shifts, where we swap out the labels for sub-strings in the full label string and train on the new labels - hope to get a more general model with better organised space
    
    # get mappings from EC - global label indexes in the dataset
    ECs = []
    Chebis = []
    AAs = []
    EC_Chebis = []
    
    for i in pd.Series(train_dataset.dataset.labels[train_dataset.indices]).unique():
        if index_to_label[i].split('_')[0] not in ECs:
            ECs.append(index_to_label[i].split('_')[0])
        
        if index_to_label[i].split('_')[1] not in Chebis:
            Chebis.append(index_to_label[i].split('_')[1])
            
        if index_to_label[i].split('_')[2] not in AAs:
            AAs.append(index_to_label[i].split('_')[2])
        
        if index_to_label[i].split('_')[0] + '_' + index_to_label[i].split('_')[1] not in EC_Chebis:
            EC_Chebis.append(index_to_label[i].split('_')[0] + '_' + index_to_label[i].split('_')[1])


    ECs = set(ECs)
    Chebis = set(Chebis)
    AAs = set(AAs)
    EC_Chebis = set(EC_Chebis)
    
    # create label_to_index dict for all the unique ECs, Chebis, and AAs with corresponding indec_to_labels dictionaries
    EC_to_index = {EC: i for i, EC in enumerate(ECs)}
    Chebi_to_index = {Chebi: i for i, Chebi in enumerate(Chebis)}
    AA_to_index = {AA: i for i, AA in enumerate(AAs)}
    EC_chebi_to_index = {EC_Chebi: i for i, EC_Chebi in enumerate(EC_Chebis)}
    
    # also need to make a dictionary for binary BS vs nonBS
    BS_binary_to_index = {'BS': 1, 'non-BS': 0}
    
    # create a dictionary that maps the indexes in the EC, Chebi, AA, and EC_Chebi to the global indexes in the dataset
    EC_mapping_to_global = dict()
    Chebi_mapping_to_global = dict()
    AA_mapping_to_global = dict()
    EC_Chebi_mapping_to_global = dict()
    BS_binary_to_global = dict()
    for i in pd.Series(train_dataset.dataset.labels[train_dataset.indices]).unique():
        i_global_label = index_to_label[i]
        EC = i_global_label.split('_')[0]
        Chebi = i_global_label.split('_')[1]
        AA = i_global_label.split('_')[2]
        EC_Chebi = EC + '_' + Chebi
        BS_binary = 'nonBS' if 'nonBS' in i_global_label else 'BS'
        
        for x in ECs:
            if x == EC:
                EC_index = EC_to_index[x]
                # update the EC_index_to_global dictionary I am creating
                # append i to key x in the dictionary
                if EC_index in EC_mapping_to_global:
                    EC_mapping_to_global[EC_index].append(i)
                else:
                    EC_mapping_to_global[EC_index] = [i]
        
        for x in Chebis:
            if x == Chebi:
                Chebi_index = Chebi_to_index[x]
                # update the Chebi_index_to_global dictionary I am creating
                # append i to key x in the dictionary
                if Chebi_index in Chebi_mapping_to_global:
                    Chebi_mapping_to_global[Chebi_index].append(i)
                else:
                    Chebi_mapping_to_global[Chebi_index] = [i]
                    
        for x in AAs:
            if x == AA:
                AA_index = AA_to_index[x]
                # update the AA_index_to_global dictionary I am creating
                # append i to key x in the dictionary
                if AA_index in AA_mapping_to_global:
                    AA_mapping_to_global[AA_index].append(i)
                else:
                    AA_mapping_to_global[AA_index] = [i]
        
        for x in EC_Chebis:
            if x == EC_Chebi:
                EC_Chebi_index = EC_chebi_to_index[x]
                # update the EC_Chebi_index_to_global dictionary I am creating
                # append i to key x in the dictionary
                if EC_Chebi_index in EC_Chebi_mapping_to_global:
                    EC_Chebi_mapping_to_global[EC_Chebi_index].append(i)
                else:
                    EC_Chebi_mapping_to_global[EC_Chebi_index] = [i]
                    
        if BS_binary == 1:
            # Check if key 1 is empty
            if 1 in BS_binary_to_global:
                BS_binary_to_global[1].append(i)
            else:
                BS_binary_to_global[1] = [i]
        else:
            # Check if key 0 is empty
            if 0 in BS_binary_to_global:
                BS_binary_to_global[0].append(i)
            else:
                BS_binary_to_global[0] = [i]
        
        
    schemes = [BS_binary_to_global, AA_mapping_to_global, EC_mapping_to_global, Chebi_mapping_to_global]
    schemes = [EC_mapping_to_global, Chebi_mapping_to_global, AA_mapping_to_global, BS_binary_to_global]
    scheme_margins = [5,4,3,2]
    
    contraster = ContrastiveModel().cuda()
    count = 0
    for scheme in schemes:
        scheme_train_dataset = train_dataset
        scheme_val_dataset = val_dataset
        # use the scheme mapping to convert the training dataset labels to the corresponding scheme labels
        for i in range(len(scheme_train_dataset.dataset.labels[scheme_train_dataset.indices])):
            for key, value in scheme.items():
                if i in value:
                    scheme_train_dataset.dataset.labels[i] = key
        
        # use the scheme mapping to convert the validation dataset labels to the corresponding scheme labels
        for i in range(len(scheme_val_dataset.dataset.labels[scheme_val_dataset.indices])):
            for key, value in scheme.items():
                if i in value:
                    scheme_val_dataset.dataset.labels[i] = key 
                    
        # make the data loader for the training and validation datasets
        train_loader = DataLoader(scheme_train_dataset, batch_size=1500, shuffle=True)
        val_loader = DataLoader(scheme_val_dataset, batch_size=1500, shuffle=True)
        
        # train the model with the scheme train_loader and val_loader
        optimizer = torch.optim.Adam(contraster.parameters(), lr=0.0001, weight_decay=1e-5)  # L2 regularization
        train_with_early_stopping(contraster, train_loader, val_loader, optimizer, num_epochs=per_shift_epochs, margin=scheme_margins[count], patience=per_shift_epochs)
    
    # convert the training dataset labels to EC_Chebi_mapping_to_global
    for i in range(len(train_dataset.dataset.labels[train_dataset.indices])):
        for key, value in EC_Chebi_mapping_to_global.items():
            if i in value:
                train_dataset.dataset.labels[i] = key
    
    # now that the model has been trained on all the schemes, we can do a final training on the EC_Chebi schemes to double 
    optimizer = torch.optim.Adam(contraster.parameters(), lr=0.0001, weight_decay=1e-5)
    #train_with_early_stopping(contraster, train_loader, val_loader, optimizer, num_epochs=num_epochs, margin=post_shift_margin, patience=post_shift_patience)
    
    return contraster
    
    


def main():
    # load validation sequences
    file = "/scratch/project/squid/OMEGA/BS_model/Low40_mmseq_ID_300_exp_subset.txt"
    with open(file, 'r') as f:
        val_Entries = f.readlines()
    val_Entries = [Entry.strip() for Entry in val_Entries]
    
    if 'df_BS_hierarchical_sampling_BS.tsv' in os.listdir('/scratch/project/squid/OMEGA/dfs/'):
        # load the tsvs
        df_BS = pd.read_csv('/scratch/project/squid/OMEGA/dfs/df_BS_hierarchical_sampling_BS.tsv', sep='\t')
        df_nonBS = pd.read_csv('/scratch/project/squid/OMEGA/dfs/df_nonBS_hierarchical_sampling_BS.tsv', sep='\t')
        val_df = pd.read_csv('/scratch/project/squid/OMEGA/dfs/val_df.tsv', sep='\t')
    else:
        df = pd.read_csv('/scratch/project/squid/OMEGA/dfs/has_BS_df.tsv', sep='\t')
        
        # get all the entries that exist in the embeddings folder
        entries = [x.split('|')[1] for x in os.listdir('/scratch/project/squid/OMEGA/BS_model/embeddings')]
        
        # subset the df to only include the entries that exist in the embeddings folder
        df = df[df['Entry'].isin(entries)]
        df = df.reset_index(drop=True)

        # get the cofactor sites and and EC tiers for the df
        df = get_EC_tiers(df)
        
        # process the cofactor IDs
        df = get_cofactor_sites(df)
        
        df["collated_all_BS"] = get_BS_idx_from_uniprot(df)
        
        # get the BS positions and IDs
        df = get_BS_pos_from_uniprot_processed_tsv(df)
        
        # get the Flexible EC_labeling for the df, use a threshold that is high like 100-200
        df = Flexible_EC_labeling(df, EC_threshold = 300)
        
        # get the BS site ID columns
        df = BS_site_ID_column_splitting(df)
        
        # Make val_df
        val_df = df[df['Entry'].isin(val_Entries)]
        
        # drop all rows that have Entry in val_Entries
        df = df[~df['Entry'].isin(val_Entries)]
        
        # save the df to a tsv file
        df.to_csv('/scratch/project/squid/OMEGA/dfs/processed_for_new_label_scheme_clustering.tsv', sep='\t', index=False)

        df_BS, df_nonBS = sample_residue_embeddings(df, embeddings_dir='/scratch/project/squid/OMEGA/BS_model/embeddings', EC = "EC_T-filtered", sample_limit=100)
        print("Embeddings sampled")
        print("number of samples:")
        print("BS: ", len(df_BS))
        print("non-BS: ", len(df_nonBS))
        df_BS.to_csv('/scratch/project/squid/OMEGA/dfs/df_BS_hierarchical_sampling_BS.tsv', sep='\t', index=False)
        df_nonBS.to_csv('/scratch/project/squid/OMEGA/dfs/df_nonBS_hierarchical_sampling_BS.tsv', sep='\t', index=False)
        # save the val_df
        val_df.to_csv('/scratch/project/squid/OMEGA/dfs/val_df.tsv', sep='\t', index=False)

    # load the embeddings and labels from Binding SITES
    if 'all_BS_sampled.pt' in os.listdir('/scratch/project/squid/OMEGA/dfs/'):
        print("Loading BS embeddings...")
        embeddings_BS = torch.load('/scratch/project/squid/OMEGA/dfs/all_BS_sampled.pt')
        # normalise the embeddings
        embeddings_BS = F.normalize(embeddings_BS, p=2, dim=1)
        print("Loading BS labels...")
        with open('/scratch/project/squid/OMEGA/dfs/labels_BS.pkl', 'rb') as f:
            labels_BS = pickle.load(f)
    else:
        # reset the df_BS index
        df_BS = df_BS.reset_index(drop=True)
        BS_samples_to_get = df_BS['pos']
        BS_index = df_BS.index
        embeddings_BS, labels_BS = process_pair_data(BS_samples_to_get, BS_index,'/scratch/project/squid/OMEGA/dfs/all_BS_sampled', batch_size=2000, cleanup=True)
        torch.save(embeddings_BS, '/scratch/project/squid/OMEGA/dfs/all_BS_sampled.pt')
        with open('/scratch/project/squid/OMEGA/dfs/labels_BS.pkl', 'wb') as f:
            pickle.dump(labels_BS, f)
        embeddings_BS = F.normalize(embeddings_BS, p=2, dim=1)
        
    # load the embeddings and labels from non-BS
    if 'all_nonBS_sampled.pt' in os.listdir('/scratch/project/squid/OMEGA/dfs/'):
        print("Loading non-BS embeddings...")
        embeddings_nonBS = torch.load('/scratch/project/squid/OMEGA/dfs/all_nonBS_sampled.pt')
        # normalise the embeddings
        embeddings_nonBS = F.normalize(embeddings_nonBS, p=2, dim=1)
        print("Loading non-BS labels...")
        with open('/scratch/project/squid/OMEGA/dfs/labels_nonBS.pkl', 'rb') as f:
            labels_nonBS = pickle.load(f)
    else:
        # reset the index
        df_nonBS = df_nonBS.reset_index(drop=True)
        nonBS_samples_to_get = df_nonBS['pos']
        nonBS_index = df_nonBS.index
        embeddings_nonBS, labels_nonBS = process_pair_data(nonBS_samples_to_get, nonBS_index,'/scratch/project/squid/OMEGA/dfs/all_nonBS_sampled', batch_size=2000, cleanup=True)
        torch.save(embeddings_nonBS, '/scratch/project/squid/OMEGA/dfs/all_nonBS_sampled.pt')
        with open('/scratch/project/squid/OMEGA/dfs/labels_nonBS.pkl', 'wb') as f:
            pickle.dump(labels_nonBS, f)
        embeddings_nonBS = F.normalize(embeddings_nonBS, p=2, dim=1)
        
    #TODO Check if the non-BS somehow include BS.
    
    # use the indexes in labels_BS
    labels_BS_string = list()
    print(labels_BS[-100:])
    print(len(labels_BS))
    for index in labels_BS:
        # get the corresponding index labels in df_BS
        #print(index)
        BS = str(df_BS['BS_ID'][index])
        
        # iterate through the chebi_groupings dictionary and find the key that contains the BS
        for key, value in chebi_groupings.items():
            if BS in value:
                BS = key
                break
        
        
        label = df_BS['EC'][index] + '_' + BS + '_' + df_BS['AA'][index]
        #label = df_BS['EC'][index]
        #label = BS
        labels_BS_string.append(label)
    
    # do the same for the indexes in labels_nonBS
    labels_nonBS_string = list()
    for index in labels_nonBS:
        # get the corresponding index labels in df_nonBS
        label = str(df_nonBS['EC'][index]) + "_non-BS_" + df_nonBS['AA'][index]
        #label = "non-BS"
        labels_nonBS_string.append(label)
    
    # get the value counts, and save any labels that have less than 20 samples
    print("Value counts of labels_BS_string: ")
    print(pd.Series(labels_BS_string).value_counts())
    # get the less than threshold labels
    less_than_threshold = pd.Series(labels_BS_string).value_counts()[pd.Series(labels_BS_string).value_counts() < 20].index
    
    # print them
    print("Labels with less than 20 samples: ")
    print(less_than_threshold)
    
    MERGE = True
    if MERGE:
        # concatenate the BS and non-BS embeddings/labels
        embeddings_BS = torch.cat((embeddings_BS, embeddings_nonBS), dim=0)
        # normalise again
        embeddings_BS = F.normalize(embeddings_BS, p=2, dim=1)
        labels_BS_string = labels_BS_string + labels_nonBS_string
        
        
    # put the labels into a tensor    
    label_to_index = {label: idx for idx, label in enumerate(set(labels_BS_string))}
    labels_BS_encoded = [label_to_index[label] for label in labels_BS_string]
    labels_BS_tensor = torch.tensor(labels_BS_encoded)
    index_to_label = {v: k for k, v in label_to_index.items()}  # for getting back to labels
    
    # now add it to the class dataset
    BS_dataset = Binding_site_dataset(embeddings_BS, labels_BS_tensor)
    
    # convert the labels_tensor indices back to string labels
    labels_BS_string = [index_to_label[label] for label in labels_BS_encoded]
    print("Value counts of labels_BS_string: ")
    print(pd.Series(labels_BS_string).value_counts())

    # get the value counts of labels_BS_tensor
    print("Value counts of labels_BS_tensor: ")
    print(torch.unique(labels_BS_tensor, return_counts=True))
    
    torch.manual_seed(1)
    batch_size = 1200
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(BS_dataset, [0.6, 0.3, 0.1])
    
    #hyper_contrastive_learning(train_dataset, val_dataset, index_to_label, label_to_index, num_epochs=100, post_shift_margin=1.0, post_shift_patience=10, per_shift_epochs = 10)
    #return
    # Count occurrences of each label
    #class_sample_count = np.array([len(np.where(train_dataset.dataset.labels == t)[0]) for t in np.unique(train_dataset.dataset.labels)])
    #weight = 1. / class_sample_count

    # Assign weights for each sample in the dataset
    #samples_weight = np.array([weight[t] for t in train_dataset.dataset.labels])

    # Convert to tensor and create WeightedRandomSampler
    #samples_weight = torch.from_numpy(samples_weight)
    #sampler = WeightedRandomSampler(weights=samples_weight, num_samples=len(samples_weight), replacement=True)

    # Use sampler in DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    if 'BS_contrastive_learner.pthX' in os.listdir('/scratch/project/squid/OMEGA/'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        contraster = ContrastiveModel()
        contraster.load_state_dict(torch.load('/scratch/project/squid/OMEGA/BS_contrastive_learner.pth'))
        contraster.to(device)
    else:                        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(device)
        #train_loader.to(device)
        #test_loader.to(device)
        #val_loader.to(device)
        contraster = hyper_contrastive_learning(train_dataset, val_dataset, index_to_label, label_to_index, num_epochs=40, post_shift_margin=1.0, post_shift_patience=5, per_shift_epochs = 5)
        #contraster.to(device)
        #optimizer = torch.optim.Adam(contraster.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
        #train_with_early_stopping(contraster, train_loader, val_loader, optimizer, num_epochs=100, margin=2.0, patience=5)
        torch.save(contraster.state_dict(), '/scratch/project/squid/OMEGA/BS_contrastive_learner.pth')
    
    print(test_dataset.dataset.embeddings[test_dataset.indices].shape)
    print(train_dataset.dataset.embeddings[train_dataset.indices].shape)
    contraster.eval()
    with torch.no_grad():
        test_contrasted_embeddings = contraster(test_dataset.dataset.embeddings[test_dataset.indices].cuda()).cpu()
        train_contrasted_embeddings = contraster(train_dataset.dataset.embeddings[train_dataset.indices].cuda()).cpu()
        val_contrasted_embeddings = contraster(val_dataset.dataset.embeddings[val_dataset.indices].cuda()).cpu()
        
    test_labels = list(test_dataset.dataset.labels[test_dataset.indices])
    test_labels = [int(label) for label in test_labels]
    test_labels_BS = list()
    for label in test_labels:
        test_labels_BS.append(label)
    test_labels_BS = [index_to_label[label] for label in test_labels_BS]    # convert the indexes to string labels
    #test_labels_BS = ['non-BS' if 'non-BS' in label else "BS" for label in test_labels_BS]
    
    # drop the AA in the labels
    test_labels_BS = [label.split('_')[0] + '_' + label.split('_')[1] for label in test_labels_BS]
        
    # get the hclustering of the test_contrasted_embeddings
    h_cluster(test_contrasted_embeddings.cpu(), test_labels_BS)
    
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    #                                                   Binary Model                                               #
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    
    labels = list(train_dataset.dataset.labels[train_dataset.indices])
    labels = [int(label) for label in labels]
    labels_BS = list()
    for label in labels:
        labels_BS.append(label)
    print(labels_BS[0:40])
    labels_BS = [index_to_label[label] for label in labels_BS]    # convert the indexes to string labels
    print(labels_BS[0:40])
    labels_BS = ['non-BS' if 'non-BS' in label else "BS" for label in labels_BS]
    print(labels_BS[0:40])
        
    test_labels = list(test_dataset.dataset.labels[test_dataset.indices])
    test_labels = [int(label) for label in test_labels]
    test_labels_BS = list()
    for label in test_labels:
        test_labels_BS.append(label)
    test_labels_BS = [index_to_label[label] for label in test_labels_BS]    # convert the indexes to string labels
    test_labels_BS = ['non-BS' if 'non-BS' in label else "BS" for label in test_labels_BS]
    
    val_labels = list(val_dataset.dataset.labels[val_dataset.indices])
    val_labels = [int(label) for label in val_labels]
    val_labels_BS = list()
    for label in val_labels:
        val_labels_BS.append(label)
    val_labels_BS = [index_to_label[label] for label in val_labels_BS]    # convert the indexes to string labels
    val_labels_BS = ['non-BS' if 'non-BS' in label else "BS" for label in val_labels_BS]
    
    NN_binary_label_to_index = {label: idx for idx, label in enumerate(set(test_labels_BS))}
    NN_binary_labels_BS_encoded = [NN_binary_label_to_index[label] for label in test_labels_BS]
    test_NN_labels_BS_tensor = torch.tensor(NN_binary_labels_BS_encoded)
    
    # do the same for the train_labels and test_labels
    NN_binary_label_to_index = {label: idx for idx, label in enumerate(set(labels_BS))}
    NN_binary_labels_BS_encoded = [NN_binary_label_to_index[label] for label in labels_BS]
    train_NN_labels_BS_tensor = torch.tensor(NN_binary_labels_BS_encoded)
    index_to_label_NN = {v: k for k, v in NN_binary_label_to_index.items()}  # for getting back to labels
    
    # do the same for val_dataset
    NN_binary_label_to_index = {label: idx for idx, label in enumerate(set(val_labels_BS))}
    NN_binary_labels_BS_encoded = [NN_binary_label_to_index[label] for label in val_labels_BS]
    val_NN_labels_BS_tensor = torch.tensor(NN_binary_labels_BS_encoded)
    
    # convert back to indexes so that we can use them in the model
    train_dataset = Binding_site_dataset(train_contrasted_embeddings, train_NN_labels_BS_tensor)
    test_dataset = Binding_site_dataset(test_contrasted_embeddings, test_NN_labels_BS_tensor)
    val_dataset = Binding_site_dataset(val_contrasted_embeddings, val_NN_labels_BS_tensor)
    
    # make dataloaders
    train_loader = DataLoader(train_dataset, batch_size=5000, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5000, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=5000, shuffle=True)
    
    val_contrasted_embeddings, val_binary, val_non_binary= get_validation_ready_for_val_entries(contraster, val_df, '/scratch/project/squid/OMEGA/BS_model/embeddings') # actual validation, not the training validation set
    
    if 'binary_NN_classifier.pthX' not in os.listdir('/scratch/project/squid/OMEGA/BS_model/'):
        model = train_NN_model(train_loader, val_loader, input_size=128, num_classes=2, is_binary=True)
        torch.save(model.state_dict(), '/scratch/project/squid/OMEGA/BS_model/binary_NN_classifier.pth')
        test_predictions = model(test_contrasted_embeddings.cuda())
        test_predictions = (test_predictions > 0.5).float()
        test_predictions = [index_to_label_NN[int(label[0])] for label in test_predictions.cpu().detach().numpy()]
    else:
        model = ClassifierNN(128, 1).to(device)
        model.load_state_dict(torch.load('/scratch/project/squid/OMEGA/BS_model/binary_NN_classifier.pth'))
        test_predictions = model(test_contrasted_embeddings.cuda())
        test_predictions = (test_predictions > 0.5).float()
        test_predictions = [index_to_label_NN[int(label[0])] for label in test_predictions.cpu().detach().numpy()]
        
        
    # print the accuracy of the model
    accuracy = accuracy_score(test_labels_BS, test_predictions)
    precision = precision_score(test_labels_BS, test_predictions, average='weighted', pos_label='BS')
    recall = recall_score(test_labels_BS, test_predictions, average='weighted', pos_label='BS')
    f1 = f1_score(test_labels_BS, test_predictions, average='weighted', pos_label='BS')
    print()
    print("##################################")
    print("Binary NN test results")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("##################################")
    print()
    val_predictions = model(val_contrasted_embeddings.cuda())
    val_predictions = (val_predictions > 0.5).float()
    val_predictions = [index_to_label_NN[int(label[0])] for label in val_predictions.cpu().detach().numpy()]
    print("Value counts of val_predictions: ")
    print(pd.Series(val_predictions).value_counts())
    print("Value counts of val_non_binary: ")
    print(pd.Series(val_binary).value_counts())  
    # print the accuracy of the model
    accuracy = accuracy_score(val_binary, val_predictions)
    precision = precision_score(val_binary, val_predictions, pos_label = "BS")
    recall = recall_score(val_binary, val_predictions, pos_label = "BS")
    f1 = f1_score(val_binary, val_predictions, pos_label = "BS")    
    # plot and save a confusion matrix of the model
    cm = confusion_matrix(val_binary, val_predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['non-BS', 'BS'], yticklabels=['non-BS', 'BS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix NN Binary BS prediction')
    plt.savefig('/scratch/project/squid/OMEGA/BS_model/RF_confusion_matrix_binary_BS_prediction.png')
    print()
    print("##################################")
    print("Binary NN validation results")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("##################################")
    print()
    
    return
    
    if 'binary_random_forest_classifier.pklX' not in os.listdir('/scratch/project/squid/OMEGA/BS_model/'):
        clf, test_predictions = random_forest_classification(train_contrasted_embeddings, labels_BS, test_contrasted_embeddings)
        # save clf
        with open('/scratch/project/squid/OMEGA/BS_model/binary_random_forest_classifier.pkl', 'wb') as f:
            pickle.dump(clf, f)
    else:
        with open('/scratch/project/squid/OMEGA/BS_model/binary_random_forest_classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
        test_predictions = clf.predict(test_contrasted_embeddings.cpu().detach().numpy())

    
    # print the accuracy of the model
    accuracy = accuracy_score(test_labels_BS, test_predictions)
    precision = precision_score(test_labels_BS, test_predictions, average='weighted')
    recall = recall_score(test_labels_BS, test_predictions, average='weighted')
    f1 = f1_score(test_labels_BS, test_predictions, average='weighted')
    print()
    print("##################################")
    print("Binary RF test results")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("##################################")
    print()
    val_predictions = clf.predict(val_contrasted_embeddings.cpu().detach().numpy())
    # get value counts in val_predictions and val_non_binary
    print("Value counts of val_predictions: ")
    print(pd.Series(val_predictions).value_counts())
    print("Value counts of val_non_binary: ")
    print(pd.Series(val_binary).value_counts())  
    # print the accuracy of the model
    accuracy = accuracy_score(val_binary, val_predictions)
    precision = precision_score(val_binary, val_predictions, pos_label = "BS")
    recall = recall_score(val_binary, val_predictions, pos_label = "BS")
    f1 = f1_score(val_binary, val_predictions, pos_label = "BS")    
    # plot and save a confusion matrix of the model
    cm = confusion_matrix(val_binary, val_predictions)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['non-BS', 'BS'], yticklabels=['non-BS', 'BS'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix RF Binary BS prediction')
    plt.savefig('/scratch/project/squid/OMEGA/BS_model/RF_confusion_matrix_binary_BS_prediction.png')
    print()
    print("##################################")
    print("Binary RF validation results")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("##################################")
    print()
    
    
    
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    #                                               multi_class Model                                              #
    #                                           predicts BS-type and non_BS                                        #
    ################################################################################################################
    ################################################################################################################
    ################################################################################################################
    
    # TODO need to remove anything that has BS but isn't labelled with chebi--- probs gonna be misslabeled by model as non-BS
    
    labels = list(train_dataset.dataset.labels[train_dataset.indices])
    labels = [int(label) for label in labels]
    labels_BS = list()
    for label in labels:
        labels_BS.append(label)
    labels_BS = [index_to_label[label] for label in labels_BS]    # convert the indexes to string labels
    
    # filter the labels_BS so that all labels with _non-BS are relabelled to non-BS and all else are BS
    labels_BS = ['non-BS' if 'non-BS' in label else label.split('_')[1] for label in labels_BS]

    
    if 'multi_class_random_forest_classifier.pklX' not in os.listdir('/scratch/project/squid/OMEGA/BS_model/'):
        clf, test_predictions = random_forest_classification(train_contrasted_embeddings, labels_BS, test_contrasted_embeddings)
        # save clf
        with open('/scratch/project/squid/OMEGA/BS_model/multi_class_random_forest_classifier.pkl', 'wb') as f:
            pickle.dump(clf, f)
    else:
        with open('/scratch/project/squid/OMEGA/BS_model/multi_class_random_forest_classifier.pkl', 'rb') as f:
            clf = pickle.load(f)
            # get the test results from the model
            test_predictions = clf.predict(test_contrasted_embeddings.cpu().detach().numpy())
    
    
    test_labels = list(test_dataset.dataset.labels[test_dataset.indices])
    test_labels = [int(label) for label in test_labels]
    test_labels_BS = list()
    for label in test_labels:
        test_labels_BS.append(label)
    test_labels_BS = [index_to_label[label] for label in test_labels_BS]    # convert the indexes to string labels
    test_labels_BS = ['non-BS' if 'non-BS' in label else label.split('_')[1] for label in test_labels_BS]
    
    # print the accuracy of the model
    accuracy = accuracy_score(test_labels_BS, test_predictions)
    precision = precision_score(test_labels_BS, test_predictions, average='weighted')
    recall = recall_score(test_labels_BS, test_predictions, average='weighted')
    f1 = f1_score(test_labels_BS, test_predictions, average='weighted')
    # print the accuracy of the model
    print()
    print("##################################")
    print("Multiclass test results")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("##################################")
    print()
    
    # determine clusters which are have high confusion. Make a massive heatmap of all the labels and their confusion
    unique_labels = np.unique(test_labels_BS) # Get unique labels from the test set
    conf_mat = confusion_matrix(test_labels_BS, test_predictions, labels=unique_labels)    
    # Normalize the confusion matrix by the number of samples in each class
    #conf_mat_normalized = conf_mat / conf_mat.sum(axis=1, keepdims=True)

    # Store the confusion values (excluding diagonal)
    confusion = []

    # Collect pairs of labels with their corresponding confusion values
    for i in range(len(conf_mat)):
        for j in range(len(conf_mat)):
            if i != j:
                confusion.append((unique_labels[i], unique_labels[j], conf_mat[i][j]))

    # Sort the confusion pairs by the confusion value in descending order
    confusion.sort(key=lambda x: x[2], reverse=True)

    # Print the top 50 most confused label pairs
    print("Most confused labels: ")
    for i in range(min(50, len(confusion))):
        print(f"Labels {confusion[i][0]} vs {confusion[i][1]}: Confusion = {confusion[i][2]}")
    
    val_contrasted_embeddings, val_binary, val_non_binary = get_validation_ready_for_val_entries(contraster, val_df, '/scratch/project/squid/OMEGA/BS_model/embeddings')
    # TODO check if the IDs belong to the chebi_groupings and convert them to their new labels
    
    # now go through the embeddings and the val_non_binary, find any with label BS_no_label and drop them
    # Also convert the chebi_IDs to their corresponding chebi_groupings

    for i, label in enumerate(val_non_binary):
        if label == "BS_no_label":
            val_non_binary.pop(i)
            val_binary.pop(i)
            # drop the corresponding embedding
            val_contrasted_embeddings = torch.cat((val_contrasted_embeddings[:i], val_contrasted_embeddings[i+1:]), dim=0)
        # convert the chebi_IDs to their corresponding chebi_groupings
        for key, value in chebi_groupings.items():
            if label in value:
                val_non_binary[i] = key
                break
    
    
    print("value counts of val_binary: ")
    print(pd.Series(val_binary).value_counts())
    val_predictions = clf.predict(val_contrasted_embeddings.cpu().detach().numpy())
    print("value counts of val_predictions: ")
    print(pd.Series(val_predictions).value_counts())
    
    # print the accuracy of the model
    accuracy = accuracy_score(val_non_binary, val_predictions)
    precision = precision_score(val_non_binary, val_predictions, zero_division=1, negative_label='non-BS')
    recall = recall_score(val_non_binary, val_predictions, zero_division=1)
    f1 = f1_score(val_non_binary, val_predictions, zero_division=1)
    print()
    print("##################################")
    print("Multiclass validation results")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)
    print("##################################")
    print()
    
    h_cluster(test_contrasted_embeddings.cpu(), test_labels_BS)
    


if __name__ == "__main__":
    main()
