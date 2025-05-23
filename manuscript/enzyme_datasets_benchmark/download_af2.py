# Download AF2 files for each of the splits
import os
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
import requests
from docko import *
import numpy as np
   
# Now lets run the download in paralell 
def run(uniprot_accessions):
    print("Downloading ", len(uniprot_accessions))
    for uniprot_accession in uniprot_accessions:
        output_file = f'{output_dir}{uniprot_accession}.pdb'
        get_alphafold_structure(uniprot_accession, output_file)
        
runs = []

s_df = pd.read_csv('Low30_mmseq_ID_exp_subset_train.csv')#, sep='\t', header=None)
output_dir = '/disk1/ariane/vscode/squidly/enzyme_datasets_benchmark/structures/'
to_download = []

n_threads = 40
files = list(os.listdir(output_dir))
df_split = np.array_split(s_df, n_threads)

for df in df_split:
    chunks = []
    for e in df['Entry'].values:
        chunks.append(e)
    to_download.append(chunks)

# Use ThreadPool for multithreading
pool = ThreadPool() # Or how many threads you have 

# Map tasks to threads
results = pool.map(run, to_download)

# Close the pool and wait for tasks to complete
pool.close()
pool.join()
