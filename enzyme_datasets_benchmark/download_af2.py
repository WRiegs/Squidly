# Download AF2 files for each of the splits
import os
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
import requests
from docko import *
        
# Now lets run the download in paralell 
def run(uniprot_accessions):
    for uniprot_accession in uniprot_accessions:
        output_file = f'{output_dir}{uniprot_accession}.pdb'
        get_alphafold_structure(uniprot_accession, output_file)
        
runs = []

s_df = pd.read_csv('Low30_mmseq_ID_exp_subset_train.csv')#, sep='\t', header=None)
output_dir = '/disk1/ariane/vscode/squidly/enzyme_datasets_benchmark/structures/'
to_download = []
files = list(os.listdir(output_dir))
entries = set([f.split('.')[0] for f in files])
for e in s_df['Entry'].values:
    print(e)
    if e not in entries:
        print(e)
        to_download.append(e)

# Use ThreadPool for multithreading
pool = ThreadPool(20) # Or how many threads you have 

# Map tasks to threads
results = pool.map(run, [to_download])

# Close the pool and wait for tasks to complete
pool.close()
pool.join()
