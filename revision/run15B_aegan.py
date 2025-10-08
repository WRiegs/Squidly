import pandas as pd
import os 

data_dir = '/disk1/ariane/vscode/squidly/manuscript/AEGAN_extracted_sequences/'

files = ['PC',
         'NN',
         'EF_superfamily',
         'EF_fold',
         'family_specific',
         'EF_family',
         'HA_superfamily'] 


model_dir = 'output/uni3175_aegan/models/'
for f in files:
    fasta_label = f'{data_dir}{f}/{f}.fasta'
    os.system(f'mkdir output/families_15B/{f}/')
    os.system(f'squidly run {fasta_label} esm2_t48_15B_UR50D output/families_15B/{f}/ --model-folder {model_dir}')