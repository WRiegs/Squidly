
import sys
sys.path.append('/disk1/ariane/vscode/enzyme-tk/')
sys.path.append('/disk1/ariane/vscode/enzyme-tk/enzymetk/')
from enzymetk.predict_catalyticsite_step import ActiveSitePred
from enzymetk.save_step import Save
import pandas as pd
import os
from Bio import SeqIO
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import time

# This should be where you downloaded the data from zotero, there is a folder in there called AS_inference
# This contains the models and the data needed to run the tool
#squidly_dir = '/disk1/share/software/AS_inference/'
squidly_dir = '/disk1/ariane/vscode/squidly/manuscript/up_to_date_aegan/'
num_threads = 1
id_col = 'Entry'
seq_col = 'Sequence'
data_dir = '/disk1/ariane/vscode/squidly/manuscript/AEGAN_extracted_sequences/'
files = os.listdir(data_dir)
files = ['PC', 'NN', 'EF_superfamily', 'EF_fold', 'EF_family', 'HA_superfamily']
# parse sequence fasta file
with open('aegan_time_log.txt', 'a+') as fout:
    #fout.write('method,time(s),dataset,size,model\n')
    model = 'esm2_t36_3B_UR50D' #
    for f in files:
        df = pd.read_csv(f'{data_dir}{f}/{f}.tsv', sep='\t')
        start_time = time.time()
        df << (ActiveSitePred(id_col, seq_col, squidly_dir, num_threads, esm2_model=model) >> Save(f'squidly/squidly_as_pred_3B_{f}.pkl'))
        fout.write(f'squidly,{time.time() - start_time},{f},{len(df)},{model}\n')
        print("--- %s seconds ---" % (time.time() - start_time))