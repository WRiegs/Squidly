###############################################################################
#                                                                             #
#    This program is free software: you can redistribute it and/or modify     #
#    it under the terms of the GNU General Public License as published by     #
#    the Free Software Foundation, either version 3 of the License, or        #
#    (at your option) any later version.                                      #
#                                                                             #
#    This program is distributed in the hope that it will be useful,          #
#    but WITHOUT ANY WARRANTY; without even the implied warranty of           #
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            #
#    GNU General Public License for more details.                             #
#                                                                             #
#    You should have received a copy of the GNU General Public License        #
#    along with this program. If not, see <http://www.gnu.org/licenses/>.     #
#                                                                             #
###############################################################################

"""
Author: Ariane Mora
Date: September 2024
"""
import re

import typer
import sys
import pandas as pd
import os
from typing_extensions import Annotated
from os.path import dirname, join as joinpath
import subprocess
from Bio import SeqIO
import subprocess
import timeit
import logging
from sciutil import SciUtil
from tqdm import tqdm
import numpy as np
from enzymetk.sequence_search_blast import BLAST
from Bio import SeqIO
import os
from Bio import AlignIO
# Read in squidly results
import numpy as np
import re
from sciutil import SciUtil
import pandas as pd

u = SciUtil()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


app = typer.Typer()



u = SciUtil()


def align_blast_to_seq(df, database, output_folder) -> pd.DataFrame:
    """ 
    Align the sequneces into BLAST. 
    Note expects the database to have an entry as the sequnece ID and the residue to be the catalytic residues.
    """
    predicted_active_sites = {}
    missing = 0
    uniprot_id_to_active_site = dict(zip(database['Entry'], database['Residue']))
    for query, uniprot in df[['From', 'target']].values:
        missing = 0
        if not uniprot or not isinstance(uniprot, str):
            missing += 1
        else:
            fin = os.path.join(output_folder, f'{uniprot}_{query}.msa')
            # Read the alignment
            active_sites = [int(x) for x in uniprot_id_to_active_site.get(uniprot).split('|')] # type: ignore
            alignment = AlignIO.read(fin, 'fasta')
            # get the existing one and then calculate the position gapped
            records = {}
            for record in alignment:
                records[record.id] = record.seq
            # Now get the active site
            position_count = 0
            active_pred = []
            query_seq = records[query]
            query_count = 0
            x = 0
            for i, v in enumerate(records[uniprot]):
                if position_count in active_sites:
                    if query_count < len(query_seq.replace('-', '')):
                        active_pred.append(query_count)
                    if query_seq[i] != v:
                        x += 1
                        #print(query, uniprot, v, query_seq[i])
                if v != '-':
                    position_count += 1
                if query_seq[i] != '-' and query_seq[i] != ' ':
                    query_count += 1
        
            predicted_active_sites[query] = '|'.join([str(s) for s in active_pred])
            # Now we can just add on the
    df['BLAST_residues'] = [predicted_active_sites.get(label) for label in df['From'].values]
    return df


def run_blast(query_df, database_df, output_folder, run_name, id_col='id', seq_col='seq') -> pd.DataFrame:
    # Convert databaset to a fasta file in the output folder
    database_fasta = os.path.join(output_folder, f'{run_name}_database.fasta')
    with open(database_fasta, 'w+') as fout:
        for seq_id, seq in database_df[['Entry', 'Sequence']].values:
            done_records = []
            # Remove all the ids
            new_id = re.sub('[^0-9a-zA-Z]+', '', seq_id)
            if new_id not in done_records:
                fout.write(f">{new_id}\n{seq}\n")
            else:
                u.warn_p(['Had a duplicate record! Only keeping the first entry, duplicate ID:', new_id])
    # Run BLAST now with the db
    blast_df = (query_df << (BLAST(id_col, seq_col, database=database_fasta, args=['--ultra-sensitive'])))
    blast_df = blast_df.sort_values(by='sequence identity', ascending=False)
    
    # Remove duplicates 
    blast_df.drop_duplicates('query', inplace=True)

    #Then join up with all df
    blast_df.set_index('query', inplace=True)
    query_df.set_index(id_col, inplace=True)
    query_df['From'] = query_df.index
    test_df = query_df.join(blast_df, how='left')
    
    # Make a dictionary from the fasta file
    uniprot_id_to_seq = dict(zip(database_df.Entry, database_df.Sequence))
    output_folder = os.path.join(output_folder, "msa")
    os.system(f'mkdir {output_folder}')
    for name, seq, uniprot in test_df[['From', 'seq', 'target']].values:
        fin = os.path.join(output_folder, f'{uniprot}_{name}.fa')
        with open(fin, 'w+') as fout:
            fout.write(f'>{uniprot}\n{uniprot_id_to_seq.get(uniprot)}\n')
            fout.write(f'>{name}\n{seq}')
        # Now run clustalomega
        os.system(f'clustalo --force -i {fin} -o {fin.replace(".fa", ".msa")}')
        
    # Now we can align to the sequneces
    return align_blast_to_seq(test_df, database_df, output_folder)        
      
                    
                    
def compute_uncertainties(df, prob_columns, mean_prob=0.5):
    means, aleatorics, epistemics, residues, variance  = [], [], [], [], []
    for p1, p2, p3, p4, p5 in tqdm(df[prob_columns].values):
        mean_values = []
        aleatoric_values = []
        epistemic_values = []
        variance_values = []
        indicies = []
        for j in range(0, len(p1)):
            # Aleatoric: average predicted entropy
            eps = 1e-8  # for numerical stability
            # For each value we want the mean and the variance and the uncertainty
            all_probs = [p1[j] + eps, p2[j] + eps, p3[j] + eps, p4[j] + eps, p5[j] + eps]
            mean_probs = np.mean(all_probs)
            aleatoric = -(mean_probs * np.log(mean_probs + eps) + (1 - mean_probs) * np.log(1 - mean_probs + eps))
            epistemic = np.var(all_probs) # epistemic ~ variance across preds
            if mean_probs > mean_prob:
                indicies.append(j)
                mean_values.append(mean_probs)
                aleatoric_values.append(aleatoric)
                epistemic_values.append(epistemic)
        means.append(mean_values)
        aleatorics.append(aleatoric_values) 
        epistemics.append(epistemic_values)
        residues.append('|'.join([str(s) for s in indicies]))
    return means, aleatorics, epistemics, residues



def combine_squidly_blast(query_df, squidly_df, blast_df):
    # Take from squidly and BLAST
    if len(squidly_df) > 0:
        squidly_dict = dict(zip(squidly_df.label, squidly_df.Squidly_CR_Position))
    else:
        squidly_dict = {}
    if len(blast_df) > 0:
        blast_dict = dict(zip(blast_df.From, blast_df.BLAST_residues))
    else:
        blast_dict = {}
    rows = []
    for seq_id in query_df['id'].values:
        if blast_dict.get(seq_id):
            rows.append([seq_id, blast_dict.get(seq_id), 'BLAST'])
        elif squidly_dict.get(seq_id):
            rows.append([seq_id, squidly_dict.get(seq_id), 'squidly'])
        else:
            rows.append([seq_id, None, 'Not-found'])
    return pd.DataFrame(rows, columns=['id', 'residues', 'tool'])

@app.command()
def install():
    """
    Install the models for the package.
    """
    u.dp(['Installing models... '])
    u.dp(['If this fails please see the github and follow the installation instructions.'])

    pckage_dir = dirname(__file__)
    os.system(f'{pckage_dir}/./install.sh')
    os.system(f'python {pckage_dir}/download_models_hf.py')
        
@app.command()
def run(fasta_file: Annotated[str, typer.Argument(help="Full path to query fasta (note have simple IDs otherwise we'll remove all funky characters.)")],
        esm2_model: Annotated[str, typer.Argument(help="Name of the esm2_model, esm2_t36_3B_UR50D or esm2_t48_15B_UR50D")], 
        output_folder: Annotated[str, typer.Argument(help="Where to store results (full path!)")] = 'Current Directory', 
        run_name: Annotated[str, typer.Argument(help="Name of the run")] = 'squidly', 
        ensemble: Annotated[bool, typer.Option(help="Whether or not to do the ensemble.")] = True,
        model_folder: Annotated[str, typer.Option(help="Full path to the model folder.")] = '',
        database:  Annotated[str, typer.Option(help="Full path to database csv (if you want to do the ensemble), needs 3 columns: 'Entry', 'Sequence', 'Residue' where residue is a | separated list of residues. See default DB provided by Squidly.")] = 'None',
        cr_model_as: Annotated[str, typer.Option(help="Optional: Model for the catalytic residue prediction i.e. not using the default with the package. Ensure it matches the esmmodel.")] = '', 
        lstm_model_as: Annotated[str, typer.Option(help="Optional: LSTM model path for the catalytic residue prediction i.e. not using the default with the package. Ensure it matches the esmmodel.")] = '', 
        toks_per_batch: Annotated[int, typer.Option(help="Run method (filter or complete) i.e. filter = only annotates with the next tool those that couldn't be found.")] = 5, 
        as_threshold: Annotated[float, typer.Option(help="Whether or not to keep multiple predicted values if False only the top result is retained.")] = 0.99,
        blast_threshold: Annotated[float, typer.Option(help="Sequence identity with which to use Squidly over BLAST defualt 0.3 (meaning for seqs with < 0.3 identity in the DB use Squidly).")] = 0.3,
        chunk: Annotated[int, typer.Option(help="Max chunk size for the dataset.")] = 0, 
        mean_prob: Annotated[float, typer.Option(help="Mean probability threshold for the dataset.")] = 0.8):

    """ 
    Find catalytic residues using Squidly and BLAST.
    """
    u.dp(['Starting squidly... '])
    model_folder = model_folder if model_folder != '' else os.path.join(dirname(__file__), 'models')
    pckage_dir = dirname(__file__)
    # Other parsing
    if esm2_model not in ['esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D']:
        u.err_p(['ERROR: your ESM model must be one of', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D']) 
        return
    if esm2_model == 'esm2_t36_3B_UR50D':
        esm2_model_dir = f'3B'
    elif esm2_model == 'esm2_t48_15B_UR50D':
        esm2_model_dir = f'15B'
    if ensemble and not os.path.exists(model_folder):
        u.err_p(['ERROR: The model folder does not exist:', model_folder, ". You might need to download it from huggingface. Ensure it is placed in the correct location."])
        return
    output_folder = output_folder if output_folder != 'Current Directory' else os.getcwd()
    query_rows = []
    # Clean fasta file
    id_to_new_id = {}
    with open(os.path.join(output_folder, f'{run_name}_input_fasta.fasta'), 'w+') as fout:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        done_records = []
        # Remove all the ids
        for record in tqdm(records):
            new_id = re.sub('[^0-9a-zA-Z]+', '', record.id)
            id_to_new_id[record.id] = new_id
            if new_id not in done_records:
                query_rows.append([new_id, record.seq])
                fout.write(f">{new_id}\n{record.seq}\n")
                done_records.append(new_id)
            else:
                u.warn_p(['Had a duplicate record! Only keeping the first entry, duplicate ID:', record.id])
                
    blast_df = pd.DataFrame([], columns=['Entry', 'Residues'])
    query_df = pd.DataFrame(query_rows, columns=['id', 'seq'])  
    # Also drop duplicates if there are any 
    squidly_df = pd.DataFrame()  
    if database != 'None': # 
        u.warn_p(["Running BLAST on the following DB: ", database])
        
        # First run BLAST and then we'll run Squidly on the ones that were not able to be annotated
        database_df = pd.read_csv(database)
        if 'Entry' not in database_df.columns or 'Sequence' not in database_df.columns or 'Residue' not in database_df.columns:
            u.err_p(['You need the following columns in your database file csv (Entry, Sequence, Residue)'])
            return 
        
        # Run blast 
        blast_df = run_blast(query_df, database_df, output_folder, run_name, id_col='id', seq_col='seq')
        blast_df['id'] = blast_df['From'].map(id_to_new_id)
        # Now filter to use squidly on those that weren't identified
        entries_found = []
        for entry, seq_identity, residue in blast_df[['From', 'sequence identity', 'BLAST_residues']].values:
            if seq_identity > blast_threshold and isinstance(residue, str) and len(residue) > 0:
                entries_found.append(entry)
        # Now we can filter the query DF.
        # Re-create
        print(set(entries_found))
        query_df = pd.DataFrame(query_rows, columns=['id', 'seq'])    
        remaining_df = query_df[~query_df['id'].isin(entries_found)]

        # Now resave as as fasta file
        with open(os.path.join(output_folder, f'{run_name}_input_fasta.fasta'), 'w+') as fout:
            records = list(SeqIO.parse(fasta_file, "fasta"))
            for seq_id, seq in remaining_df[['id', 'seq']].values:
                fout.write(f">{seq_id}\n{seq}\n")
        fasta_file = os.path.join(output_folder, f'{run_name}_input_fasta.fasta')
        # Now run squidly 
        u.warn_p(["Running Squidly on the following number of seqs: ", len(remaining_df)])
        if len(remaining_df) < 1:
            u.warn_p(['All sequences had a residue found with BLAST. Saving and returning.\n', 
                      'Data saved to:', os.path.join(output_folder, f'{run_name}_blast.csv')])
            blast_df.to_csv(os.path.join(output_folder, f'{run_name}_blast.csv'), index=False)

            return
    elif cr_model_as != '' and lstm_model_as != '':
        u.warn_p(["Running with user supplied squidly models:  ", cr_model_as, lstm_model_as])
    else:
        if esm2_model == '3B':
            lstm_model_as = os.path.join(model_folder, 'Squidly_LSTM_3B.pth')
            cr_model_as = os.path.join(model_folder, 'Squidly_CL_3B.pt')
        elif esm2_model == '15B':
            lstm_model_as = os.path.join(model_folder, 'Squidly_LSTM_15B.pth')
            cr_model_as = os.path.join(model_folder, 'Squidly_CL_15B.pt')
    if ensemble:
        u.warn_p(["Running ensemble"])
        print(os.path.join(model_folder, f'CataloDB_{esm2_model_dir}_CR_1.pt'))
        print(model_folder)
        print(f'{model_folder}CataloDB_{esm2_model_dir}_CR_1.pt')
        models = [
            [os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_CR_1.pt'), os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_LSTM_1.pth')],
            [os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_CR_2.pt'), os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_LSTM_2.pth')],
            [os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_CR_3.pt'), os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_LSTM_3.pth')],
            [os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_CR_4.pt'), os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_LSTM_4.pth')],
            [os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_CR_5.pt'), os.path.join(model_folder, esm2_model_dir, f'CataloDB_{esm2_model}_LSTM_5.pth')]]
    else:
        models = [[cr_model_as, lstm_model_as]]
        u.warn_p(["Running single model"])
    squidly_ensemble = pd.DataFrame()
    for model_i, model in enumerate(models):
        cr_model_as, lstm_model_as = model
        if chunk != 0:
            u.dp(["Chunking"])
            output_filenames = []
            df_list = []
            prev_chunk = 0
            for i in range(chunk, len(query_df) + chunk, chunk):
                df_end = i
                if df_end > len(query_df):
                    df_end = len(query_df) - 1
                tmp_df = query_df.iloc[prev_chunk:df_end]
                df_list.append(tmp_df)
                print(len(tmp_df))
                prev_chunk = i 
            for i, df_chunk in tqdm(enumerate(df_list)):
                chunk_fasta = os.path.join(output_folder, f'{run_name}_{i}_input_fasta.fasta')
                with open(chunk_fasta, 'w+') as fout:
                    for seq_id, seq in df_chunk[['id', 'seq']].values:  # type: ignore 
                        fout.write(f">{seq_id}\n{seq}\n")
                cmd = ['python', os.path.join(pckage_dir, 'squidly.py'), chunk_fasta, esm2_model, cr_model_as, lstm_model_as, output_folder, '--toks_per_batch', 
                str(toks_per_batch), '--AS_threshold',  str(as_threshold)]
                u.warn_p(["Running command:", ' '.join(cmd)])
                subprocess.run(cmd, capture_output=True, text=True)
                input_filename = chunk_fasta.split('/')[-1].split('.')[0]
                output_filenames.append(os.path.join(output_folder, f'{input_filename}_results.pkl'))
            df = pd.DataFrame()
            print(output_filenames)
            for p in output_filenames:
                sub_df = pd.read_pickle(p)
                df = pd.concat([df, sub_df])
            # Save to a consolidated file
            input_filename = fasta_file.split('/')[-1].split('.')[0]
            squidly_df = df
            squidly_df.to_csv(os.path.join(output_folder, f'{input_filename}_squidly_{model_i}.csv'), index=False)
            squidly_ensemble = squidly_ensemble.join(squidly_df, how='outer', rsuffix=f'_{model_i}')
        else:
            cmd = ['python', os.path.join(pckage_dir, 'squidly.py'), fasta_file, esm2_model, cr_model_as, lstm_model_as, output_folder, '--toks_per_batch', 
            str(toks_per_batch), '--AS_threshold',  str(as_threshold)]
            print(cmd)
            u.warn_p(["Running non-batched command:", ' '.join(cmd)])
            subprocess.run(cmd, capture_output=True, text=True)       
            # Now combine the two and save all to the output folder
            # get the input filename 
            input_filename = fasta_file.split('/')[-1].split('.')[0]
        
            squidly_df = pd.read_pickle(os.path.join(output_folder, f'{input_filename}_results.pkl'))
            squidly_df.to_pickle(os.path.join(output_folder, f'{input_filename}_squidly_{model_i}.pkl'))

            squidly_ensemble = squidly_ensemble.join(squidly_df, how='outer', rsuffix=f'_{model_i}')

    means, aleatorics, epistemics, residues = compute_uncertainties(squidly_ensemble, ['all_AS_probs', 'all_AS_probs_1', 'all_AS_probs_2', 'all_AS_probs_3', 'all_AS_probs_4'], mean_prob)
    squidly_ensemble['mean'] = means
    squidly_ensemble['aleatoric'] = aleatorics
    squidly_ensemble['variance'] = epistemics

    squidly_ensemble['Squidly_Ensemble_Residues'] = residues
    squidly_ensemble.set_index('label', inplace=True)
    
    ensemble = combine_squidly_blast(query_df, squidly_df, blast_df)
    blast_df.to_csv(os.path.join(output_folder, f'{run_name}_blast.csv'), index=False)
    squidly_ensemble.to_pickle(os.path.join(output_folder, f'{run_name}_squidly.pkl'))
    ensemble.to_csv(os.path.join(output_folder, f'{run_name}_ensemble.csv'), index=False)

    
if __name__ == "__main__":
    app()
    
# Example command
# squidly AEGAN_with_active_site_seqs_NN.fasta esm2_t36_3B_UR50D
# python __main__.py '../tests/AEGAN_with_active_site_seqs_NN.fasta' esm2_t36_3B_UR50D tmp/ --database ../data/reviewed_sprot_08042025.csv
# python __main__.py '../tests/AEGAN_with_active_site_seqs_NN.fasta' esm2_t36_3B_UR50D tmp/ --chunk 1
# nohup python __main__.py /disk1/ariane/vscode/squidly/data/CARE/fastas/train_swissprot.fasta esm2_t36_3B_UR50D CARE/ train_swissprot --chunk 1000 & 
# squidly fastas/30_protein_test.fasta esm2_t36_3B_UR50D output/ --chunk 1000

# nohup squidly /disk1/ariane/vscode/squidly/data/CARE/fastas/train_swissprot.fasta esm2_t36_3B_UR50D output/ train_swissprot --database reviewed_sprot_08042025.csv --blast-threshold 30 & 
# squidly cataloDB_test.fasta esm2_t36_3B_UR50D output/ cataloDB_3B --database swissprot_with_active_site_seqs_SquidlyBenchmark.csv --blast-threshold 30 --as-threshold 0.97 --lstm-model-as CataloDB_final_models/Squidly_LSTM_3B.pth --cr-model-as CataloDB_final_models/Squidly_CL_3B.pt
# nohup squidly /disk1/ariane/vscode/squidly/data/CARE/fastas/protein_train.fasta esm2_t36_3B_UR50D /disk1/ariane/vscode/squidly/data/train_output/ protein_train & 
# squidly /disk1/ariane/vscode/squidly/data/CARE/fastas/30_protein_test.fasta esm2_t36_3B_UR50D /disk1/ariane/vscode/squidly/data/train_output/ 30_protein_test
# squidly /disk1/ariane/vscode/squidly/data/CARE/fastas/30_protein_test.fasta esm2_t36_3B_UR50D /disk1/ariane/vscode/squidly/data/train_output/ 30_protein_test_low_thresh --as-threshold 0.5
# nohup squidly  /disk1/ariane/vscode/squidly/data/CARE/fastas/protein_train.fasta esm2_t36_3B_UR50D /disk1/ariane/vscode/squidly/data/train_output/ protein_train_low_thresh --as-threshold 0.5 --chunk 3000 &
# squidly TPP_swissprot_predictions_top_20.fasta esm2_t36_3B_UR50D output/ TPP_swissprot_predictions_top_20 --database /disk1/ariane/vscode/squidly/data/reviewed_sprot_08042025.csv --blast-threshold 30 & 
#  python __main__.py /disk1/ariane/vscode/squidly/helen/TPP_swissprot_predictions_top_20.fasta esm2_t36_3B_UR50D output/ /disk1/ariane/vscode/squidly/helen/TPP_swissprot_predictions_top_20 --database /disk1/ariane/vscode/squidly/data/reviewed_sprot_08042025.csv --blast-threshold 30

# python squidly.py ../manuscript/cataloDB_test.fasta esm2_t36_3B_UR50D ../models/FinalModels/CataloDB_models_3_esm2_t36_3B_UR50D_2025-04-13/Scheme3_16000_2/models/temp_best_model.pt ../models/FinalModels/CataloDB_models_3_esm2_t36_3B_UR50D_2025-04-13/Scheme3_16000_2/LSTM/models/13-04-25_16-48_128_2_0.2_100_best_model.pth output/ --toks_per_batch 5 --AS_threshold 0.9
# squidly input_data/cataloDB_test.fasta esm2_t36_3B_UR50D cataloDB_output/ --mean-prob 0.5

#  squidly run /disk1/ariane/vscode/cec_degrader/to_publish/extremeophiles.fasta esm2_t36_3B_UR50D   --database data/reviewed_sprot_08042025.csv