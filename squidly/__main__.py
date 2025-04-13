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
from squidly.blast import *

u = SciUtil()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


app = typer.Typer()

def combine_squidly_blast(query_df, squidly_df, blast_df):
    # Take from squidly and BLAST
    squidly_dict = dict(zip(squidly_df.label, squidly_df.Squidly_CR_Position))
    blast_dict = dict(zip(blast_df.From, blast_df.Residues))
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
def run(fasta_file: Annotated[str, typer.Argument(help="Full path to query fasta (note have simple IDs otherwise we'll remove all funky characters.)")],
        esm2_model: Annotated[str, typer.Argument(help="Name of the esm2_model, esm2_t36_3B_UR50D or esm2_t48_15B_UR50D")], 
        output_folder: Annotated[str, typer.Argument(help="Where to store results (full path!)")] = 'Current Directory', 
        run_name: Annotated[str, typer.Argument(help="Name of the run")] = 'squidly', 
        database:  Annotated[str, typer.Option(help="Full path to database csv (if you want to do the ensemble), needs 3 columns: 'Entry', 'Sequence', 'Residue' where residue is a | separated list of residues. See default DB provided by Squidly.")] = 'None',
        cr_model_as: Annotated[str, typer.Option(help="Optional: Model for the catalytic residue prediction i.e. not using the default with the package. Ensure it matches the esmmodel.")] = '', 
        lstm_model_as: Annotated[str, typer.Option(help="Optional: LSTM model path for the catalytic residue prediction i.e. not using the default with the package. Ensure it matches the esmmodel.")] = '', 
        toks_per_batch: Annotated[int, typer.Option(help="Run method (filter or complete) i.e. filter = only annotates with the next tool those that couldn't be found.")] = 5, 
        as_threshold: Annotated[float, typer.Option(help="Whether or not to keep multiple predicted values if False only the top result is retained.")] = 0.99,
        blast_threshold: Annotated[float, typer.Option(help="Sequence identity with which to use Squidly over BLAST defualt 0.3 (meaning for seqs with < 0.3 identity in the DB use Squidly).")] = 0.3
        ):

    """ 
    Find catalytic residues using Squidly and BLAST.
    """
    pckage_dir = dirname(__file__)
    model_folder = os.path.join(dirname(__file__), 'models')
    output_folder = output_folder if output_folder != 'Current Directory' else os.getcwd()
    query_rows = []
    # Clean fasta file
    with open(os.path.join(output_folder, f'{run_name}_input_fasta.fasta'), 'w+') as fout:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        done_records = []
        # Remove all the ids
        for record in records:
            new_id = re.sub('[^0-9a-zA-Z]+', '', record.id)
            if new_id not in done_records:
                query_rows.append([new_id, record.seq])
                fout.write(f">{new_id}\n{record.seq}\n")
            else:
                u.warn_p(['Had a duplicate record! Only keeping the first entry, duplicate ID:', record.id])
                
    blast_df = pd.DataFrame([], columns=['Entry', 'Residues'])
    query_df = pd.DataFrame(query_rows, columns=['id', 'seq'])    
    if database != 'None': # 
        u.warn_p(["Running BLAST on the following DB: ", database])
        
        # First run BLAST and then we'll run Squidly on the ones that were not able to be annotated
        database_df = pd.read_csv(database)
        if 'Entry' not in database_df.columns or 'Sequence' not in database_df.columns or 'Residue' not in database_df.columns:
            u.err_p(['You need the following columns in your database file csv (Entry, Sequence, Residue)'])
            return 
        
        # Run blast 
        blast_df = run_blast(query_df, database_df, output_folder, run_name, id_col='id', seq_col='seq')
        # Now filter to use squidly on those that weren't identified
        entries_found = []
        for entry, seq_identity, residue in blast_df[['From', 'sequence identity', 'BLAST_residues']].values:
            if seq_identity > blast_threshold and isinstance(residue, str) and len(residue) > 0:
                entries_found.append(entry)
        # Now we can filter the query DF.
        # Re-create
        query_df = pd.DataFrame(query_rows, columns=['id', 'seq'])    
        remaining_df = query_df[~query_df['id'].isin(entries_found)]
        # Now resave as as fasta file
        with open(os.path.join(output_folder, f'{run_name}_input_fasta.fasta'), 'w+') as fout:
            records = list(SeqIO.parse(fasta_file, "fasta"))
            for seq_id, seq in remaining_df[['id', 'seq']].values:
                fout.write(f">{seq_id}\n{seq}\n")
        # Now run squidly 
    
    # Other parsing
    if esm2_model not in ['esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D']:
        u.err_p(['ERROR: your ESM model must be one of', 'esm2_t36_3B_UR50D', 'esm2_t48_15B_UR50D']) 
        return
    elif cr_model_as != '' and lstm_model_as != '':
        cmd = ['python', os.path.join(pckage_dir, 'squidly.py'), fasta_file, esm2_model, cr_model_as, lstm_model_as, output_folder, '--toks_per_batch', 
            toks_per_batch, '--AS_threshold',  as_threshold]
    else:
        if esm2_model == 'esm2_t36_3B_UR50D':
            lstm_model_as = os.path.join(model_folder, 'Squidly_LSTM_3B.pth')
            cr_model_as = os.path.join(model_folder, 'Squidly_CL_3B.pt')
        elif esm2_model == 'esm2_t48_15B_UR50D':
            lstm_model_as = os.path.join(model_folder, 'Squidly_LSTM_15B.pth')
            cr_model_as = os.path.join(model_folder, 'Squidly_CL_15B.pt')
            
        cmd = ['python', os.path.join(pckage_dir, 'squidly.py'), fasta_file, esm2_model, cr_model_as, lstm_model_as, output_folder, '--toks_per_batch', 
            str(toks_per_batch), '--AS_threshold',  str(as_threshold)]
    u.warn_p(["Running command:", ' '.join(cmd)])
    subprocess.run(cmd, capture_output=True, text=True)       
    # Now combine the two and save all to the output folder
    # get the input filename 
    input_filename = fasta_file.split('/')[-1].split('.')[0]
    squidly_df = pd.read_pickle(os.path.join(output_folder, f'{input_filename}_results.pkl'))

    ensemble = combine_squidly_blast(query_df, squidly_df, blast_df)
    blast_df.to_csv(os.path.join(output_folder, f'{run_name}_blast.csv'), index=False)
    squidly_df.to_pickle(os.path.join(output_folder, f'{run_name}_squidly.pkl'))
    ensemble.to_csv(os.path.join(output_folder, f'{run_name}_ensemble.csv'), index=False)

    
if __name__ == "__main__":
    app()
    
# Example command
# squidly AEGAN_with_active_site_seqs_NN.fasta esm2_t36_3B_UR50D
#python __main__.py '../tests/AEGAN_with_active_site_seqs_NN.fasta' esm2_t36_3B_UR50D tmp/ --database ../data/reviewed_sprot_08042025.csv