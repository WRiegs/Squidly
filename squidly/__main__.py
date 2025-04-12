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
from sciutil import SciUtil
import subprocess
import timeit
import logging


u = SciUtil()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
u = SciUtil()


app = typer.Typer()

@app.command()
def run(fasta_file: Annotated[str, typer.Argument(help="Full path to query fasta or csv (note have simple IDs otherwise we'll remove all funky characters.)")],
        esm2_model: Annotated[str, typer.Argument(help="Name of the esm2_model, esm2_t36_3B_UR50D or esm2_t48_15B_UR50D")], 
        output_folder: Annotated[str, typer.Argument(help="Where to store results (full path!)")] = 'Current Directory', 
        run_name: Annotated[str, typer.Argument(help="Name of the run")] = 'squidly', 
        cr_model_as: Annotated[str, typer.Option(help="Optional: Model for the catalytic residue prediction i.e. not using the default with the package. Ensure it matches the esmmodel.")] = '', 
        lstm_model_as: Annotated[str, typer.Option(help="Optional: LSTM model path for the catalytic residue prediction i.e. not using the default with the package. Ensure it matches the esmmodel.")] = '', 
        toks_per_batch: Annotated[int, typer.Option(help="Run method (filter or complete) i.e. filter = only annotates with the next tool those that couldn't be found.")] = 5, 
        as_threshold: Annotated[float, typer.Option(help="Whether or not to keep multiple predicted values if False only the top result is retained.")] = 0.99
        ):

    """ 
    Find catalytic residues using Squidly and BLAST.
    """
    pckage_dir = dirname(__file__)
    model_folder = os.path.join(dirname(__file__), 'models')
    output_folder = output_folder if output_folder != 'Current Directory' else os.getcwd()

    # Clean fasta file
    with open(os.path.join(output_folder, f'{run_name}_input_fasta.fasta'), 'w+') as fout:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        done_records = []
        # Remove all the ids
        for record in records:
            new_id = re.sub('[^0-9a-zA-Z]+', '', record.id)
            if new_id not in done_records:
                fout.write(f">{new_id}\n{record.seq}\n")
            else:
                u.warn_p(['Had a duplicate record! Only keeping the first entry, duplicate ID:', record.id])
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
    u.warn_p(["Running command:", ''.join(cmd)])
    result = subprocess.run(cmd, capture_output=True, text=True)       
    return result

if __name__ == "__main__":
    app()
    
# Example command
# squidly AEGAN_with_active_site_seqs_NN.fasta esm2_t36_3B_UR50D