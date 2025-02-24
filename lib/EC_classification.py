from step import Step
import pandas as pd
from tempfile import TemporaryDirectory
import subprocess
from pathlib import Path
import logging
import numpy as np
from tqdm import tqdm 
import torch
import os
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

    
# First run this: nohup python esm-extract.py esm2_t33_650M_UR50D /disk1/ariane/vscode/degradeo/data/DEHP/uniprot/EC3.1.1_training.fasta /disk1/ariane/vscode/degradeo/data/DEHP/uniprot/encodings --include per_tok & 
def extract_active_site_embedding(df, id_column, residue_columns, encoding_dir): 
    """ Expects that the entries for the active site df are saved as the filenames in the encoding dir. """
    combined_tensors = []
    mean_tensors = []
    count_fail = 0
    count_success = 0
    for entry, residues in tqdm(df[[id_column, residue_columns]].values):
        file = Path(encoding_dir + f'{entry}.pt')
        tensors = []
        residues = [int(r) for r in residues.split('|')]
        embedding_file = torch.load(file)
        tensor = embedding_file['representations'][33] # have to get the last layer (36) of the embeddings... very dependant on ESM model used! 36 for medium ESM2
        tensors = []
        mean_tensors.append(np.mean(np.asarray(tensor).astype(np.float32), axis=0))
        for residue in residues:
            t = np.asarray(tensor[residue]).astype(np.float32)
            tensors.append(t)
        combined_tensors.append(tensors)
    # HEre is where you do something on the combined tensors
    df['active_embedding'] = combined_tensors
    df['esm_embedding'] = mean_tensors
    print(count_success, count_fail, count_fail + count_success)
    return df

# First run this: nohup python esm-extract.py esm2_t33_650M_UR50D /disk1/ariane/vscode/degradeo/data/DEHP/uniprot/EC3.1.1_training.fasta /disk1/ariane/vscode/degradeo/data/DEHP/uniprot/encodings --include per_tok & 
def extract_mean_embedding(df, id_column, encoding_dir, rep_num=33): 
    """ Expects that the entries for the active site df are saved as the filenames in the encoding dir. """
    tensors = []
    count_fail = 0
    count_success = 0
    for entry in tqdm(df[id_column].values):
        file = Path(os.path.join(encoding_dir, f'{entry}.pt'))
        embedding_file = torch.load(file)
        tensor = embedding_file['representations'][rep_num] # have to get the last layer (36) of the embeddings... very dependant on ESM model used! 36 for medium ESM2
        t = np.mean(np.asarray(tensor).astype(np.float32), axis=0)
        tensors.append(t)

    df['embedding'] = tensors
    print(count_success, count_fail, count_fail + count_success)
    return df

class EmbedESM(Step):
    
    def __init__(self, id_col: str, seq_col: str, extraction_method='mean', active_site_col: str = None, num_threads=1, tmp_dir: str = None):
        self.seq_col = seq_col
        self.id_col = id_col
        self.active_site_col = active_site_col
        self.num_threads = num_threads or 1
        self.extraction_method = extraction_method
        self.tmp_dir = tmp_dir

    def __execute(self, df: pd.DataFrame, tmp_dir: str) -> pd.DataFrame:
        input_filename = f'{tmp_dir}input.fasta'
        # write fasta file which is the input for proteinfer
        with open(input_filename, 'w+') as fout:
            for entry, seq in df[[self.id_col, self.seq_col]].values:
                fout.write(f'>{entry.strip()}\n{seq.strip()}\n')
        # Might have an issue if the things are not correctly installed in the same dicrectory 
        result = subprocess.run(['python', Path(__file__).parent/'esm-extract.py', 'esm2_t33_650M_UR50D', input_filename, tmp_dir, '--include', 'per_tok'], capture_output=True, text=True)
        if self.extraction_method == 'mean':
            df = extract_mean_embedding(df, self.id_col, tmp_dir)
        elif self.extraction_method == 'active_site':
            if self.active_site_col is None:
                raise ValueError('active_site_col must be provided if extraction_method is active_site')
            df = extract_active_site_embedding(df, self.id_col, self.active_site_col, tmp_dir)
        if result.stderr:
            logger.error(result.stderr)
        logger.info(result.stdout)
        
        return df
    
    def execute(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.tmp_dir is None:
            with TemporaryDirectory() as tmp_dir:
                if self.num_threads > 1:
                    dfs = []
                    df_list = np.array_split(df, self.num_threads)
                    for df_chunk in df_list:
                        dfs.append(self.__execute(df_chunk, tmp_dir))
                    df = pd.DataFrame()
                    for tmp_df in dfs:
                        df = pd.concat([df, tmp_df])
                    return df
                else:
                    df = self.__execute(df, tmp_dir)
                    return df
        else:
            df = self.__execute(df, self.tmp_dir)
            return df