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
      
                    