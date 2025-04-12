# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import pathlib
import torch
import torch.nn as nn
from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained 
import json
import pandas as pd
import time
import numpy as np
import psutil
import threading
from Bio import SeqIO


def monitor_resources():
    """
    Continuously monitor system usage and update max values.
    """
    global max_ram_used, max_gpu_memory_used, stop_monitoring

    while not stop_monitoring:
        # Check RAM usage
        memory_info = psutil.virtual_memory()
        current_ram_used = memory_info.used / (1024**3)  # Convert to GB
        max_ram_used = max(max_ram_used, current_ram_used)

        # Check GPU memory usage
        if torch.cuda.is_available():
            current_gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)  # Convert to GB
            max_gpu_memory_used = max(max_gpu_memory_used, current_gpu_memory_used)

        time.sleep(0.1)  # Check every 100ms (adjust as needed)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument(
        "file",
        type=pathlib.Path,
        help="FASTA or CSV file on which to extract representations. CSV must have column Sequence, with sequences, and Entry with id. Such as is used by Uniprot",
    )
    parser.add_argument(
        "esm2_model",
        type=str,
        help="Model to use for extracting representations - just used model name. E.g. esm2_t36_3B_UR50D or esm2_t48_15B_UR50D",
    )
    parser.add_argument(
        "AS_contrastive_model",
        type=pathlib.Path,
        help="contrastive model to use for generating contrastive representations"
    )
    parser.add_argument(
        "AS_model",
        type=pathlib.Path,
        help="Model to use for predicting active sites"
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
    )
    parser.add_argument(
        "--toks_per_batch", 
        type=int, 
        default=10, 
        help="maximum batch size"
    )
    parser.add_argument(
        "--AS_threshold",
        type = float,
        default=0.9,
        help="Threshold for active site predict6on"
    )
    parser.add_argument(
        "--logits",
        action="store_true",
        default=False,
        help="Whether to output the logits for the active and binding site predictions"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=False,
        help="Monitor system resources during script execution"
    )
    return parser


def log_system_usage(start_time, log_file="system_usage.log"):
    """
    Logs RAM, GPU memory usage, and elapsed time to a file.
    """
    # Get elapsed time
    elapsed_time = time.time() - start_time

    # Get CPU and RAM usage
    memory_info = psutil.virtual_memory()
    ram_used = memory_info.used / (1024**3)  # Convert to GB
    ram_total = memory_info.total / (1024**3)  # Convert to GB

    # Get GPU memory usage
    if torch.cuda.is_available():
        gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)  # Convert to GB
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    else:
        gpu_memory_used = 0
        gpu_memory_total = 0

    # Log the stats
    with open(log_file, "a") as f:
        f.write(
            f"Elapsed Time: {elapsed_time:.2f} s, "
            f"RAM Used: {ram_used:.2f} GB / {ram_total:.2f} GB, "
            f"GPU Memory Used: {gpu_memory_used:.2f} GB / {gpu_memory_total:.2f} GB\n"
        )


class ContrastiveModel(nn.Module):
    def __init__(self, input_dim=5120, output_dim=128, dropout_prob=0.1):
        super(ContrastiveModel, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(input_dim, int(input_dim/2))
        self.fc2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.fc3 = nn.Linear(int(input_dim/4),output_dim)
        
    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class ProteinLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(ProteinLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes protein embeddings as inputs and outputs hidden states with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

        # The linear layer that maps from hidden state space to the output space
        self.hidden2out = nn.Linear(hidden_dim*2, output_dim)
        
        self.best_model_path = ""
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.hidden2out(lstm_out)
        return output
        

def manual_pad_sequence_tensors(tensors, target_length, padding_value=0):
    """
    Manually pads a list of 2-dimensional tensors along the first dimension to the specified target length.

    Args:
    - tensors (list of Tensors): List of input tensors to pad.
    - target_length (int): Target length to pad/trim the tensors along the first dimension.
    - padding_value (scalar, optional): Value for padding, default is 0.

    Returns:
    - padded_tensors (list of Tensors): List of padded tensors.
    """
    padded_tensors = []
    for tensor in tensors:
        # Check if padding is needed along the first dimension
        if tensor.size(0) < target_length:
            pad_size = target_length - tensor.size(0)
            # Create a padding tensor with the specified value
            padding_tensor = torch.full((pad_size, tensor.size(1)), padding_value, dtype=tensor.dtype, device=tensor.device)
            # Concatenate the padding tensor to the original tensor along the first dimension
            padded_tensor = torch.cat([tensor, padding_tensor])
        # If the tensor is longer than the target length, trim it along the first dimension
        else:
            padded_tensor = tensor[:target_length, :]
        padded_tensors.append(padded_tensor)
    return padded_tensors


def get_fasta_from_df(df, output):
    # remove duplicates from the dataframe at the entry level
    # print how many are above 1024
    print(f"Number of sequences above 1024 length: {len(df[df['seq_len'] > 1024])}")
    df = df[df['seq_len'] <= 1024]
    duplicate_free = df.drop_duplicates(subset='Entry')
    # reset the index
    duplicate_free.reset_index(drop=True, inplace=True)
    seqs = duplicate_free['Sequence']
    entry = duplicate_free['Entry']
    # write the sequences to a fasta file
    with open(output+'.fasta', 'w') as f:
        for i, seq in enumerate(seqs):
            f.write(f'>{entry[i]}\n')
            f.write(f'{seq}\n')
    return pathlib.Path(output+'.fasta')


def main(args):
    if args.esm2_model == "esm2_t48_15B_UR50D":
        args.esm2_model=  "esm2_t48_15B_UR50D" 
        final_layer = 48
        embedding_size = 5120
        model_name = "esm2_t48_15B_UR50D"
    elif args.esm2_model == "esm2_t36_3B_UR50D":
        final_layer = 36
        embedding_size = 2560
        model_name = "esm2_t36_3B_UR50D"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    include = "per_tok"
    
    if args.file.suffix == '.csv':
        csv = args.file
        # filter out any sequences that are longer than 1024 from the csv
        df = pd.read_csv(args.file)
        df['seq_len'] = df['Sequence'].apply(lambda x: len(x))
        args.file = get_fasta_from_df(df, str(args.output_dir / args.file.stem))
    elif args.file.suffix == '.tsv':
        csv = args.file
        # filter out any sequences that are longer than 1024 from the csv
        df = pd.read_csv(args.file, sep='\t')
        df['seq_len'] = df['Sequence'].apply(lambda x: len(x))
        args.file = get_fasta_from_df(df, str(args.output_dir / args.file.stem))
    else: 
        csv = None
    
    model, alphabet = pretrained.load_model_and_alphabet(args.esm2_model)
    model.eval()
    
    # load the contrastive model
    AS_contrastive_model = ContrastiveModel(input_dim=embedding_size, output_dim=128)
    AS_contrastive_model.load_state_dict(torch.load(args.AS_contrastive_model))
    AS_contrastive_model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        AS_contrastive_model = AS_contrastive_model.cuda()
        print("Transferred representation models to GPU")
        
    # Load AS LSTM models
    AS_model = ProteinLSTM(embedding_dim=128, hidden_dim=128, output_dim=1, num_layers=2, dropout_rate=0.1)
    AS_model.load_state_dict(torch.load(args.AS_model))
    AS_model.eval()
    if torch.cuda.is_available():
        AS_model = AS_model.cuda()
        print("Transferred AS models to GPU")

    dataset = FastaBatchedDataset.from_file(args.file)
    batches = dataset.get_batch_indices(args.toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {args.file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        embeddings_labels = []
        predicted_AS_residues = []
        AS_probabilities_list = []
        active_sites_reps = []
        CR_probabilities_predicted = []
        seq_lens = []

        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[final_layer], return_contacts=False)

            representations = {
                layer: t.to(device="cpu") for layer, t in out["representations"].items()
            }
            
            for i, label in enumerate(labels):
                args.output_file = args.output_dir / f"{label}.pt"
                args.output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}

                if "per_tok" in include:
                    result["representations"] = {
                        layer: t[i, 1 : len(strs[i]) + 1].clone()
                        for layer, t in representations.items()
                    }
                active_sites = []
                pos = 0
                
                seq = result["representations"][final_layer]
                
                dataloader = torch.utils.data.DataLoader(seq, batch_size=len(seq), shuffle=False)
                AS_contrastive_rep = []
                # put the dataloader through the model and get the new sequence
                for batch in dataloader:
                    batch = batch.to(device)
                    AS_contrastive_rep.extend(AS_contrastive_model(batch))
                    del batch

                # make tensor of the representations
                AS_contrastive_rep = torch.stack(AS_contrastive_rep)
                seq_len = AS_contrastive_rep.size(0)
                seq_lens.append(seq_len)
                AS_contrastive_rep = manual_pad_sequence_tensors([AS_contrastive_rep], 1024)[0]

                # predict using the AS LSTM model
                active_sites = AS_model(AS_contrastive_rep.cuda())
                active_sites = torch.sigmoid(active_sites)
                AS_probabilities = active_sites.cpu()
                as_site_reps = []
                if args.logits:
                    logits = [round(logit.item(), 3) for logit in active_sites.cpu()]
                    AS_probabilities_list.append(logits)
                active_sites = [i for i, x in enumerate(active_sites) if x > args.AS_threshold] 
                as_site_reps = [AS_contrastive_rep[i].cpu() for i in active_sites]
                # find the probabilities of the active_sites
                active_sites_probs = [AS_probabilities[i] for i in active_sites]
                CR_probabilities_predicted.append(active_sites_probs)
                
                predicted_AS_residues.append(active_sites)
                active_sites_reps.append(as_site_reps)
                embeddings_labels.append(result["label"])
    
    if csv is not None:
        results = []
        for label, AS_residues, AS_probs, AS_site_reps in zip(embeddings_labels, predicted_AS_residues, CR_probabilities_predicted, active_sites_reps):        
            # grab the AA from the sequence itself given the AS_residue positions
            seq = df[df['Entry'] == label]['Sequence'].values[0]
            AS_AA = '|'.join([seq[residue] for residue in AS_residues])
            AS_residues = '|'.join([str(residue) for residue in AS_residues])
            # round the AS_probs to 4 decimal places
            # Convert to a numpy array firt and flatten
            AS_probs = np.array(AS_probs).flatten()
            AS_probs = [round(float(prob), 4) for prob in AS_probs]
            print(AS_probs)
            AS_probs = '|'.join([str(prob) for prob in AS_probs])
            AS_site_reps = np.array(AS_site_reps)
            results.append([label, AS_residues, AS_AA, AS_probs, AS_site_reps])

        # save the results as a df, using the list of lists, with no index
        results_df = pd.DataFrame(results, columns=["label", "Squidly_CR_Position", "Squidly_CR_AA", "Squidly_CR_probabilities", "Squidly_CR_representations"], index=None)
        results_df = pd.merge(df, results_df, how='left', left_on='Entry', right_on='label')
        results_df.to_csv(str(args.output_dir / args.file.stem) + '_results.pkl')
        args.file.unlink()  
    else:
        results = []
        for label, AS_residues, AS_probs, AS_site_reps in zip(embeddings_labels, predicted_AS_residues, CR_probabilities_predicted, active_sites_reps):        
            AS_residues = '|'.join([str(residue) for residue in AS_residues])
            AS_probs = np.array(AS_probs).flatten()
            AS_probs = '|'.join([str(prob) for prob in AS_probs])
            AS_site_reps = np.array(AS_site_reps)
            results.append([label, AS_residues, AS_probs, AS_site_reps])

        # save the results as a df, using the list of lists, with no index
        results_df = pd.DataFrame(results, columns=["label", "Squidly_CR_Position", "Squidly_CR_probabilities", "Squidly_CR_representations"], index=None)
        results_df.to_pickle(str(args.output_dir / args.file.stem) + '_results.pkl')
    
    if args.logits:
        # make dict for seq lens
        dict_of_lens = dict(zip(embeddings_labels, seq_lens))
        
        # go through each sequence and shorten the probabilities to be the length of the sequence, to remove the padding output logits
        for i, seq_len in enumerate(seq_lens):
            AS_probabilities_list[i] = AS_probabilities_list[i][:seq_len]
        
        # now make a dictionary where every sublist in the list is a dictionary with the label as key and the probabilities as a list of values
        dict_of_AS_probs = dict(zip(embeddings_labels, AS_probabilities_list))
        
        # save the dictionary as a json file
        with open(str(args.output_dir / 'Squidly_CR_probabilities.json'), 'w') as f:
            json.dump(dict_of_AS_probs, f)
    

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    
    if args.monitor:
        max_ram_used = 0
        max_gpu_memory_used = 0
        start_time = time.time()
        stop_monitoring = False
        # Start the monitoring thread
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
    
    main(args)
    
    if args.monitor:
        # Stop the monitoring thread
        stop_monitoring = True
        monitor_thread.join()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        # Log results
        log_file = args.output_dir / "resource_usage.log"
        with open(log_file, "w") as f:
            f.write("Maximum Resource Usage During Script Execution:\n")
            f.write(f"Elapsed Time: {elapsed_time:.2f} seconds\n")
            f.write(f"Maximum RAM Used: {max_ram_used:.2f} GB\n")
            f.write(f"Maximum GPU Memory Used: {max_gpu_memory_used:.2f} GB\n")