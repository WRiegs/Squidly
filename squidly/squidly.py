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
import tqdm
import pandas as pd
import time
import numpy as np
import psutil
import os
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
        "model_dir",
        type=pathlib.Path,
        help="directory containing the models for the ensemble",
    )
    parser.add_argument(
        "--mean_prob",
        type=float,
        default=0.6,
        help="Mean probability threshold for identifying active site residues in the ensemble",
    )
    parser.add_argument(
        "--mean_var",
        type=float,
        default=0.225,
        help="Variance threshold for identifying active site residues in the ensemble",
    )
    parser.add_argument(
        "output_dir",
        type=pathlib.Path,
        help="output directory for extracted representations",
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


def compute_uncertainties(df, prob_columns, mean_prob=0.5, mean_var=1):
    means, variances, residues, entropy_values  = [], [], [], []
    for p1, p2, p3, p4, p5 in df[prob_columns].values:
        mean_values = []
        variance_values = []
        entropys = []
        indicies = []
        for j in range(0, len(p1)):
            try:
                if j > len(p1): # only go to 1024 - a limitation atm
                    mean_probs = 0
                    entropy = 1
                    vars = 1 # Highlight these are incorrect
                else:
                    eps = 1e-8 # For non-zeros
                    all_probs = [p1[j] + eps, p2[j] + eps, p3[j] + eps, p4[j] + eps, p5[j] + eps]
                    mean_probs = np.mean(all_probs)
                    entropy = -((mean_probs * np.log2(mean_probs)) + ((1 - mean_probs) * np.log2(1 - mean_probs)))
                    vars = np.var(all_probs) # use variance as a proxy
                    if mean_probs > mean_prob and vars < mean_var: # Use the supplied cutoffs
                        indicies.append(j)
                mean_values.append(mean_probs)
                variance_values.append(vars)
                entropys.append(entropy)
            except:
                mean_values.append(0)
                variance_values.append(1)
                entropys.append(1)
        means.append(mean_values)
        variances.append(variance_values)
        entropy_values.append(entropys)
        residues.append('|'.join([str(s) for s in indicies]))
    return means, entropy_values, variances, residues


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
    model = model.to(device)

    # AS_models
    AS_model_paths = list(args.model_dir.glob("*.pt"))
    AS_model_paths = sorted(AS_model_paths, key=lambda x: int(x.stem.split('_')[-1])) # models must be named model_1.pt etc
    LSTM_model_paths = list(args.model_dir.glob("*.pth"))
    LSTM_model_paths = sorted(LSTM_model_paths, key=lambda x: int(x.stem.split('_')[-1]))
    if len(AS_model_paths) != len(LSTM_model_paths):
        raise ValueError("Number of AS models and LSTM models in model dir must be the same")

    # now load each model
    AS_models = []
    LSTM_models = []
    ensemble_outputs = {}
    count = 0
    for as_model_path, lstm_model_path in zip(AS_model_paths, LSTM_model_paths):
        count += 1
        model_number = count
        AS_model = ContrastiveModel(input_dim=embedding_size, output_dim=128)
        AS_model.load_state_dict(torch.load(as_model_path))
        AS_model.eval().to(device)
        LSTM_model = ProteinLSTM(embedding_dim=128, hidden_dim=128, output_dim=1, num_layers=2, dropout_rate=0.1)
        LSTM_model.load_state_dict(torch.load(lstm_model_path))
        LSTM_model.eval().to(device)
        AS_models.append(AS_model)
        LSTM_models.append(LSTM_model)
        ensemble_outputs[f'model_{model_number}'] = {
            "labels": [],
            "all_AS_probs_list": [],
        }
    print(f"Loaded {len(ensemble_outputs.keys())} models for the ensemble")
    dataset = FastaBatchedDataset.from_file(args.file)
    batches = dataset.get_batch_indices(10, extra_toks_per_seq=1) # set tokens per batch to be 10 so that we default to 1 sequence per batch for 1024 len
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=alphabet.get_batch_converter(), batch_sampler=batches
    )
    print(f"Read {args.file} with {len(dataset)} sequences")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    #https://www.youtube.com/shorts/8k5RlykEw3w
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in tqdm.tqdm(enumerate(data_loader),
                                                         total=len(data_loader),
                                                         desc="Squidly Ensemble Inference",
                                                         bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"):
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

                streams = [torch.cuda.Stream() for _ in range(len(AS_models))]
                futures = []
                def run_in_stream(stream, model_name, AS_contrastive_model, AS_model):
                    with torch.cuda.stream(stream):
                        AS_contrastive_model.eval()
                        AS_model.eval()
                        dataloader = torch.utils.data.DataLoader(seq, batch_size=len(seq), shuffle=False)
                        AS_contrastive_rep = []
                        for batch in dataloader:
                            batch = batch.to(device)
                            AS_contrastive_rep.extend(AS_contrastive_model(batch))
                            del batch
                        AS_contrastive_rep = torch.stack(AS_contrastive_rep)
                        seq_len = AS_contrastive_rep.size(0)
                        AS_contrastive_rep = manual_pad_sequence_tensors([AS_contrastive_rep], 1024)[0]
                        AS_probabilities = AS_model(AS_contrastive_rep)
                        AS_probabilities = torch.sigmoid(AS_probabilities)
                        ensemble_outputs[model_name]["labels"].append(result["label"])
                        ensemble_outputs[model_name]["all_AS_probs_list"].append(AS_probabilities[:seq_len].detach().cpu().numpy())

                for stream, (model_name, AS_contrastive_model, AS_model) in zip(
                    streams, zip(ensemble_outputs.keys(), AS_models, LSTM_models)
                ):
                    t = threading.Thread(
                        target=run_in_stream, args=(stream, model_name, AS_contrastive_model, AS_model)
                    )
                    futures.append(t)

                for f in futures:
                    f.start()
                for f in futures:
                    f.join()
                torch.cuda.synchronize()
    ensemble_results = {}
    for model_name, model_outputs in ensemble_outputs.items():
        results = []
        for label, all_AS_probs in zip(
            model_outputs["labels"],
            model_outputs["all_AS_probs_list"],
        ):
            results.append([label, all_AS_probs])
        columns = ["label", "all_AS_probs"]
        model_df = pd.DataFrame(results, columns=columns)

        if csv is not None:
            model_df = pd.merge(df, model_df, how='left', left_on='Entry', right_on='label')

        #output_path = args.output_dir / f"{args.file.stem}_{model_name}_results.pkl"
        ensemble_results[model_name] = model_df
        # model_df.to_pickle(output_path)
    # get the ensemble uncertainties and results
    squidly_df = pd.DataFrame()
    for model_i, (model_name, model_df) in enumerate(ensemble_results.items()):
        model_df = model_df.rename(columns={"all_AS_probs": f"all_AS_probs_{model_i+1}"})
        if squidly_df.empty:
            squidly_df = model_df
        else:
            squidly_df = squidly_df.merge(model_df[['label', f'all_AS_probs_{model_i+1}']], on='label', how='outer')

    prob_cols = [f'all_AS_probs_{i+1}' for i in range(len(ensemble_results.keys()))]
    squidly_ensemble = squidly_df
    means, entropy_values, epistemics, residues = compute_uncertainties(squidly_ensemble, prob_cols, args.mean_prob, args.mean_var)
    squidly_ensemble['mean'] = means
    squidly_ensemble['entropy'] = entropy_values
    squidly_ensemble['variance'] = epistemics
    squidly_ensemble['Squidly_Ensemble_Residues'] = residues
    squidly_ensemble.set_index('label', inplace=True)
    squidly_ensemble.to_pickle(os.path.join(args.output_dir, f'{args.file.stem}_squidly_ensemble.pkl'))
    squidly_ensemble.to_csv(args.output_dir / f"{args.file.stem}_squidly_ensemble.csv")
    squidly_ensemble[[c for c in squidly_ensemble if c not in ['all_AS_probs_5', 'all_AS_probs_1', 'all_AS_probs_2', 'all_AS_probs_3', 'all_AS_probs_4', 'mean', 'entropy', 'variance']]].to_csv(args.output_dir / f"{args.file.stem}_squidly_ensemble_predictions_only.csv")

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
