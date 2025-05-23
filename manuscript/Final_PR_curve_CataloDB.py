import pathlib
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch import nn
import numpy as np
import glob
import random
from Bio import SeqIO
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from sklearn.metrics import auc
import math
from sklearn.metrics import roc_curve, precision_recall_curve

random.seed(420)

def get_test_results(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    model = model.cuda()
    with torch.no_grad():  # Disables gradient calculation
        for inputs, labels, lengths in test_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            predicted = torch.sigmoid(outputs)
            predicted = predicted.squeeze().tolist()
            labels = labels.tolist()
            # resize labels and predicted to original sequence length (remove right padding)
            labels = [label[:length] for label, length in zip(labels, lengths)]
            predicted = [pred[:length] for pred, length in zip(predicted, lengths)]

            predictions.extend(predicted)
            actuals.extend(labels)
            
    # need to unpad the results, so that they are the same length as the original sequences
    # get the lengths of the original sequences


    actuals = [item for sublist in actuals for item in sublist]
    predictions = [item for sublist in predictions for item in sublist]
    return actuals, predictions

def get_evaluation_metrics(actuals, predictions):
    # Calculate evaluation metrics
    TP = sum([1 for i in range(len(actuals)) if actuals[i] == 1 and predictions[i] == 1])
    TN = sum([1 for i in range(len(actuals)) if actuals[i] == 0 and predictions[i] == 0])
    FP = sum([1 for i in range(len(actuals)) if actuals[i] == 0 and predictions[i] == 1])
    FN = sum([1 for i in range(len(actuals)) if actuals[i] == 1 and predictions[i] == 0])

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    
    
    # get MCC
    # MCC is the Matthews correlation coefficient
    
    denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = (TP * TN - FP * FN) / math.sqrt(denominator) if denominator != 0 else 0

    return accuracy, precision, recall, f1, MCC

def get_threshold_results(actuals, predictions):
    best_threshold = 0
    best_f1 = 0
    precisions = []
    recalls = []
    accuracies = []
    for threshold in np.arange(0.1, 99.9, 0.1):
        threshold = threshold / 100
        predictions_binary = [1 if prediction > threshold else 0 for prediction in predictions]
        accuracy, precision, recall, f1, MCC = get_evaluation_metrics(actuals, predictions_binary)
        precisions.append(precision)
        recalls.append(recall)
        accuracies.append(accuracy)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1, precisions, recalls, accuracies

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

def process_data(df, emb_dir, max_len=1024):
    # get the residues of each sequence in the dataset using a for loop
    active_sites = []
    
    if 'BS' in df.columns:
        col = 'BS'
        # iterate through the df and get the residues
        for index, row in df.iterrows():
            active_site_string = row[col]
            active_site_string = active_site_string.replace(" ", "")
            active_site_string = active_site_string.replace("[", "")
            active_site_string = active_site_string.replace("]", "")
            active_site_list = active_site_string.split(",")
            active_site_list = [int(site) for site in active_site_list if site != '']
            active_sites.append(active_site_list)
                
        df["residues"] = active_sites
    elif 'Active site' in df.columns:
        col = 'Active site'
        active_sites = []
        # iterate through the df and get the active sites
        for index, row in df.iterrows():
            active_site_string = row["Active site"]
            active_site_list = []
            intermediate_list = active_site_string.split(";")
            #iterate through the intermediate list and get the active sites
            for item in intermediate_list:
                if item.startswith("ACT_SITE") or item.startswith(" ACT_SITE"):
                    active_site_list.append(int(item.split("ACT_SITE ")[1])-1)
            active_sites.append(active_site_list)
        df["residues"] = active_sites
    elif 'AS' in df.columns:
        col = 'AS'
        # iterate through the df and get the residues
        for index, row in df.iterrows():
            active_site_string = row[col]
            active_site_string = active_site_string.replace(" ", "")
            active_site_string = active_site_string.replace("[", "")
            active_site_string = active_site_string.replace("]", "")
            active_site_list = active_site_string.split(",")
            active_site_list = [int(site) for site in active_site_list if site != '']
            active_sites.append(active_site_list)
                
        df["residues"] = active_sites
    else:
        raise ValueError("No column with active sites or BS found. processing the data likely failed")
    
    # Convert the df to a dictionary, with Entry as its key
    dict_df = df.set_index("Entry").T.to_dict()
    
    # Now get all the embedding files
    files = []

    for file in glob.glob(emb_dir + "/**/*.pt", recursive=True):
        files.append(file)   

    embeddings_tensors = []
    embeddings_labels = []
    for file in files:
        tensor = torch.load(file)
        label = file.split("/")[-1].split("|")[1]
        embeddings_tensors.append(tensor)
        embeddings_labels.append(label)

    paired_active_sites = []
    paired_labels = []
    for label in embeddings_labels:
        try:
            paired_active_sites.append(dict_df[label]["residues"])
        except:
            print(f"Label {label} not found in the metadata")
        paired_labels.append(label)
    
    # Reorder the metadata df to match the order of the embeddings paired_labels
    # reorder them by the paired_labels
    df_paired = df.set_index("Entry")
    df_paired = df_paired.reindex(paired_labels)
    df_paired = df_paired.reset_index()
    
    tensors = embeddings_tensors
    
    # Get a list of all the lengths of the tensors
    lengths = [tensor.shape[0] for tensor in tensors]
    #print("Lengths of the tensors:", lengths)
    active_sites = paired_active_sites
    meta_data = df_paired


    # NOTE: the max length should always be consistent! The model will only accept a specific length
    max_length = max(tensor.shape[0] for tensor in tensors)
    
    if max_length != max_len:
        # override and manually pad them
        max_length = max_len
        padded_tensor = manual_pad_sequence_tensors(tensors, max_length)
        padded_tensor = torch.stack(padded_tensor)
    else:
        print("Max length of the input tensors:", max_length)
        padded_tensor = pad_sequence(tensors, batch_first=True, padding_value=0.0)
        #padded_tensor = torch.stack(padded_tensor)

    print("Shape of the padded input tensor:", padded_tensor.shape)  # (number of samples, max_length, embedding dimension)

    max_length = padded_tensor.shape[1]  # Maximum sequence length from the padded input tensor
    num_sequences = len(active_sites)

    # Initialize the target tensor with zeros
    targets = torch.zeros(num_sequences, max_length, dtype=torch.long)

    # Populate the target tensor
    for i, sites in enumerate(active_sites):
        #convert the strings in the list to integers
        sites = [int(site) for site in sites] # take -1 from the site to get the correct index (index from 1 in uniprot)
        targets[i, sites] = 1  # Set active site positions to 1

    print("Shape of the targets tensor:", targets.shape)  # (number of samples, max_length)
    
    
    # Pack the padded_tensor so that it is more space efficient
    #packed_tensor = torch.nn.utils.rnn.pack_padded_sequence(padded_tensor, lengths, batch_first=True, enforce_sorted=False)
    
    # tensors = tuple(tensors)
    return padded_tensor.cpu(), targets.cpu(), df_paired

class CustomDataset(Dataset):
    def __init__(self, embeddings, targets, seq_length):
        self.embeddings = embeddings
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        # Retrieve the data
        embedding = self.embeddings[idx]
        target = self.targets[idx]

        # Convert to tensors if they aren't already
        if not torch.is_tensor(embedding):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        if not torch.is_tensor(target):
            target = torch.tensor(target, dtype=torch.float32)

        # Ensure that requires_grad is False
        embedding = embedding.detach()
        target = target.detach()

        return embedding, target, self.seq_length[idx]

class ProteinLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate):
        super(ProteinLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rate, 
            bidirectional=True
        )
        self.hidden2out = nn.Linear(hidden_dim * 2, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.hidden2out(lstm_out)
        return output

def main():
    output_dir = pathlib.Path(f"PR_Curves_CataloDB")
    
    if not output_dir.exists():
        os.makedirs(output_dir)
        os.makedirs(f"{output_dir}/BIG_embeddings")
        os.makedirs(f"{output_dir}/BIG_squidly")
        os.makedirs(f"{output_dir}/MED_embeddings")
        os.makedirs(f"{output_dir}/MED_squidly")
    
        big = f"{output_dir}/BIG_embeddings"
        big_squidly = f"{output_dir}/BIG_squidly"
        med = f"{output_dir}/MED_embeddings"
        med_squidly = f"{output_dir}/MED_squidly"
        
        # get the fastafile
        fasta = "/scratch/project/squid/code_modular/CataloDB/0.9reduced_set_no_family_specific.fasta"
        benchmark_entries = "/scratch/project/squid/code_modular/CataloDB/final_test_set_post_structural_filtering.txt"
        metadata = "/scratch/project/squid/code_modular/datasets/dataset_2.tsv"
        with open(fasta, "r") as f:
            fasta = list(SeqIO.parse(f, "fasta"))
        with open(benchmark_entries, "r") as f:
            benchmark_entries = f.readlines()
        entries = [entry.strip() for entry in benchmark_entries]
        # get all the seqs that are in the benchmark
        fasta = [seq for seq in fasta if seq.id.split('|')[1] in entries]
        with open(f"{output_dir}/benchmark.fasta", "w") as f:
            SeqIO.write(fasta, f, "fasta")
        metadata = pd.read_csv(metadata, sep="\t")
        metadata = metadata[metadata["Entry"].isin(entries)]
        
        # get big embeddings
        ESM_MODEL = "/scratch/project/squid/models/ESM2/esm2_t48_15B_UR50D.pt"
        extracting_esm = f"python lib/extract_esm2.py {ESM_MODEL} {output_dir}/benchmark.fasta {big} --toks_per_batch 10 --include per_tok"
        os.system(extracting_esm)
        
        # get med embeddings
        ESM_MODEL = "esm2_t36_3B_UR50D"
        extracting_esm = f"python lib/extract_esm2.py {ESM_MODEL} {output_dir}/benchmark.fasta {med} --toks_per_batch 10 --include per_tok"
        os.system(extracting_esm)
        
        # convert to squidly embeddings using CL models
        
        med_CL = "/scratch/project/squid/code_modular/final_models/CataloDB_models/3B/CL.pt"
        med_CL = "/scratch/project/squid/code_modular/CataloDB_3_esm2_t36_3B_UR50D_2025-04-02/Scheme3_16000_3/models/temp_best_model.pt"
        big_CL = "/scratch/project/squid/code_modular/final_models/CataloDB_models/15B/CL.pt"
        big_CL = "/scratch/project/squid/code_modular/CataloDB_3_esm2_t48_15B_UR50D_2025-04-02/Scheme3_16000_8/models/temp_best_model.pt"
        converting_esm2_to_squidly = f"python lib/esm2squidly.py --model_location {med_CL} --embeddings_dir {med} --embedding_size {2560} --out {med_squidly} --layer {36} --save_new_pt"
        os.system(converting_esm2_to_squidly)
        converting_esm2_to_squidly = f"python lib/esm2squidly.py --model_location {big_CL} --embeddings_dir {big} --embedding_size {5120} --out {big_squidly} --layer {48} --save_new_pt"
        os.system(converting_esm2_to_squidly)
    
    med_LSTM = "/scratch/project/squid/code_modular/final_models/CataloDB_models/3B/LSTM.pth"
    med_LSTM = "/scratch/project/squid/code_modular/CataloDB_3_esm2_t36_3B_UR50D_2025-04-02/Scheme3_16000_3/LSTM/models/02-04-25_22-24_128_2_0.2_400_best_model.pth"
    big_LSTM = "/scratch/project/squid/code_modular/final_models/CataloDB_models/15B/LSTM.pth"
    big_LSTM = "/scratch/project/squid/code_modular/CataloDB_3_esm2_t48_15B_UR50D_2025-04-02/Scheme3_16000_8/LSTM/models/03-04-25_02-30_128_2_0.2_400_best_model.pth"
    med_LSTM_model = torch.load(med_LSTM)
    big_LSTM_model = torch.load(big_LSTM)
    # now load the model from the weights dict
    med_LSTM_model = ProteinLSTM(embedding_dim=128, hidden_dim=128, output_dim=1, num_layers=2, dropout_rate=0.1)
    big_LSTM_model = ProteinLSTM(embedding_dim=128, hidden_dim=128, output_dim=1, num_layers=2, dropout_rate=0.1)
    med_LSTM_model.load_state_dict(torch.load(med_LSTM))
    big_LSTM_model.load_state_dict(torch.load(big_LSTM))
    
    # now make a dataloader from the converted embeddings
    big_squidly = f"{output_dir}/BIG_squidly"
    med_squidly = f"{output_dir}/MED_squidly"
    
    # get the fastafile
    benchmark_entries = "/scratch/project/squid/code_modular/CataloDB/final_test_set_post_structural_filtering.txt"
    metadata = "/scratch/project/squid/code_modular/datasets/dataset_2.tsv"
    with open(benchmark_entries, "r") as f:
        benchmark_entries = f.readlines()
    entries = [entry.strip() for entry in benchmark_entries]
    metadata = pd.read_csv(metadata, sep="\t")
    metadata = metadata[metadata["Entry"].isin(entries)]
    
    
    padded_tensor, targets, df_paired = process_data(metadata, big_squidly, max_len=1024)
    # get the lengths of the original sequences from df_paired
    seqs = df_paired["Sequence"]
    seq_lengths = [len(seq) for seq in seqs]
    big_dataset = CustomDataset(padded_tensor, targets, seq_lengths)
    BIG_test_loader = torch.utils.data.DataLoader(big_dataset, batch_size=300, shuffle=False)

    
    padded_tensor, targets, df_paired = process_data(metadata, med_squidly, max_len=1024)
    # get the lengths of the original sequences from df_paired
    seqs = df_paired["Sequence"]
    seq_lengths = [len(seq) for seq in seqs]
    med_dataset = CustomDataset(padded_tensor, targets, seq_lengths)
    MED_test_loader = torch.utils.data.DataLoader(med_dataset, batch_size=300, shuffle=False)
    
    # okay, now we need to get an ROC curve plot with both models included, changing the threshold from 0-100
    # get the predictions from the models
    big_actuals, big_predictions = get_test_results(big_LSTM_model, BIG_test_loader)
    med_actuals, med_predictions = get_test_results(med_LSTM_model, MED_test_loader)

    # get the ROC curve
    big_fpr, big_tpr, big_thresholds = roc_curve(big_actuals, big_predictions)
    med_fpr, med_tpr, med_thresholds = roc_curve(med_actuals, med_predictions)
    big_precision, big_recall, _ = precision_recall_curve(big_actuals, big_predictions)
    med_precisions, med_recalls, _ = precision_recall_curve(med_actuals, med_predictions)
    
    # get the PR AUC
    big_pr_auc = np.trapz(big_recall, big_precision)
    med_pr_auc = np.trapz(med_recalls, med_precisions)
    print(f"3B PR AUC: {med_pr_auc:.2f}")
    print(f"15B PR AUC: {big_pr_auc:.2f}")
    
    Big_training_threshold = 0.99
    Med_training_threshold = 0.99
    
    # get the f1 score for the training threshold
    big_training_predictions = [1 if prediction > Big_training_threshold else 0 for prediction in big_predictions]
    med_training_predictions = [1 if prediction > Med_training_threshold else 0 for prediction in med_predictions]
    
    big_training_accuracy, big_training_precision, big_training_recall, big_training_f1, big_training_MCC = get_evaluation_metrics(big_actuals, big_training_predictions)
    med_training_accuracy, med_training_precision, med_training_recall, med_training_f1, med_training_MCC = get_evaluation_metrics(med_actuals, med_training_predictions)
    
    plt.figure(figsize=(8, 8))
    plt.plot(big_precision, big_recall, label=f"15B (F1 = {big_training_f1:.3f})")
    plt.plot(med_precisions, med_recalls, label=f"3B (F1 = {med_training_f1:.3f})")
    # add the AUC to the plot
    plt.text(0.5, 0.5, f"15B PR AUC: {big_pr_auc:.3f}\n3B PR AUC: {med_pr_auc:.3f}", fontsize=12, ha='center')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Squidly Precision-Recall Curve on CataloDB Benchmark")
    plt.legend()
    plt.grid()
    plt.show()
    # save to svg with text
    plt.savefig(f"{output_dir}/PR_curve.svg", format="svg")
    plt.savefig(f"{output_dir}/PR_curve.png")
    
    
    # Now we want to stratify the test data by the 1st tier EC class it belongs to.
    # get the fastafile
    benchmark_entries = "/scratch/project/squid/code_modular/CataloDB/final_test_set_post_structural_filtering.txt"
    metadata = "/scratch/project/squid/code_modular/datasets/dataset_2.tsv"
    with open(benchmark_entries, "r") as f:
        benchmark_entries = f.readlines()
    entries = [entry.strip() for entry in benchmark_entries]
    metadata = pd.read_csv(metadata, sep="\t")
    metadata = metadata[metadata["Entry"].isin(entries)]
    
    # get the unique EC tier 1s that exist:
    ECs = metadata["EC number"].tolist()
    ECs = [x.split(".")[0] for x in ECs]
    metadata['tier1_EC'] = ECs
    
    # get the unique ECs
    unique_ECs = list(set(ECs))
    print(unique_ECs)
    big_stratified_F1s = []
    big_stratified_AUCs = []
    big_stratified_Precisions = []
    big_stratified_Recalls = []
    big_stratified_MCC = []
    med_stratified_F1s = []
    med_stratified_AUCs = []
    med_stratified_Precisions = []
    med_stratified_Recalls = []
    med_stratified_MCC = []
    big_total_catalytic_residues = []
    med_total_catalytic_residues = []
    med_total_predicted_residues = []
    big_total_predicted_residues = []
    for EC in unique_ECs:
        print()
        print(EC)

        # get the entries that belong to this EC
        entries = metadata[metadata["tier1_EC"] == EC]
        # get the entries that belong to this EC
        entries = entries["Entry"].tolist()
        
        # get the fastafile
        benchmark_entries = "/scratch/project/squid/code_modular/CataloDB/final_test_set_post_structural_filtering.txt"
        with open(benchmark_entries, "r") as f:
            benchmark_entries = f.readlines()
        test_entries = [entry.strip() for entry in benchmark_entries]

        # get the entries which are in the test_entries
        emtries = [entry for entry in test_entries]

        metadata_subset = metadata[metadata["Entry"].isin(entries)]
        print(len(metadata_subset))

        
        # get the squidly embeddings
        big_squidly = f"{output_dir}/BIG_squidly"
        med_squidly = f"{output_dir}/MED_squidly"
        big_EC_dir = f"{output_dir}/BIG_EC_{EC}"
        med_EC_dir = f"{output_dir}/MED_EC_{EC}"
        if not os.path.exists(big_EC_dir):
            os.makedirs(big_EC_dir)
        if not os.path.exists(med_EC_dir):
            os.makedirs(med_EC_dir)
        for file in glob.glob(big_squidly + "/*.pt", recursive=True):
            if any(entry in file for entry in entries):
                os.system(f"cp \"{file}\" {big_EC_dir}/")
        for file in glob.glob(med_squidly + "/*.pt", recursive=True):
            if any(entry in file for entry in entries):
                os.system(f"cp \"{file}\" {med_EC_dir}/")
        

        padded_tensor, targets, df_paired = process_data(metadata_subset, big_EC_dir, max_len=1024)
        # get the lengths of the original sequences from df_paired
        seqs = df_paired["Sequence"]
        seq_lengths = [len(seq) for seq in seqs]
        big_dataset = CustomDataset(padded_tensor, targets, seq_lengths)
        BIG_test_loader = torch.utils.data.DataLoader(big_dataset, batch_size=300, shuffle=False)

        
        padded_tensor, targets, df_paired = process_data(metadata_subset, med_EC_dir, max_len=1024)
        # get the lengths of the original sequences from df_paired
        seqs = df_paired["Sequence"]
        seq_lengths = [len(seq) for seq in seqs]
        med_dataset = CustomDataset(padded_tensor, targets, seq_lengths)
        MED_test_loader = torch.utils.data.DataLoader(med_dataset, batch_size=300, shuffle=False)
        
        big_actuals, big_predictions = get_test_results(big_LSTM_model, BIG_test_loader)
        med_actuals, med_predictions = get_test_results(med_LSTM_model, MED_test_loader)

        big_precision, big_recall, _ = precision_recall_curve(big_actuals, big_predictions)
        med_precisions, med_recalls, _ = precision_recall_curve(med_actuals, med_predictions)

        # get the PR AUC
        big_pr_auc = np.trapz(big_recall, big_precision)
        med_pr_auc = np.trapz(med_recalls, med_precisions)
        print(f"3B PR AUC: {med_pr_auc:.2f}")
        print(f"15B PR AUC: {big_pr_auc:.2f}")
        
        Big_training_threshold = 0.99
        Med_training_threshold = 0.99
        
        # get the f1 score for the training threshold
        big_training_predictions = [1 if prediction > Big_training_threshold else 0 for prediction in big_predictions]
        med_training_predictions = [1 if prediction > Med_training_threshold else 0 for prediction in med_predictions]

        # get the total number of catalytic residues
        big_total_catalytic_residues.append(sum([1 for i in range(len(big_actuals)) if big_actuals[i] == 1]))
        med_total_catalytic_residues.append(sum([1 for i in range(len(med_actuals)) if med_actuals[i] == 1]))
        big_total_predicted_residues.append(sum([1 for i in range(len(big_training_predictions)) if big_training_predictions[i] == 1]))
        med_total_predicted_residues.append(sum([1 for i in range(len(med_training_predictions)) if med_training_predictions[i] == 1]))
        
        big_training_accuracy, big_training_precision, big_training_recall, big_training_f1, big_training_MCC = get_evaluation_metrics(big_actuals, big_training_predictions)
        big_stratified_F1s.append(big_training_f1)
        big_stratified_AUCs.append(big_pr_auc)
        big_stratified_Precisions.append(big_training_precision)
        big_stratified_Recalls.append(big_training_recall)
        big_stratified_MCC.append(big_training_MCC)
        med_training_accuracy, med_training_precision, med_training_recall, med_training_f1, med_training_MCC = get_evaluation_metrics(med_actuals, med_training_predictions)
        med_stratified_F1s.append(med_training_f1)
        med_stratified_AUCs.append(med_pr_auc)
        med_stratified_Precisions.append(med_training_precision)
        med_stratified_Recalls.append(med_training_recall)
        med_stratified_MCC.append(med_training_MCC)

        
    # now make a dataframe with the results
    big_stratified_results = pd.DataFrame({
        "EC": unique_ECs,
        "F1": big_stratified_F1s,
        "AUC": big_stratified_AUCs,
        "Precision": big_stratified_Precisions,
        "Recall": big_stratified_Recalls,
        "MCC": big_stratified_MCC,
        "total_residues": big_total_catalytic_residues,
        "total_predicted_residues": big_total_predicted_residues
    })
    med_stratified_results = pd.DataFrame({
        "EC": unique_ECs,
        "F1": med_stratified_F1s,
        "AUC": med_stratified_AUCs,
        "Precision": med_stratified_Precisions,
        "Recall": med_stratified_Recalls,
        "MCC": med_stratified_MCC,
        "total_residues": med_total_catalytic_residues,
        "total_predicted_residues": med_total_predicted_residues
    })
    # save the results to csv
    big_stratified_results.to_csv(f"{output_dir}/big_stratified_results.csv", index=False)
    
    med_stratified_results.to_csv(f"{output_dir}/med_stratified_results.csv", index=False)

    # reorder the data so the EC labels are in numeric order
    big_stratified_results = big_stratified_results.sort_values(by="total_residues", ascending=False)
    med_stratified_results = med_stratified_results.sort_values(by="total_residues", ascending=False)
    # reset the index
    big_stratified_results = big_stratified_results.reset_index(drop=True)
    med_stratified_results = med_stratified_results.reset_index(drop=True)
    
    # make a bar chart for each set of EC/F1
    plt.figure(figsize=(10, 5))
    # make the bar chart separate bars for big and medium data
    plt.bar(big_stratified_results["EC"], big_stratified_results["F1"], label="15B", alpha=0.5)
    plt.bar(med_stratified_results["EC"], med_stratified_results["F1"], label="3B", alpha=0.5)
    plt.xlabel("EC (tier 1)")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by EC")
    plt.legend()
    plt.xticks(rotation=60)
    # save svg
    plt.savefig(f"{output_dir}/F1_by_EC.svg", format="svg")
    plt.savefig(f"{output_dir}/F1_by_EC.png")

    # now make a bar chart separately for the big and medium which plots a gray 0.3 alpha bar for total residues, and a foreground bar for total predicted residues
    plt.figure(figsize=(10, 5))
    # make the bar chart separate bars for big and medium data
    plt.bar(big_stratified_results["EC"], big_stratified_results['total_residues'], label="Total Catalytic Residues", alpha=0.3)    
    plt.bar(big_stratified_results["EC"], big_stratified_results['total_predicted_residues'], label="Predicted Catalytic Residues", alpha=0.5)
    plt.xlabel("EC (tier 1)")
    plt.ylabel("Number of Catalytic Residues")
    plt.title("Catalytic Residue Recall by EC")
    plt.legend()
    plt.xticks(rotation=60)
    # save svg
    plt.savefig(f"{output_dir}/Catalytic_Residue_Recall_by_EC_big.svg", format="svg")
    plt.savefig(f"{output_dir}/Catalytic_Residue_Recall_by_EC_big.png")


    # now do the same for medium
    plt.figure(figsize=(10, 5))
    # make the bar chart separate bars for big and medium data
    plt.bar(med_stratified_results["EC"], med_stratified_results['total_residues'], label="Total Catalytic Residues", alpha=0.3)
    plt.bar(med_stratified_results["EC"], med_stratified_results['total_predicted_residues'], label="Predicted Catalytic Residues", alpha=0.5)
    plt.xlabel("EC (tier 1)")
    plt.ylabel("Number of Catalytic Residues")
    plt.title("Catalytic Residue Recall by EC")
    plt.legend()
    plt.xticks(rotation=60)
    # save svg
    plt.savefig(f"{output_dir}/Catalytic_Residue_Recall_by_EC_med.svg", format="svg")
    plt.savefig(f"{output_dir}/Catalytic_Residue_Recall_by_EC_med.png")
    
    # now I want to make a combined verion of the above 2 residue number bar charts.
    # each EC should have 2 sets of bars, with total and predicted, and have different colours
    # make a bar chart for each set of EC/F1

    # Ensure consistent EC order
    # Step 1: Merge and align by EC
    ecs = sorted(set(big_stratified_results["EC"]).intersection(set(med_stratified_results["EC"])))

    # Step 2: Extract and organize data
    data = []
    for ec in ecs:
        total = big_stratified_results.loc[big_stratified_results["EC"] == ec, "total_residues"].values[0]
        big_pred = big_stratified_results.loc[big_stratified_results["EC"] == ec, "total_predicted_residues"].values[0]
        med_pred = med_stratified_results.loc[med_stratified_results["EC"] == ec, "total_predicted_residues"].values[0]
        data.append((ec, total, big_pred, med_pred))

    # Step 3: Sort by total residues (descending)
    data.sort(key=lambda x: x[1], reverse=True)

    # Step 4: Unpack sorted data
    ecs_sorted = [d[0] for d in data]
    total_residues = [d[1] for d in data]
    big_predicted = [d[2] for d in data]
    med_predicted = [d[3] for d in data]

    # Step 5: Plot
    x = np.arange(len(ecs_sorted))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))

    # Total (background)
    plt.bar(x, total_residues, width=bar_width * 2, color='gray', alpha=0.3, label="Total Catalytic Residues")

    # Predicted foreground bars
    plt.bar(x - bar_width/2, big_predicted, width=bar_width, color='blue', alpha=0.7, label="15B - Predicted")
    plt.bar(x + bar_width/2, med_predicted, width=bar_width, color='green', alpha=0.7, label="3B - Predicted")

    plt.xticks(x, ecs_sorted, rotation=60)
    plt.xlabel("EC (tier 1)")
    plt.ylabel("Number of Catalytic Residues")
    plt.title("Catalytic Residue Prediction by EC")
    plt.legend()
    plt.tight_layout()

    plt.savefig(f"{output_dir}/Catalytic_Residue_Recall_by_EC_combined_sorted.svg", format="svg")
    plt.savefig(f"{output_dir}/Catalytic_Residue_Recall_by_EC_combined_sorted.png")


        
    
if __name__ == '__main__':
    main()
    












