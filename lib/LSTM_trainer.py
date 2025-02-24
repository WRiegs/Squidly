'''
This script trains a model on the Contrastive learning ASTformer dataset. 
The model is a simple LSTM model that takes in learned contrastive embeddings and outputs a binary classification label for each amino acid in the sequence as catalytic site residue or not. 
The training loop includes early stopping based on the validation loss.
'''


# general
import datetime
import argparse
import os
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import glob

# model
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils.rnn import pad_sequence
import torch.utils.checkpoint as checkpoint
import torch.nn as nn

# eval/vis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from matplotlib import pyplot as plt


#torch.backends.cuda.matmul.allow_tf32 = True
torch.cuda.set_per_process_memory_fraction(1.0)
#torch.backends.cudnn.allow_tf32 = True

# parse args
parser = argparse.ArgumentParser(description='Train a model on the ASTformer dataset')
parser.add_argument('--contrastive_representations', 
                    type=str, 
                    default='/scratch/user/uqwriege/masters_thesis/ASTformer/data_EC/processed_seqlen900_tensors/padded_tensor.pt',
                    help='Path to the directory containing preprocessed (padded) tensor of contrastive representations of sequence data')

parser.add_argument('--metadata', 
                    type=str,
                    default='/scratch/user/uqwriege/masters_thesis/ASTformer/data_EC/processed_seqlen900_tensors/metadata_paired.tsv',
                    help='Path to the metadata for each sequences... must be in same order as embeddings/targets')

parser.add_argument('--leave_out', 
                    type=str,
                    default=None,
                    required=False,
                    help='Path to the file containing the list of sequences to leave out of training if needed')

parser.add_argument('--evaluation_set',
                    type=str,
                    default=None,
                    help='Path to the file containing the list of sequences to use as an evaluation set')

parser.add_argument('--test_set',
                    type=str,
                    default=None,
                    help='Path to the file containing the list of sequences to use as a test set')

parser.add_argument('--hyperparams', 
                    type=str, 
                    help='JSON file containing hyperparameters for training')

parser.add_argument('--output', 
                    type=str, 
                    help='Path to the directory to save everything... subdirs will be created for models, logs, etc.')

parser.add_argument('--test',
                    action='store_true',
                    help='If true, the code will stop before training to help test the code')

parser.add_argument('--benchmark',
                    default=None,
                    required=False,
                    type=str,
                    help='If given, the code will test the model on the benchmark dataset given the path for the squidly embeddings of the family specific sequences')

parser.add_argument('--leaked_entries',
                    default=None,
                    required=False,
                    type=str,
                    help='If given, the code will test the model on the leaked sequences given the path the list of dirty sequences - used to benchmark against AEGAN and its leaked benchmark')

args = parser.parse_args()


# class ProteinLSTM(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers, dropout_rate):
#         super(ProteinLSTM, self).__init__()
#         self.hidden_dim = hidden_dim

#         # The LSTM takes protein embeddings as inputs and outputs hidden states with dimensionality hidden_dim.
#         self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)

#         # The linear layer that maps from hidden state space to the output space
#         self.hidden2out = nn.Linear(hidden_dim*2, output_dim)
        
#         self.best_model_path = ""
        
#     def forward(self, x):
#         lstm_out, _ = self.lstm(x)
#         output = self.hidden2out(lstm_out)
#         return output



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


def train_model(model, epochs, patience, train_loader, val_loader, optimizer, loss_function, path_to_save_models):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    best_val_loss = float('inf')
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for inputs, labels in tqdm(train_loader):
            optimizer.zero_grad()
            inputs = inputs.contiguous().to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze()
            
            with torch.autograd.detect_anomaly():
                loss = loss_function(outputs, labels)                    
                loss.backward()

            optimizer.step()
            train_losses.append(loss.item())

        # Validation loop can be added similarly
            
        
        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}')
        
        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_losses = []
        with torch.no_grad():  # No need to compute gradient when evaluating
            for inputs, labels in tqdm(val_loader):
                outputs = model(inputs.cuda())
                outputs = outputs.squeeze()
                labels = labels.float() # Convert labels to float, so both outputs and labels are floats (expected by loss_func)
                labels = labels.cuda()
                loss = loss_function(outputs, labels)
                val_losses.append(loss.item())

        # Calculate average validation loss
        avg_val_loss = sum(val_losses) / len(val_losses)
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss:.4f}')
        
        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            counter = 0  # Reset counter
            best_model_path = path_to_save_models + 'temp_best_model.pth'
            torch.save(model.state_dict(), best_model_path)  # Save the best model
            print("Model saved as the best model for this epoch")
        else:
            counter += 1
            if counter >= patience:
                print(f'Early stopping triggered after {patience} epochs of no improvement')
                break  # Stop training if validation loss does not improve after several epochs
        # save GPU memory by clearing cache
        # torch.cuda.empty_cache()
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1e6} MB")
    model.best_model_path = best_model_path
    return model


def save_best_model(hyperparams, best_model_path, save_location):
    model = ProteinLSTM(embedding_dim=hyperparams['embedding_dim'], hidden_dim=hyperparams['hidden_dim'], output_dim=hyperparams['output_dim'], num_layers=hyperparams['num_layers'], dropout_rate=hyperparams['dropout_rate'])
    model.load_state_dict(torch.load(best_model_path))
    
    # get best_model_path file name and remove prepath
    best_model_path = best_model_path.split('/')[-1]
    
    date = datetime.datetime.now().strftime("%d-%m-%y_%H-%M")
    
    hidden_dim = hyperparams["hidden_dim"]
    num_layers = hyperparams["num_layers"]
    dropout = hyperparams["dropout_rate"]
    batch_size = hyperparams["batch_size"]
    
    path = save_location + f"/{date}_{hidden_dim}_{num_layers}_{dropout}_{batch_size}_best_model.pth"
    torch.save(model.state_dict(), path)
    return path

    
def get_test_results(model, test_loader):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    actuals = []
    model = model.cuda()
    with torch.no_grad():  # Disables gradient calculation
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            predicted = torch.sigmoid(outputs)  # Convert logits to probabilities 
            predictions.extend(predicted.squeeze().tolist())
            actuals.extend(labels.tolist())
            
    # squeeze the sublists in actuals and predictions into 1 list
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

    return accuracy, precision, recall, f1


def get_cm_plot(actuals, predictions, save_location):
    cm = confusion_matrix(actuals, predictions)
    
    # clear old plots, initialise new one
    plt.clf()
    
    # plot the matrix
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_location + 'confusion_matrix.png')
    

def get_ROC_curve(actuals, predictions, save_location):
    fpr, tpr, _ = roc_curve(actuals, predictions)
    plt.clf()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(save_location + 'ROC_curve.png')
    
    AUC_ROC = np.trapz(tpr, fpr)
    return AUC_ROC


def get_PR_curve(actuals, predictions, save_location):
    precision, recall, _ = precision_recall_curve(actuals, predictions)
    plt.clf()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(save_location + 'PR_curve.png')
    
    AUC_PR = np.trapz(precision, recall)
    return AUC_PR
    

    
def get_best_threshold(actuals, predictions):
    best_threshold = 0
    best_f1 = 0
    for threshold in range(0, 101):
        threshold = threshold / 100
        predictions_binary = [1 if prediction > threshold else 0 for prediction in predictions]
        accuracy, precision, recall, f1 = get_evaluation_metrics(actuals, predictions_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1    

    
def evaluate_model(model, test_loader, save_location, val_threshold):
    actuals, predictions = get_test_results(model, test_loader)
    
    best_threshold, best_f1 = get_best_threshold(actuals, predictions)
    
    AUC_ROC = get_ROC_curve(actuals, predictions, save_location)
    
    AUC_PR = get_PR_curve(actuals, predictions, save_location)
    
    # covert the predictions to binary predictions using the threshold
    predictions = [1 if prediction > val_threshold else 0 for prediction in predictions]
    
    print(len(actuals),len(predictions))
    
    results = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})
    results.to_csv(save_location + 'results.tsv', sep='\t', index=False)

    # Save evaluation metrics
    accuracy, precision, recall, f1 = get_evaluation_metrics(actuals, predictions)
    with open(save_location + 'evaluation_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1: {f1:.4f}\n')
        f.write(f'best_threshold: {str(best_threshold)}\n')
        f.write(f'val_threshold(used): {str(val_threshold)}\n')
        f.write(f'AUC_ROC: {AUC_ROC:.4f}\n')
        f.write(f'AUC_PR: {AUC_PR:.4f}\n')
    
    get_cm_plot(actuals, predictions, save_location)

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


def process_data(df, emb_dir, emb_layer, max_len=1024):
    # get the residues of each sequence in the dataset using a for loop
    active_sites = []
    
    if 'BS' in df.columns:
        col = 'BS'
    elif 'Active sites' in df.columns:
        col = 'Active sites'
    elif 'AS' in df.columns:
        col = 'AS'
    else:
        raise ValueError("No column with active sites or BS found. processing the data likely failed")

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
        paired_active_sites.append(dict_df[label]["residues"])
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
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

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

        return embedding, target


def get_best_threshold_from_validation_data(model, val_loader):
    # put the model in evaluation mode
    model.eval()
    model = model.cuda()
    # get the predictions from the model
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            predicted = torch.sigmoid(outputs)
            predictions.extend(predicted.squeeze().tolist())
            actuals.extend(labels.tolist())
            
        
    # squeeze the sublists in actuals and predictions into 1 list
    actuals = [item for sublist in actuals for item in sublist]
    predictions = [item for sublist in predictions for item in sublist]
            
    # now determine the best threshold by adjusting the threshold and getting the best f1 score
    best_threshold = 0
    best_f1 = 0
    for threshold in range(0, 101):
        threshold = threshold / 100
        predictions_binary = [1 if prediction > threshold else 0 for prediction in predictions]
        accuracy, precision, recall, f1 = get_evaluation_metrics(actuals, predictions_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold



def test_on_Benchmarks(model, squid_embeddings ,save_location, best_threshold):
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    
    # load the benchmarking dataset
    benchmark_embedding_dir = squid_embeddings
    
    # load all the embeddings
    files = []
    for file in glob.glob(benchmark_embedding_dir + "/*.pt"):
        files.append(file)
        
    embeddings_tensors = []
    embeddings_labels = []
    for file in files:
        tensor = torch.load(file, map_location='cpu')
        label = file.split("/")[-1].split("|")[1]
        embeddings_tensors.append(tensor)
        embeddings_labels.append(label)
    
    # pad the tensors
    padded_tensor = manual_pad_sequence_tensors(embeddings_tensors, 1024)
    padded_tensor = torch.stack(padded_tensor)
    
    # make a test_loader
    test_dataset = CustomDataset(padded_tensor, torch.zeros(padded_tensor.shape[0], padded_tensor.shape[1]))
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # now predict the padded tensor
    model.eval()  # Set the model to evaluation mode
    predictions = []
    model = model.cuda()
    with torch.no_grad():  # Disables gradient calculation
        for inputs, labels in test_loader:
            inputs = inputs.cuda()
            outputs = model(inputs)
            predicted = torch.sigmoid(outputs)  # Convert logits to probabilities
            # convert to 0 or 1
            predicted = [1 if prediction > best_threshold else 0 for prediction in predicted.squeeze()]
            predictions.append(predicted)
            
    
    # make a dictionary using the entry label and the prediction sublists
    benchmark_predictions = {}
    for i in range(len(embeddings_labels)):
        benchmark_predictions[embeddings_labels[i]] = predictions[i]
            
    # load the jsons for the 6 files
    # use glob to retrieve the 6 jsons from the path
    json_path = "/scratch/project/squid/code_modular/benchmark_data/benchmark_jsons"
    json_files = []
    for file in glob.glob(json_path + "/*.json"):
        json_files.append(file)
    
    # make a list of the family names from the file name itself to help with tracking results
    family_names = [file.split("/")[-1].split(".")[0] for file in json_files]
    
    # for each json, load the json and get the actuals and then compare it with the predictions to get an f1 score
    accuracy_scores, precision_scores, recall_scores, f1_scores = [], [], [], []
    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)
        # get the keys of the dictionary, these are the entries
        entries = list(json_data.keys())
        
        # get the predicted scores from the list of predicted results, using the dictionary
        actuals = [json_data[entry] for entry in entries]
        # now make a list of 1024 zeros with 1s at the indexes of actuals
        actuals = [[1 if i in actuals[j] else 0 for i in range(1024)] for j in range(len(actuals))]
         
        predictions = [benchmark_predictions[entry] for entry in entries]
        
        # flatten the sublists
        actuals = [item for sublist in actuals for item in sublist]
        predictions = [item for sublist in predictions for item in sublist]
        
        accuracy, precision, recall, f1 = get_evaluation_metrics(actuals, predictions)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)
        precision_scores.append(precision)
        recall_scores.append(recall)
    
    # make a df with the results and the family names
    results = pd.DataFrame({'Family': family_names, 'Accuracy': accuracy_scores, 'Recall': recall_scores,  'Precision': precision_scores, 'F1': f1_scores})
    results.to_csv(save_location + '/benchmark_results.tsv', sep='\t', index=False)
    
################
################
################

def main(args):# Create the model instance
    # Make a base directory
    os.makedirs(args.output, exist_ok=True)
    # Make a subdirectory for the models
    os.makedirs(args.output + '/models', exist_ok=True)
    # Make a subdirectory for the logs
    os.makedirs(args.output + '/logs', exist_ok=True)
    # Make a subdirectory for the results
    os.makedirs(args.output + '/results', exist_ok=True)
    
    # Save the hyperparams used to the log files
    # with open(args.output + '/logs/hyperparams_used.json', 'w') as f:
    #     json.dump(hyperparams, f)
    
    with open(args.hyperparams, 'r') as f:
        hyperparams = json.load(f)
    
    model = ProteinLSTM(embedding_dim=hyperparams['embedding_dim'], hidden_dim=hyperparams['hidden_dim'], output_dim=hyperparams['output_dim'], num_layers=hyperparams['num_layers'], dropout_rate=hyperparams['dropout_rate'])
    #model = ProteinGRU(embedding_dim=hyperparams['embedding_dim'], hidden_dim=hyperparams['hidden_dim'], output_dim=hyperparams['output_dim'], num_layers=hyperparams['num_layers'], dropout_rate=hyperparams['dropout_rate'])
    # put it on the GPU
    model = model.cuda()
    
    # Load the metadata
    metadata = pd.read_csv(args.metadata, sep='\t')
    
    embeddings, targets , metadata = process_data(metadata, args.contrastive_representations, 48)
    
    if args.leaked_entries:
        with open(args.leaked_entries, 'r') as f:
            leaked_entries = f.readlines()
        leaked_entries = [x.strip() for x in leaked_entries]
    else:
        leaked_entries = None
    
    if args.leave_out and args.evaluation_set:
        # Load the leave out list
        with open(args.leave_out, 'r') as f:
            leave_out = f.read().splitlines()
            
        # Load the evaluation set list
        with open(args.evaluation_set, 'r') as f:
            evaluation_set = f.read().splitlines()
        
        # Split the data into training, leave out validation and evaluation sets
        train_indices = []
        leave_out_indices = []
        evaluation_indices = []
        for i, row in metadata.iterrows():
            if row['Entry'] in leave_out:
                leave_out_indices.append(i)
            elif row['Entry'] in evaluation_set:
                evaluation_indices.append(i)       # Should exclude entries that are already in leave_out (redundant)
            else:
                train_indices.append(i)
                if args.leaked_entries:
                    if row['Entry'] in leaked_entries:
                        train_indices.append(i)
    
        
        # Extra evaluation dataset
        eval_embeddings = embeddings[evaluation_indices]
        eval_targets = targets[evaluation_indices]
        
        embeddings = embeddings[train_indices]
        targets = targets[train_indices]
        
        left_out_val_dataset = CustomDataset(eval_embeddings, eval_targets)
        left_out_val_loader = DataLoader(left_out_val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Keep memory free
        # del eval_embeddings, eval_targets, left_out_val_dataset, train_indices
    elif (args.leave_out):
        # Load the leave out list
        with open(args.leave_out, 'r') as f:
            leave_out = f.read().splitlines()
            
        # Split the data into training and leave out sets
        train_indices = []
        leave_out_indices = []
        for i, row in metadata.iterrows():
            if row['Entry'] in leave_out:
                leave_out_indices.append(i)
            else:
                train_indices.append(i)
                if args.leaked_entries:
                    if row['Entry'] in leaked_entries:
                        train_indices.append(i)
        
        embeddings = embeddings[train_indices]
        targets = targets[train_indices]
        
        # Keep memory free
        # del train_indices
    elif (args.evaluation_set):
        with open(args.evaluation_set, 'r') as f:
            evaluation_set = f.read().splitlines()
            
        # Split the data into training and leave out validation sets
        train_indices = []
        evaluation_indices = []
        for i, row in metadata.iterrows():
            if row['Entry'] in evaluation_set:
                evaluation_indices.append(i)
            else:
                train_indices.append(i)
                if args.leaked_entries:
                    if row['Entry'] in leaked_entries:
                        train_indices.append(i)
                
        # Extra evaluation dataset
        eval_embeddings = embeddings[evaluation_indices]
        eval_targets = targets[evaluation_indices]
        
        embeddings = embeddings[train_indices]
        targets = targets[train_indices]
        
        left_out_val_dataset = CustomDataset(eval_embeddings, eval_targets)
        left_out_val_loader = DataLoader(left_out_val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Keep mem free
        # del train_indices, left_out_val_dataset, eval_embeddings, eval_targets
        
    else:
        embeddings = embeddings
        targets = targets
    
    # Create the dataset and dataloaders
    dataset = CustomDataset(embeddings, targets)
    
    # del embeddings, targets
    
    # Generate indices of Train, test and validation dataset from train_dataset for training split
    train_size = int(hyperparams['train'] * len(dataset))
    val_size = int(hyperparams['val'] * len(dataset))
    test_size = int(hyperparams['test'] * len(dataset))
    
    if (train_size+val_size+test_size) < len(dataset):
        # add the difference to test_size
        test_size += len(dataset) - (train_size+val_size+test_size)
    
    # Generate a list of all indices
    indices = list(range(len(dataset)))
    
    # Subset the indices randomly for train, val and test
    train_indices, val_indices, test_indices = random_split(indices, [train_size, val_size, test_size])
    
    # Subset the datasets using the indices
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
    
    # Add a column to the metadata to track the split
    for index, row in metadata.iterrows():
        #check which split the row is in
        if index in train_indices:
            metadata.loc[index, 'split'] = 'train'
        elif index in val_indices:
            metadata.loc[index, 'split'] = 'val'
        elif index in test_indices:
            metadata.loc[index, 'split'] = 'test'
        else:
            metadata.loc[index, 'split'] = 'left_out'
                        
    if args.evaluation_set != None:
        # change the indices of metadata['split'] to evaluation that correspond with the evaluation set
        for index, row in metadata.iterrows():
            if row['Entry'] in evaluation_set:
                metadata.loc[index, 'split'] = 'evaluation'
    
    print(len(metadata[metadata['split']=='left_out']))

    # Save the metadata
    metadata.to_csv(args.output + '/logs/metadata_with_split.tsv', sep = '\t', index=False)
    
    #del train_dataset, val_dataset, test_dataset, dataset
    
    # Define the loss function and optimizer
    # BCEWithLogitsLoss is used when the model outputs logits, which is the case here
    # I define a weighted loss penalty because of class imbalance
    weight = torch.tensor([1, hyperparams["loss_weight"]], dtype=torch.float)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weight[1]).cuda()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    # cancel if test
    #if args.test:
    #    print("Test complete")
    #    return
    
        
    if args.test is False:
        print(train_loader)
        print(val_loader)
        print("Memory before training")
        print(f"Allocated memory: {torch.cuda.memory_allocated() / 1e6} MB")
        print(f"Cached memory: {torch.cuda.memory_reserved() / 1e6} MB")
        
        # Train the model
        model = train_model(
            model,
            epochs=hyperparams['epochs'],
            patience=hyperparams['patience'],
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            path_to_save_models=args.output + '/models/'
        )
        # del the val and train loader
        # del val_loader, train_loader

        # Save the best model
        best_path = save_best_model(hyperparams, model.best_model_path, args.output + '/models/')
    
        # Remove temporary models:
        os.remove(args.output + '/models/temp_best_model.pth')
        
        # load the best model
        
        print(best_path)
        model = ProteinLSTM(embedding_dim=hyperparams['embedding_dim'], hidden_dim=hyperparams['hidden_dim'], output_dim=hyperparams['output_dim'], num_layers=hyperparams['num_layers'], dropout_rate=hyperparams['dropout_rate'])
        model.load_state_dict(torch.load(best_path))
    
    if args.test:
        model = ProteinLSTM(embedding_dim=hyperparams['embedding_dim'], hidden_dim=hyperparams['hidden_dim'], output_dim=hyperparams['output_dim'], num_layers=hyperparams['num_layers'], dropout_rate=hyperparams['dropout_rate'])
        model.load_state_dict(torch.load('/scratch/project/squid/Hydrolases/3.X/Scheme3_1000/LSTM/models/19-09-24_10-30_96_2_0.1_350_best_model.pth'))
    
    # Evaluate the model
    #evaluate_model(model, test_loader, args.output + '/results/test_set_')
    
    if args.evaluation_set != None:
        val_threshold = get_best_threshold_from_validation_data(model, val_loader)
        evaluate_model(model, left_out_val_loader, args.output + '/results/left_out_super_valid_set_', val_threshold)
        if args.benchmark != None:
            test_on_Benchmarks(model, args.benchmark, args.output + '/results', val_threshold)

    
# run the main
if __name__ == '__main__':
    main(args)