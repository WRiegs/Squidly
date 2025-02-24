'''
This script trains a model on the Contrastive learning ASTformer dataset. 
The model is a simple LSTM model that takes in learned contrastive embeddings and outputs a binary classification label for each amino acid in the sequence as catalytic site residue or not. 
The training loop includes early stopping based on the validation loss.
'''

# python /scratch/project/squid/code/ESM2_LSTM_trainer.py --embeddings /scratch/project/squid/code_modular/reproduction_runs/esm2/results/padded_tensor.pt --targets /scratch/project/squid/code_modular/reproduction_runs/esm2/results/padded_targets_index0.pt --metadata /scratch/project/squid/code_modular/reproduction_runs/esm2/results/metadata_paired.tsv --evaluation_set /scratch/project/squid/code_modular/reproduction_runs/esm2/Low30_mmseq_ID_250_exp_subset.txt --hyperparams /scratch/project/squid/code/ESM2_Hyperparameters_bidirectional.json --output /scratch/project/squid/code_modular/reproduction_runs/esm2/results

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
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn

# eval/vis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import seaborn as sns
from matplotlib import pyplot as plt

# parse args
parser = argparse.ArgumentParser(description='Train a model on the ASTformer dataset')
parser.add_argument('--embeddings', 
                    type=str, 
                    help='Path to the directory containing preprocessed (padded) tensor of contrastive representations of sequence data')

parser.add_argument('--emb_size',
                    type=int,
                    help='Size of the embeddings')

parser.add_argument('--targets',
                    type=str,
                    help='Path to the directory containing preprocessed tensor of target labels')

parser.add_argument('--metadata', 
                    type=str,
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

parser.add_argument('--leaked_entries',
                    type=str,
                    default=None,
                    help='Path to the file containing the list of sequences that are leaked entries')

args = parser.parse_args()


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
        
    def train_model(self, epochs, patience, train_loader, val_loader, optimizer, loss_function, path_to_save_models):
        best_val_loss = float('inf')
        counter = 0
        # Training loop
        for epoch in tqdm(range(epochs)):
            self.train()  # Set model to training mode
            train_losses = []
            
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # Clear gradients
                inputs = inputs.cuda()
                labels = labels.cuda()
                outputs = self(inputs)  # Forward pass
                outputs = outputs.squeeze()
                labels = labels.float() # Convert labels to float, so both outputs and labels are floats (expected by loss_func)
                loss = loss_function(outputs, labels)  # Compute loss
                loss.backward(retain_graph=True)  # Backpropagation
                optimizer.step()  # Update weights
                train_losses.append(loss.item())
                torch.cuda.empty_cache()
            
            # Calculate average training loss
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}')
            
            # Validation phase
            self.eval()  # Set model to evaluation mode
            val_losses = []
            with torch.no_grad():  # No need to compute gradient when evaluating
                for inputs, labels in val_loader:
                    inputs = inputs.cuda()
                    outputs = self(inputs)
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
                torch.save(self.state_dict(), best_model_path)  # Save the best model
                print("Model saved as the best model for this epoch")
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs of no improvement')
                    break  # Stop training if validation loss does not improve after several epochs
            # save GPU memory by clearing cache
            torch.cuda.empty_cache()
        self.best_model_path = best_model_path
 

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

    
def evaluate_model(model, test_loader, save_location):
    actuals, predictions = get_test_results(model, test_loader)
    
    best_threshold, best_f1 = get_best_threshold(actuals, predictions)
    
    AUC_ROC = get_ROC_curve(actuals, predictions, save_location)
    
    AUC_PR = get_PR_curve(actuals, predictions, save_location)
    
    # covert the predictions to binary predictions using the threshold
    predictions = [1 if prediction > best_threshold else 0 for prediction in predictions]
    
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
        f.write(f'AUC_ROC: {AUC_ROC:.4f}\n')
        f.write(f'AUC_PR: {AUC_PR:.4f}\n')
    
    get_cm_plot(actuals, predictions, save_location)


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
    
    model = ProteinLSTM(embedding_dim=args.emb_size, hidden_dim=hyperparams['hidden_dim'], output_dim=hyperparams['output_dim'], num_layers=hyperparams['num_layers'], dropout_rate=hyperparams['dropout_rate'])
    
    # Load the metadata
    metadata = pd.read_csv(args.metadata, sep='\t')
    print('stugats')
    embeddings = torch.load(args.embeddings)
    
    # move the embeddings and targets to cpu
    embeddings = embeddings.cpu()
    
    print('stugats amassage')
    targets = torch.load(args.targets)
    targets = targets.cpu()
    
    # put it on the GPU
    model = model.cuda()
    
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
        
        left_out_val_dataset = TensorDataset(eval_embeddings, eval_targets)
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
        
        left_out_val_dataset = TensorDataset(eval_embeddings, eval_targets)
        left_out_val_loader = DataLoader(left_out_val_dataset, batch_size=hyperparams['batch_size'], shuffle=False)
        
        # Keep mem free
        # del train_indices, left_out_val_dataset, eval_embeddings, eval_targets
        
    else:
        embeddings = embeddings
        targets = targets
    
    # Create the dataset and dataloaders
    dataset = TensorDataset(embeddings, targets)
    
    del embeddings, targets
    
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
    
    # Make the loaders
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['batch_size'], shuffle=True)
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
    
    del train_dataset, val_dataset, test_dataset, dataset
    
    # Define the loss function and optimizer
    # BCEWithLogitsLoss is used when the model outputs logits, which is the case here
    # I define a weighted loss penalty because of class imbalance
    weight = torch.tensor([1, hyperparams["loss_weight"]], dtype=torch.float)
    loss_function = nn.BCEWithLogitsLoss(pos_weight=weight[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    
    # cancel if test
    #if args.test:
    #    print("Test complete")
    #    return
    
        
    if args.test is False:
        print(train_loader)
        print(val_loader)
        
        # Train the model
        model.train_model(
            epochs=hyperparams['epochs'],
            patience=hyperparams['patience'],
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            path_to_save_models=args.output + '/models/'
        )
        # del the val and train loader
        del val_loader, train_loader

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
        model.load_state_dict(torch.load('/scratch/project/squid/LSTMs/first_test_probsbroken/models/19-07-24_16-07_128_2_0.1_200_best_model.pth'))
    
    # Evaluate the model
    evaluate_model(model, test_loader, args.output + '/results/test_set_')
    
    if args.evaluation_set != None:
        evaluate_model(model, left_out_val_loader, args.output + '/results/left_out_super_valid_set_')
    
# run the main
if __name__ == '__main__':
    main(args)