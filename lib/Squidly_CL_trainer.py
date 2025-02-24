import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import pickle
import pandas as pd
import argparse
import pathlib
import torch
import glob
import psutil
import os
import time
import tracemalloc
import json
from matplotlib import pyplot as plt

tracemalloc.start()

def create_parser():
    # parse args
    parser = argparse.ArgumentParser(description='Train a model on the ASTformer dataset')
    
    parser.add_argument('--embedding_size',
                        type=int,
                        default=2560,
                        help='Size of the input tensor')

    parser.add_argument('--pair_scheme', 
                        type=pathlib.Path, 
                        help='Path to the directory containing pkl files that describe pair schema of embeddings used for training')

    parser.add_argument('--metadata', 
                        type=pathlib.Path,
                        default='/scratch/user/uqwriege/masters_thesis/ASTformer/data_EC/processed_seqlen900_tensors/metadata_paired.tsv',
                        help='Path to the metadata for each sequences... must be in same order as embeddings/targets')

    parser.add_argument('--leave_out', 
                        type=pathlib.Path, 
                        default='/scratch/user/uqwriege/masters_thesis/ASTformer/data_EC/processed_seqlen900_tensors/remove_empty.txt',
                        help='Path to the file containing the list of sequences to leave out of training')

    parser.add_argument('--evaluation_set',
                        type=pathlib.Path,
                        default=pathlib.Path("/scratch/project/squid/data/full_set_AS-EC_data/Low40ID_250subset.txt"),
                        help='Path to the file containing the list of sequences to use as an evaluation set')

    parser.add_argument('--hyperparams', 
                        type=pathlib.Path, 
                        help='JSON file containing hyperparameters for training')

    parser.add_argument('--output', 
                        type=pathlib.Path, 
                        help='Path to the directory to save everything... subdirs will be created for models, logs, etc.')

    parser.add_argument('--test',
                        action='store_true',
                        help='If true, the code will stop before training to help test the code')

    return parser


# Define the dataset class
class PairedEmbeddingsDataset(Dataset):
    def __init__(self, tensor1, tensor2, labels):
        assert tensor1.shape == tensor2.shape, "The two tensors must have the same shape"
        assert tensor1.shape[0] == len(labels), "The number of labels must match the number of rows in the tensors"
        
        self.tensor1 = tensor1
        self.tensor2 = tensor2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.tensor1[idx], dtype=torch.float32), \
               torch.tensor(self.tensor2[idx], dtype=torch.float32), \
               self.labels[idx]


               
class PointerPairedDataset(Dataset):
    def __init__(self, main_store, pair_indices, labels):
        self.main_store = main_store     # Dict - e.g.  {i: torch.rand(512) for i in range(1000)}
        self.pair_indices = pair_indices # Each pair is a tuple of keys from `main_store`, like (0, 5) or (2, 8)
        self.labels = labels             # List of labels for each pair

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        # Retrieve indices for the current pair
        idx1, idx2 = self.pair_indices[idx]
        # Look up the actual data in the main store using these indices
        item1, item2 = self.main_store[idx1], self.main_store[idx2]
        label = self.labels[idx]
        return item1, item2, label  # Return the pair for contrastive training



# Define the ffnn CR model
class ContrastiveModel(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_prob):
        super(ContrastiveModel, self).__init__()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(input_dim, int(input_dim/2))
        self.fc2 = nn.Linear(int(input_dim/2), int(input_dim/4))
        self.fc3 = nn.Linear(int(input_dim/4), output_dim)
        
    def forward(self, x):
        x = self.dropout(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    def train_model(self, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, model_out, patience=3):
        best_val_loss = float('inf')
        counter = 0

        all_train_losses = []
        all_val_losses = []
        
        for epoch in range(num_epochs):
            self.train()
            train_losses = []
            for batch in train_dataloader:
                embedding1, embedding2, labels = batch
                optimizer.zero_grad()
                output1 = self(embedding1.cuda())
                output2 = self(embedding2.cuda())
                loss = criterion(output1, output2, labels.float().cuda())
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            avg_train_loss = sum(train_losses) / len(train_losses)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss}')
            all_train_losses.append(avg_train_loss)

            # Validation
            self.eval()
            val_losses = []
            with torch.no_grad():
                for vbatch in val_dataloader:
                    embedding1, embedding2, labels = vbatch
                    output1 = self(embedding1.cuda())
                    output2 = self(embedding2.cuda())
                    loss = criterion(output1, output2, labels.float().cuda())
                    val_losses.append(loss.item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}')
            all_val_losses.append(avg_val_loss)
            
            # Early stopping logic
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                counter = 0  # Reset counter
                best_model_path = f'{model_out}/temp_best_model.pt'
                torch.save(self.state_dict(), best_model_path)  # Save the best model
                print("Model saved as the best model for this epoch")
            else:
                counter += 1
                if counter >= patience:
                    print(f'Early stopping triggered after {patience} epochs of no improvement')
                    break  # Stop training if validation loss does not improve after several epochs
        
        return all_train_losses, all_val_losses
    
    def loss_plot(self, all_train_losses, all_val_losses, dir):
        epochs = range(1, len(all_train_losses) + 1)
        plt.plot(epochs, all_train_losses, label='Training Loss')
        plt.plot(epochs, all_val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        plt.savefig(dir / 'loss_plot.png')


def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss * 0.000001  # Convert bytes to MB


def monitor_memory_usage(interval=120):
    mem_usage = []
    try:
        while True:
            mem_usage = get_memory_usage()
            print(f"Memory Usage: {mem_usage:.2f} MB")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped.")


def main(args):
    # Make a base directory
    os.makedirs(args.output, exist_ok=True)
    # Make a subdirectory for the models
    os.makedirs(args.output / 'models', exist_ok=True)
    # Make a subdirectory for the logs
    os.makedirs(args.output / 'logs', exist_ok=True)
    model_out = args.output / 'models'

    if args.pair_scheme.is_file():
        # load the torch dataset directly
        print("Loading the pair scheme as a torch dataset...")
        dataset = torch.load(args.pair_scheme)
    else:
        # raise an error
        raise ValueError("The pair scheme must be a pytorch data file that is correctly processed")
    
    if args.hyperparams is not None:
        with open(args.hyperparams, 'r') as f:
            hyperparams = json.load(f)
        input_dim = hyperparams['embedding_dim']
        output_dim = hyperparams['output_dim']
        dropout_prob = hyperparams['dropout_rate']
        num_epochs = hyperparams['epochs']
        # save the hyperparams to the outfile so that its available later
        with open(args.output / 'logs' / 'hyperparams_used_for_contrastive.json', 'w') as f:
            json.dump(hyperparams, f)
    else:
        dropout_prob = 0.1
        input_dim = args.embedding_size
        output_dim = 128
        num_epochs = 40
    
    # split the dataset into training and validation-for-early-stopping sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    batch_size = 60000
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize the model, loss function, and optimizer
    model = ContrastiveModel(input_dim, output_dim, dropout_prob)
    model.cuda()
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    all_train_losses, all_val_losses = model.train_model(train_dataloader, val_dataloader, criterion, optimizer, num_epochs, model_out)
    model.loss_plot(all_train_losses, all_val_losses, args.output / 'logs')


if __name__ == '__main__':
    import threading
    # Start the memory monitoring in a separate thread
    #monitor_thread = threading.Thread(target=monitor_memory_usage, daemon=True)
    #monitor_thread.start()
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)