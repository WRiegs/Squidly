import argparse
import pathlib
import glob
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import os


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument(
        "--model_location",
        type=pathlib.Path,
        help="The contrastive model file location",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=pathlib.Path,
        help="Dir of pre-extracted embedding representations from LLM",
    )
    parser.add_argument(
        '--embedding_size',
        type=int,
        default=2560,
        help='Size of the input tensor'
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="output directory for extracted contrastive representations",
    )
    parser.add_argument(
        '--save_new_pt',
        action='store_true',
        default=False,
        help="Tell script to save the contrastive representations as a new .pt for each sequence"
    )
    parser.add_argument(
        '--layer',
        type=int,
        default=36,
        help='Layer to extract the embeddings from'
    )
    parser.add_argument('--hyperparams', 
        type=pathlib.Path, 
        help='JSON file containing hyperparameters for training'
    )
    return parser


# SUBJECT TO CHANGE
class ContrastiveModel(nn.Module):
    def __init__(self, input_dim=2560, output_dim=128, dropout_prob=0.1):
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


def main(args):
    if not args.output_dir.exists():
        os.makedirs(args.output_dir)
        
    
    if args.hyperparams is not None:
        with open(args.hyperparams, 'r') as f:
            hyperparams = json.load(f)
        input_dim = hyperparams['embedding_dim']
        output_dim = hyperparams['output_dim']
        dropout_prob = hyperparams['dropout_rate']
    else:
        dropout_prob = 0.1
        input_dim = args.embedding_size
        output_dim = 128 # the output dimension based on your requirement
        
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveModel(input_dim, output_dim, dropout_prob).to(device)
    # load the trained model
    model.load_state_dict(torch.load(args.model_location))
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    layer = args.layer
    
    # load the embeddings
    embedding_files = glob.glob(str(args.embeddings_dir / '**/*'), recursive=True)
    embedding_files = [pathlib.Path(emb_file) for emb_file in embedding_files]
    
    embedding_files = []

    # Walk through the directory tree
    for root, dirs, files in os.walk(str(args.embeddings_dir)):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            embedding_files.append(file_path)
    
    embedding_files = [pathlib.Path(emb_file) for emb_file in embedding_files]

    
    # Generating contrastive representations!
    memory_filled = False #used to track and adjust to memory problems.
    for embedding_file in tqdm(embedding_files):
        # if the embeddings have already been processed, skip them
        try:
            exists = (args.output_dir / f'{embedding_file.name.split("|")[0]}|{embedding_file.name.split("|")[1]}|contrastive.pt').exists()
        except:
            exists = False
        if exists:
            continue
        
        seq = torch.load(embedding_file)['representations'][layer]
        seq_rep = []
        
        # load all the tokens in the seq into a dataloader for parrallel processing
        dataloader = torch.utils.data.DataLoader(seq, batch_size=len(seq), shuffle=False)
        
        # put the dataloader through the model and get the new sequence
        for batch in dataloader:
            batch = batch.to(device)
            seq_rep.extend(model(batch))
            del batch
        
        torch.cuda.empty_cache()

        # turn the seq_rep list of tensors into a tensor
        seq_rep = torch.stack(seq_rep)
        try:
            # move the tensor to the cpu so that, when I load it later, I don't have to worry about the device
            seq_rep = seq_rep.cpu()
            torch.save(seq_rep, args.output_dir / f'{embedding_file.name.split("|")[0]}|{embedding_file.name.split("|")[1]}|contrastive.pt')
        except:
            # get the name of the last directory that the file is in
            path = str(embedding_file.parts[-2])
            print(path)
            try:
                torch.save(seq_rep, args.output_dir / f'{path.split("|")[0]}|{path.split("|")[1]}|contrastive.pt')
            except:
                continue

    


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)