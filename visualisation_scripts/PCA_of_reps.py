import pandas
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
import argparse
import matplotlib.pyplot as plt    
import umap
from sklearn.manifold import TSNE

# python /scratch/project/squid/code_modular/final_models/PCA_of_reps.py --reps_dir /scratch/project/squid/code_modular/datasets/pred_med/squidly_reps --prediction_tsv /scratch/project/squid/code_modular/datasets/pred_med/dataset_1_results.pkl --output_dir /scratch/project/squid/code_modular/datasets/pred_med/h2_ --layer h2

# arg parser
def get_args():
    """
    Get the arguments for the script.

    Returns
    -------
    argparse.Namespace
        The arguments for the script.
    """
    parser = argparse.ArgumentParser(description="Perform PCA on the representations of the data.")
    parser.add_argument("--reps_dir", type=str, help="The path to the representations.")
    parser.add_argument("--prediction_tsv", type=str, help="The path to the predictions.")
    parser.add_argument("--output_dir", type=str, help="The path to the output directory.")
    parser.add_argument("--layer", type=str, help="The layer to get the representations from. Can be \'h1\', \'h2\', or \'out\'.")
    return parser.parse_args()



def PCA_of_reps(reps, n_components):

    # Standardize the representations
    scaler = StandardScaler()
    reps = scaler.fit_transform(reps)

    # Perform PCA
    pca = PCA(n_components=n_components)
    reps = pca.fit_transform(reps)

    # Convert the transformed representations to a tensor
    reps = torch.tensor(reps, dtype=torch.float)

    return reps



def get_pos_rep(reps_dir, layer, Entry, pos):

    # Get the path to the representation file
    if layer == "h1":
        rep_file = f"{reps_dir}/{Entry}_hidden_rep1.pt"
    elif layer == "h2":
        rep_file = f"{reps_dir}/{Entry}_hidden_rep2.pt"
    elif layer == "out":
        rep_file = f"{reps_dir}/{Entry}_output_rep.pt"
    else:
        raise ValueError("Invalid layer name, use h1, h2, or out.")
        
    # Load the representations
    reps = torch.load(rep_file)

    # Get the representation of the position
    pos_rep = reps[pos]

    return pos_rep




def get_multi_pos_rep(reps_dir, layer, Entry, possies):

    # Get the path to the representation file
    if layer == "h1":
        rep_file = f"{reps_dir}/{Entry}_hidden_rep1.pt"
    elif layer == "h2":
        rep_file = f"{reps_dir}/{Entry}_hidden_rep2.pt"
    elif layer == "out":
        rep_file = f"{reps_dir}/{Entry}_output_rep.pt"
    else:
        raise ValueError("Invalid layer name, use h1, h2, or out.")
        
    # Load the representations
    reps = torch.load(rep_file)

    # Get the representation of the position
    pos_reps = reps[possies]

    return pos_reps
    


def main():    
    # get args
    args = get_args()
    
    df = pandas.read_csv(args.prediction_tsv, sep=",")    
    
    CR_rep_list = []
    non_CR_rep_list = []
    entry = []
    # drop any rows that have NaN values in Squidly_CR_Position_x
    df = df.dropna(subset=["Squidly_CR_Position"])
    
    for i, row in df.iterrows():
        # get the catalytic residue positions from the prediction tsv
        CR_positions = row["Squidly_CR_Position"]
        # convert to a list of integers
        CR_positions = [int(X) for X in CR_positions.split("|")]
        seq = row["Sequence"]
        # get the AA of the CR_positions
        CR_AA = [seq[X] for X in CR_positions]
        
        # now randomly sample some other positions in the sequence 2-3 times to get some more of the same AAs
        # get the positions of the CR_AAs
        same_AA_positions = [X for X in range(len(seq)) if seq[X] in CR_AA]
        # sample 2-3 times the number of CR_AAs that are in the set of CR_AA
        if len(CR_AA) >= len(same_AA_positions):
            number_to_sample = len(same_AA_positions)
        else:
            number_to_sample = len(CR_AA) 
        
        non_CR_positions = random.sample(same_AA_positions, number_to_sample)
        
        Entry = row["Entry"]
        
        # get the representations of the residues in non_CR_positions and CR_positions
        non_CR_reps = get_multi_pos_rep(args.reps_dir, args.layer, Entry, non_CR_positions)
        CR_reps = get_multi_pos_rep(args.reps_dir, args.layer, Entry, CR_positions)
        CR_rep_list.append(CR_reps)
        non_CR_rep_list.append(non_CR_reps)
        
    # now run PCA on the representations and get the first 2 components
    CR_reps = torch.cat(CR_rep_list, dim=0)
    non_CR_reps = torch.cat(non_CR_rep_list, dim=0)
    
    #concat and PCA the representations
    reps = torch.cat([CR_reps, non_CR_reps], dim=0)
    reps = reps.numpy()
    
    # scale the reps
    scaler = StandardScaler()
    reps = scaler.fit_transform(reps)
    CR_reps = scaler.transform(CR_reps)
    non_CR_reps = scaler.transform(non_CR_reps)
    
    # generate mean embeddings for the CR and non-CR representations
    CR_mean = np.mean(CR_reps, axis=0)
    non_CR_mean = np.mean(non_CR_reps, axis=0)
    
    # find the std of the embeddings
    CR_std = np.std(CR_reps, axis=0)
    non_CR_std = np.std(non_CR_reps, axis=0)
    
    # now print
    print("CR mean: ", CR_mean)
    print("non-CR mean: ", non_CR_mean)
    print("CR std: ", CR_std)
    print("non-CR std: ", non_CR_std)
    num_positions = len(list(CR_mean))
    print(num_positions)
    positions = np.arange(num_positions)
    bar_width = 0.35
    # Plotting
    plt.figure(figsize=(48, 6))
    plt.bar(positions - bar_width/2, CR_mean, yerr=CR_std, width=bar_width, label='CR Mean', alpha=0.7, color='blue', capsize=5)
    plt.bar(positions + bar_width/2, non_CR_mean, yerr=non_CR_std, width=bar_width, label='non-CR Mean', alpha=0.7, color='orange', capsize=5)
    plt.xlabel('Embedding Position')
    plt.ylabel('Mean Value')
    plt.title('Mean and Standard Deviation of CR and non-CR Embeddings')
    plt.xticks(positions)
    plt.legend()
    plt.tight_layout()
    # save the plot
    plt.savefig(args.output_dir + "mean_and_std_of_reps.png")
    plt.close()
        
    # now PCA
    pca = PCA_of_reps(reps, 2)

    
    reducer = umap.UMAP()
    
    # fit them together, so that the UMAP is fit on the whole dataset
    reducer.fit(reps, dim=0)
    # now transform the representations
    UMAP_CR_reps = reducer.transform(CR_reps)
    UMAP_non_CR_reps = reducer.transform(non_CR_reps)
    
    # now plot the UMAP of the representations
    plt.scatter(UMAP_CR_reps[:,0], UMAP_CR_reps[:,1], color="red", label="Catalytic Residues")
    plt.scatter(UMAP_non_CR_reps[:,0], UMAP_non_CR_reps[:,1], color="blue", label="Non-Catalytic Residues", alpha=0.2)
    plt.legend()
    # save the plot
    plt.savefig(args.output_dir + "UMAP_of_reps.png")
    plt.close()
    
    
    # now with t-SNE
    tsne = TSNE(n_components=2)
    
    # fit together then transform separately
    tsne_reps = tsne.fit_transform(reps, dim=0)
    tSNE_CR_reps = tsne_reps[:len(CR_reps)]
    tSNE_non_CR_reps = tsne_reps[len(CR_reps):]   
    
    # now plot the t-SNE of the representations
    plt.scatter(tSNE_CR_reps[:,0], tSNE_CR_reps[:,1], color="red", label="Catalytic Residues")
    plt.scatter(tSNE_non_CR_reps[:,0], tSNE_non_CR_reps[:,1], color="blue", label="Non-Catalytic Residues", alpha=0.2)
    plt.legend()
    # save the plot
    plt.savefig(args.output_dir + "t-SNE_of_reps.png")
    plt.close()
    
    # now do 3D UMAP, t-SNE, and PCA
    reducer = umap.UMAP(n_components=3)
    tsne = TSNE(n_components=3)
    
    # fit them together, so that the UMAP is fit on the whole dataset
    reducer.fit(reps, dim=0)
    # now transform the representations
    UMAP_CR_reps = reducer.transform(CR_reps)
    UMAP_non_CR_reps = reducer.transform(non_CR_reps)
    
    # now plot the UMAP of the representations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(UMAP_CR_reps[:,0], UMAP_CR_reps[:,1], UMAP_CR_reps[:,2], color="red", label="Catalytic Residues")
    ax.scatter(UMAP_non_CR_reps[:,0], UMAP_non_CR_reps[:,1], UMAP_non_CR_reps[:,2], color="blue", label="Non-Catalytic Residues", alpha=0.2)
    plt.legend()
    # save the plot
    plt.savefig(args.output_dir + "3D_UMAP_of_reps.png")
    plt.close()
    
    # fit together then transform separately
    tsne_reps = tsne.fit_transform(reps, dim=0)
    tSNE_CR_reps = tsne_reps[:len(CR_reps)]
    tSNE_non_CR_reps = tsne_reps[len(CR_reps):]

    # now plot the t-SNE of the representations
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tSNE_CR_reps[:,0], tSNE_CR_reps[:,1], tSNE_CR_reps[:,2], color="red", label="Catalytic Residues")
    ax.scatter(tSNE_non_CR_reps[:,0], tSNE_non_CR_reps[:,1], tSNE_non_CR_reps[:,2], color="blue", label="Non-Catalytic Residues", alpha=0.2)
    plt.legend()
    # save the plot
    plt.savefig(args.output_dir + "3D_t-SNE_of_reps.png")
    plt.close()
    
    
    
    
    
    
        
        
        
        
    
    
if __name__ == "__main__":
    main()