import torch
from torch.utils.data import Dataset, DataLoader



               
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

def main():
    # load the torch dataset
    dataset = torch.load('/scratch/project/squid/data/ALL_reviewed_1024/Scheme2_2500/paired_embeddings_dataset.pt')
    
    # TEST get the first item in the dataset
    #item = dataset[1]
    print(dataset.main_store)
    #print(item)
    
    return


if __name__ == '__main__':
    main()