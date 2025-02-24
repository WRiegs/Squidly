import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

aln = "/scratch/project/squid/kari_seqs/kari_ancestors_extants.aln"
tsv = pd.read_csv('/scratch/project/squid/kari_seqs/kari_extantsfiltered_LSTM.tsv', sep='\t')

# get the pickle file with the logits
with open('/scratch/project/squid/kari_seqs/BS_site_probs.pkl', 'rb') as f:
    logits = pkl.load(f)
    
# multiply the logits in the dictionary by 100 to scale them
for key, value in logits.items():
    logits[key] = [x*100 for x in value]
    
# read the aln file, make a dictionary with the entry name and sequence
# file starts with a header, then the sequenc headers start with > and sequences without like in fasta
sequences = {}
with open(aln, 'r') as f:
    for line in f:
        if line.startswith('>'):
            entry = line[1:].strip()
            sequences[entry] = ''
        else:
            sequences[entry] += line.strip()

        
# go through sequences in the aln dictionary, and map the logits to the sequences, when there's a '-'gap character, just give it 0
logit_lists = []
labels = []
for entry, seq in sequences.items():
    if entry not in logits:
        continue
    labels.append(entry)
    logit_list = []
    index = 0
    for i, char in enumerate(seq):
        if char == '-':
            logit_list.append(0)
        else:
            logit = logits.get(entry)[index]
            logit_list.append(logit)
            index+=1
    logit_lists.append(logit_list)
    
# make a dataframe from the logit lists - each row is the logit list for a sequence
df = pd.DataFrame(logit_lists, columns=range(len(logit_lists[0])), index=labels)

# print the max of the dataframe
print(df.max().max())

print(df.max())


print(df)

# plot the dataframe as a heatmap with a colour scale from 0 to 100, and the sequences as the y axis
# don't downsample the x axis, it looks like positions are missing because of the length of the sequences
plt.figure(figsize=(20, 10))
plt.title("Heatmap of BS site probabilities")
sns.heatmap(df, cmap='viridis', yticklabels=True)
plt.show()
plt.savefig('/scratch/project/squid/kari_seqs/extant_2D_BS_heatmap.png')