# Squidly ![Squid](squidly.png)

![Overview Figure](overview_fig_.png)

Squidly, is a tool that employs a biologically informed contrastive learning approach to accurately predict catalytic residues from enzyme sequences. We offer Squidly as ensembled with Blast to achieve high accuracy at low and high sequence homology settings.

If you use squidly in your work please cite our preprint: https://www.biorxiv.org/content/10.1101/2025.06.13.659624v1

Also if you have any issues installing, please post an issue! We have tested this on ubuntu.

### Requirements
Squidly is dependant on the ESM2 3B or 15B protein language model. Running Suidly will automatically attempt to download each model.
The Smaller 3B model is lighter, runs faster and requires less VRAM. 

Currently we expect GPU access but if you require a CPU only version please let us know and we can update this!
### Installation
```
conda create --name squidly python=3.10
conda activate squidly
pip install squidly
squidly install
```
Running `squidly install` should automatically download all models from huggingface. Now you can run squidly (see **Usage** below).

Note if you get the below error:

```ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject```
 you may need to update numpy and pandas.


## Usage
For example to run the 3B model with a fasta file (in squidly only mode)
```bash
squidly run example.fasta esm2_t36_3B_UR50D 
```

Or to run as an ensemble with BLAST (you need to pass the database as well)
```
squidly run example.fasta esm2_t36_3B_UR50D output_folder/ --database reviewed_sprot_08042025.csv
```
Where `reviewed_sprot_08042025.csv` is the example database (i.e. a csv file with the following columns) 

You can see ours which is zipped in the data folder..


| Entry      | Sequence         | Residue                                  |
|------------|------------------|------------------------------------------|
| A0A009IHW8 | MSLEQKKGADIIS    | 207                                      |
| A0A023I7E1 | MRFQVIVAAATITMIY | 499\|577\|581                            |
| A0A024B7W1 | MKNPKKKSGGFRIV   | 1552\|1576\|1636\|2580\|2665\|2701\|2737 |
| A0A024RXP8 | MYRKLAVISAFL     | 228\|233                                 |


```bash
 Usage: squidly [OPTIONS] FASTA_FILE ESM2_MODEL [OUTPUT_FOLDER] [RUN_NAME]                                         
                                                                                                                   
 Find catalytic residues using Squidly and BLAST.                                                                  
                                                                                                                   
╭─ Arguments ─────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    fasta_file         TEXT             Full path to query fasta (note have simple IDs otherwise we'll remove  │
│                                          all funky characters.)                                                 │
│                                          [default: None]                                                        │
│                                          [required]                                                             │
│ *    esm2_model         TEXT             Name of the esm2_model, esm2_t36_3B_UR50D or esm2_t48_15B_UR50D        │
│                                          [default: None]                                                        │
│                                          [required]                                                             │
│      output_folder      [OUTPUT_FOLDER]  Where to store results (full path!) [default: Current Directory]       │
│      run_name           [RUN_NAME]       Name of the run [default: squidly]                                     │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ───────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --database                  TEXT     Full path to database csv (if you want to do the ensemble), needs 3        │
│                                      columns: 'Entry', 'Sequence', 'Residue' where residue is a | separated     │
│                                      list of residues. See default DB provided by Squidly.                      │
│                                      [default: None]                                                            │
│ --cr-model-as               TEXT     Optional: Model for the catalytic residue prediction i.e. not using the    │
│                                      default with the package. Ensure it matches the esmmodel.                  │
│ --lstm-model-as             TEXT     Optional: LSTM model path for the catalytic residue prediction i.e. not    │
│                                      using the default with the package. Ensure it matches the esmmodel.        │
│ --toks-per-batch            INTEGER  Run method (filter or complete) i.e. filter = only annotates with the next │
│                                      tool those that couldn't be found.                                         │
│                                      [default: 5]                                                               │
│ --as-threshold              FLOAT    Whether or not to keep multiple predicted values if False only the top     │
│                                      result is retained.                                                        │
│                                      [default: 0.99]                                                            │
│ --blast-threshold           FLOAT    Sequence identity with which to use Squidly over BLAST defualt 0.3         │
│                                      (meaning for seqs with < 0.3 identity in the DB use Squidly).              │
│                                      [default: 0.3]                                                             │
│ --install-completion                 Install completion for the current shell.                                  │
│ --show-completion                    Show completion for the current shell, to copy it or customize the         │
│                                      installation.                                                              │
│ --help                               Show this message and exit.                                                │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Data Availability
All datasets used in the paper are available here https://zenodo.org/records/15541320.

## Reproducing Squidly
We developed reproduction scripts for each benchmark training/test scenario.

- **AEGAN and Common Benchmarks**: Trained on Uni14230 (AEGAN), and tested on Uni3175 (AEGAN), HA_superfamily, NN, PC, and EF datasets.
- **CataloDB**: Trained on a curated training and test set with structural/sequence ID filtering to less than 30% identity.

The corresponding scripts can be found in the reproduction_run directory.

Before running them, download the datasets.zip file from zenodo and place them and unzip it in the base directory of Squidly.

Datasets:
https://zenodo.org/records/15541320

Model weights:
https://huggingface.co/WillRieger/Squidly

```bash
python reproduction_scripts/reproduce_squidly_CataloDB.py --scheme 2 --sample_limit 16000 --esm2_model esm2_t36_3B_UR50D --reruns 1
```

You must choose the pair scheme for the Squidly models:
<img src="pair_scheme_fig_.png" width=50%>

Scheme 2 and 3 had the sample limit parameter set to 16000, and scheme 1 at 4000000.

You must also correctly specify the ESM2 model used.
You can either use esm2_t36_3B_UR50D or esm2_t48_15B_UR50D. The scripts will automatically download these if specified like so.
You may also instead provide your own path to the models if you have them downloaded somewhere.

