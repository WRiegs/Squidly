# Squidly 
<p align="center">
 <img width="400" alt="Squidly_mascot" src="https://github.com/user-attachments/assets/5ee9dd56-9397-4ff1-b4f6-b8a6982d40be" />
</p>


Squidly, is a tool that employs a biologically informed contrastive learning approach to accurately predict catalytic residues from enzyme sequences. We offer Squidly as ensembled with Blast to achieve high accuracy at low and high sequence homology settings.

<img width="3591" height="2254" alt="Panel_1_crisp_final_colours" src="https://github.com/user-attachments/assets/dd29a9a2-67aa-45d2-9d39-9c7d6a8e976a" />

If you use squidly in your work please cite our preprint: https://www.biorxiv.org/content/10.1101/2025.06.13.659624v1

Also if you have any issues installing, please post an issue! We have tested this on ubuntu.

### Requirements
Squidly is dependant on the ESM2 3B or 15B protein language model. Running Suidly will automatically attempt to download each model. The Smaller 3B model is lighter, runs faster and requires less VRAM. The 3B and 15B models expect roughly 25GB and 74GB of VRAM, respectively. Our tests on 100 sequences with an average length of 400+ took about 40 seconds and 3 minutes for the 3B and 15B model.

Squidly can only predict sequences of less than 1024 residues!

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
By default squidly runs the ensembled squidly model. It's optimised to run as fast as the single model.

For example to run the 3B model with a fasta file
```bash
squidly run example.fasta esm2_t36_3B_UR50D 
```

Squidly can also be further ensembled with BLAST (you need to pass the database as well)
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

A threshold can be selected for sequence identity between the query and the blast database, such that Squidly will be used under a certain threshold. This is because BLAST tends to outperform all currently available ML models at high sequence identities

Running a single Squidly model (non ensembled)
```bash 
squidly run example.fasta esm2_t36_3B_UR50D output_folder/ --single-model --cr-model-as squidly/models/3B/CataloDB_esm2_t36_3B_UR50D_CR_1.pt --lstm-model-as squidly/models/3B/CataloDB_esm2_t36_3B_UR50D_LSTM_1.pth
```

#### Squidly args 
```bash
 squidly --help

 Usage: squidly [OPTIONS] COMMAND [ARGS]...

╭─ Options ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion          Install completion for the current shell.                                                                                              │
│ --show-completion             Show completion for the current shell, to copy it or customize the installation.                                                       │
│ --help                        Show this message and exit.                                                                                                            │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ install   Install the models for the package.                                                                                                                        │
│ run       Find catalytic residues using Squidly and BLAST.                                                                                                           │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

squidly run --help
 Usage: squidly run [OPTIONS] FASTA_FILE ESM2_MODEL [OUTPUT_FOLDER] [RUN_NAME]

 Find catalytic residues using Squidly and BLAST.
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *    fasta_file         TEXT             Full path to query fasta (note have simple IDs otherwise we'll remove all funky characters.) [required]                                                                                                                                                                                                                       │
│ *    esm2_model         TEXT             Name of the esm2_model, esm2_t36_3B_UR50D or esm2_t48_15B_UR50D [required]                                                                                                                                                                                                                                                    │
│      output_folder      [OUTPUT_FOLDER]  Where to store results (full path!) [default: Current Directory]                                                                                                                                                                                                                                                              │
│      run_name           [RUN_NAME]       Name of the run [default: squidly]                                                                                                                                                                                                                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --single-model       --no-single-model             Whether or not to use single model instead of the ensemble. We recommend the ensemble. It is faster than the single model version. [default: no-single-model]                                                                                                                                                       │
│ --model-folder                            TEXT     Full path to the model folder.                                                                                                                                                                                                                                                                                      │
│ --database                                TEXT     Full path to database csv (if you want to do the ensemble), needs 3 columns: 'Entry', 'Sequence', 'Residue' where residue is a | separated list of residues. See default DB provided by Squidly. [default: None]                                                                                                    │
│ --cr-model-as                             TEXT     Contrastive learning model for the catalytic residue prediction when not using the ensemble. Ensure it matches the esm model.                                                                                                                                                                                       │
│ --lstm-model-as                           TEXT     LSTM model for the catalytic residue prediction when not using the ensemble. Ensure it matches the esm model.                                                                                                                                                                                                       │
│ --as-threshold                            FLOAT    When using the single squidly models, you must specify a prediction threshold. We found >0.9 to work best in practice, depending on the model. [default: 0.95]                                                                                                                                                      │
│ --blast-threshold                         FLOAT    Sequence identity with which to use Squidly over BLAST defualt 0.3 (meaning for seqs with < 0.3 identity in the DB use Squidly). [default: 0.3]                                                                                                                                                                     │
│ --chunk                                   INTEGER  Max chunk size for the dataset. This is useful for when running Squidly on >50000 sequences as memory is storing intermediate results during inference. [default: 0]                                                                                                                                                │
│ --mean-prob                               FLOAT    Mean probability threshold used in the ensemble. [default: 0.6]                                                                                                                                                                                                                                                     │
│ --mean-var                                FLOAT    Mean variance cutoff used in the ensemble. [default: 0.225]                                                                                                                                                                                                                                                         │
│ --filter-blast       --no-filter-blast             Only run on the ones that didn't have a BLAST residue. [default: filter-blast]                                                                                                                                                                                                                                      │
│ --help                                             Show this message and exit.                                                                                                                                                                                                                                                                                         │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

```

## Threshold selection:

The below figure showcases Squidly's performance on the CataloDB benchmark at varying thresholds for probability and variance in the ensemble model. 

<img width="4207" height="4241" alt="CataloDB_threshold_heatmap" src="https://github.com/user-attachments/assets/b98ceed4-5f84-4d7d-9196-fc73405792f1" />


Optimal thresholds for specific uses may vary. Lower probability thresholds have been found to work in practice when preicting certain EC numbers:
<img width="4193" height="1405" alt="perEC_threshold_figure" src="https://github.com/user-attachments/assets/0fe07343-e349-4e7d-9824-d0e767b8a66c" />

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

