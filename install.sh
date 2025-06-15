#!/bin/bash
conda  create --name squidly python=3.10.14 -y

# Doesn't like working with conda init
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate squidly

# Install torch
conda config --env --add channels conda-forge
conda install pytorch torchvision torchaudio -c pytorch -y

# install clustalomega
wget http://www.clustal.org/omega/clustalo-1.2.4-Ubuntu-x86_64 -O clustalo
chmod +x clustalo
echo export PATH=$PATH:$PWD/bin >> ~/.bashrc 
source ~/.bashrc

# install enzyme-tk and get the models from huggingface
pip install enzymetk
pip install huggingface_hub
python lib/download_models_hf.py

# downloading and using a BLAST database
# update_blastdb.pl --decompress --blastdb_version 5 swissprot
# ./diamond prepdb -d swissprot