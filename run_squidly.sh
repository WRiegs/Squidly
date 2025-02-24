#!/bin/bash

# Default values
TEST_NAME="MDH_ALL"
DATA_DIR="/scratch/project/squid/contrastive_learning/tests/"
CPU=1
GPU="h100"
MEM=400G

FILE="/scratch/project/squid/code_modular/PREDICTING_UNIPROT/All_SwissPro_AS.tsv"
FILE="/scratch/project/squid/code_modular/speed_tests/random_1000_seqs_for_speed_testsfiltered_LSTM.tsv"
FILE="/scratch/project/squid/AEGAN_extracted_sequences/train_test/random_1000_seqs_for_speed_tests.fasta"
FILE="/scratch/project/squid/code_modular/speed_tests/random_4_seqs.tsv"
FILE="/scratch/project/squid/code_modular/datasets/dataset_1.tsv"


#CARE
#FILE="/scratch/project/squid/CARE/protein_train.csv"
FILE="/scratch/project/squid/CARE/30_protein_test.csv"
#FILE="/scratch/project/squid/CARE/30-50_protein_test.csv"

# Small model
#TIME="04:00:00"
#CR_MODEL_AS="/scratch/project/squid/code_modular/dirty_reproducing_AEGAN_benchmark_squidly_scheme_2_esm2_t36_3B_UR50D_2025-01-16/Scheme2_12000_1/models/temp_best_model.pt"
#LSTM_MODEL_AS="/scratch/project/squid/code_modular/dirty_reproducing_AEGAN_benchmark_squidly_scheme_2_esm2_t36_3B_UR50D_2025-01-16/Scheme2_12000_1/LSTM/models/16-01-25_03-58_128_2_0.2_400_best_model.pth"
#ESM2_MODEL="esm2_t36_3B_UR50D"
#OUT="/scratch/project/squid/code_modular/speed_tests/med_test_1000"
#OUT="/scratch/project/squid/CARE/squidly_predicted_newer_models"

# Big model
TIME="07:00:00"
CR_MODEL_AS="/scratch/project/squid/code_modular/SQUIDLY_YABOMBACLAT_WATCHOUTFORTHEROAD_2_esm2_t48_15B_UR50D_2025-02-07/Scheme2_24000_1/models/temp_best_model.pt"
LSTM_MODEL_AS="/scratch/project/squid/code_modular/SQUIDLY_YABOMBACLAT_WATCHOUTFORTHEROAD_2_esm2_t48_15B_UR50D_2025-02-07/Scheme2_24000_1/LSTM/models/07-02-25_18-06_128_2_0.2_400_best_model.pth"
ESM2_MODEL="esm2_t48_15B_UR50D"
OUT="/scratch/project/squid/CARE/squidly_predicted_newer_big_mean_emb"

TOKS_PER_BATCH=10
AS_THRESHOLD=0.90

# Defining output files..
OUTPUT="${DATA_DIR}${TEST_NAME}_slurm_output.txt"
ERROR="${DATA_DIR}${TEST_NAME}_slurm_error.txt"

sbatch <<- EOS
	#!/bin/bash
	#SBATCH --nodes=1
	#SBATCH --job-name=$TEST_NAME
	#SBATCH --ntasks-per-node=1
	#SBATCH --cpus-per-task=$CPU
	#SBATCH --mem=$MEM
	#SBATCH --time=$TIME
	#SBATCH -o $OUTPUT
	#SBATCH -e $ERROR
	#SBATCH --partition=gpu_cuda
	#SBATCH --gres=gpu:$GPU:1
	#SBATCH --qos=gpu
	#SBATCH --begin=now
	#SBATCH --account=a_boden

	module load anaconda3/2023.09-0
	source $EBROOTANACONDA3/etc/profile.d/conda.sh
	conda activate ASTformer

	python /scratch/project/squid/CARE/code/extracting_ESM_ASBS.py ${FILE} ${ESM2_MODEL} ${CR_MODEL_AS} ${LSTM_MODEL_AS} ${OUT} --toks_per_batch ${TOKS_PER_BATCH} --AS_threshold ${AS_THRESHOLD} --monitor --store_hidden_reps
EOS
