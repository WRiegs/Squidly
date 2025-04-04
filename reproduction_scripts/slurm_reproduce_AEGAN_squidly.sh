#!/bin/bash

# Default values
TIME="18:00:00"
DATA_DIR="/scratch/project/squid/contrastive_learning/tests/"
CPU=1
GPU="h100"
SLIM=16000
SCHEME=2
REPS=6
ESM2_MODEL="esm2_t48_15B_UR50D"
ASvBS="AS"
#ESM2_MODEL="esm2_t36_3B_UR50D"

TEST_NAME="Repro_Scheme_${SCHEME}_SLIM_${SLIM}_${ESM2_MODEL}_$(date +"%Y-%m-%d_%H-%M-%S")"

# Defining output files...
OUTPUT="${DATA_DIR}${TEST_NAME}_slurm_output.txt"
ERROR="${DATA_DIR}${TEST_NAME}_slurm_error.txt"

#python /scratch/project/squid/code/TRAST_training_contr_model.py --pair_scheme ${PAIR_SCHEME} --metadata ${META} --out ${PSCHEME_OUT}
#python /scratch/project/squid/code/TRAST_emb2rep.py --model_location ${MODEL} --embeddings_dir ${EMB} --out ${EMB_REP} --save_new_pt
#python /scratch/project/squid/code/Ohne_seq_classifier.py --rep_dir ${EMB_REP} --meta ${META} --eval ${EVAL} --test ${TEST} --output ${PSCHEME_OUT}
	

sbatch <<- EOS
	#!/bin/bash
	#SBATCH --nodes=1
	#SBATCH --job-name=$TEST_NAME
	#SBATCH --ntasks-per-node=1
	#SBATCH --cpus-per-task=$CPU
	#SBATCH --mem=400G
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
	module load mmseqs2

    python reproduce_squidly_AEGAN_benchmark.py --scheme $SCHEME --sample_limit $SLIM --reruns $REPS --esm2_model $ESM2_MODEL --ASvBS $ASvBS
	
EOS
