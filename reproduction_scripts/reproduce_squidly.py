# python reproduce_squidly.py --scheme 2 --sample_limit 4000 --esm2_model esm2_t36_3B_UR50D --reruns 5

import argparse
import pathlib
import os
import torch
import datetime
import random
random.seed(420)



def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
    )
    parser.add_argument(
        "--dataset",
        type=int,
        choices=[1, 2, 3],
        help="Dataset to use for contrastive learning. 1 = experimental data, 2 = exp data + reviewed sequences with known structures, 3 = exp data + all reviewed sequences with annotations",
    )   
    parser.add_argument(
        "--scheme", 
        type=int, 
        help="Scheme for contrastive learning pair mining.",
    )
    parser.add_argument(
        "--sample_limit", 
        type=int, 
        help="Limit the number of pairs in the pair mining",
    )
    parser.add_argument(
        "--esm2_model",
        type=str,
        help="Path or name of the ESM2 model",
        default="esm2_t36_3B_UR50D"
    )
    parser.add_argument(
        "--reruns",
        type=int,
        help="Number of reruns. Default is 1",
        default=1
    )
    return parser



def main():
    # parse args 
    parser = create_parser()
    args = parser.parse_args()
    
    # check for a GPU
    if torch.cuda.is_available():
        print("GPU available")
    else:
        raise Exception("No GPU detected")
    
    if args.dataset == 1:
        print("Dataset 1 selected")
        dataset_fasta = "datasets/dataset_1.fasta"
        dataset_metadata = "datasets/dataset_1.tsv"
    elif args.dataset == 2:
        print("Dataset 2 selected")
        dataset_fasta = "datasets/dataset_2.fasta"
        dataset_metadata = "datasets/dataset_2.tsv"
    elif args.dataset == 3:
        print("Dataset 3 selected")
        dataset_fasta = "datasets/dataset_3.fasta"
        dataset_metadata = "datasets/dataset_3.tsv"
        
        
    if args.scheme == 1:
        print("Scheme 1 selected")
    elif args.scheme == 2:
        print("Scheme 2 selected")
    elif args.scheme == 3:
        print("Scheme 3 selected")
        
    if args.sample_limit:
        print(f"Sample limit: {args.sample_limit}")
        
    if args.reruns:
        print(f"Number of reruns: {args.reruns}")
        
    # set up the output directory with todays date and the scheme/dataset
    output_dir = pathlib.Path(f"reproducing_squidly_scheme_{args.scheme}_dataset_{args.dataset}_{args.esm2_model}_{datetime.datetime.now().strftime('%Y-%m-%d')}")
    if not output_dir.exists():
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")
        
    redundancy_threshold = 0.9
    experimental_df = "datasets/dataset_1.tsv" # Used to help generate the evaluation set
    eval_num = 250
    
    if args.esm2_model == "esm2_t48_15B_UR50D":
        args.esm2_model="/scratch/project/squid/models/ESM2/esm2_t48_15B_UR50D.pt"
        layer = 48
        embedding_size = 5120
    elif args.esm2_model == "esm2_t36_3B_UR50D":
        layer = 36
        embedding_size = 2560
    
    ESM_MODEL= args.esm2_model
    PSCHEME_OUT=f"{output_dir}/Scheme{args.scheme}_{args.sample_limit}/"
    EMB=f"{output_dir}/embeddings"
    EMB_REP=f"{PSCHEME_OUT}contrastive_rep"
    PAIR_SCHEME=f"{PSCHEME_OUT}/paired_embeddings_dataset.pt"
    META=f"{PSCHEME_OUT}metadata_paired.tsv"
    HPARAM="LSTM_Hyperparameters_bidirectional.json"
    EVAL=f"{PSCHEME_OUT}Low30_mmseq_ID_250_exp_subset.txt"
    TEST=f"{PSCHEME_OUT}filt90_TEST_set.txt"
    LSTMOUT=f"{PSCHEME_OUT}LSTM/"
    MODEL=f"{PSCHEME_OUT}models/temp_best_model.pt"
    ASvBS="AS"
        
    filtering = f"python lib/Redundancy_and_eval_set_filtering.py --fasta {dataset_fasta} --out {PSCHEME_OUT} --num_samples {eval_num} --experimental_data {experimental_df} --redundancy_threshold {redundancy_threshold}"
    extracting_esm = f"python lib/extract_esm2.py {ESM_MODEL} {PSCHEME_OUT}/0.9filtered_for_redundancy.fasta {EMB} --toks_per_batch 10 --include per_tok"
    processing_pair_scheme = f"python lib/pair_schemes.py --embedding_dir {EMB} --metadata {dataset_metadata} --sample_limit {args.sample_limit} --scheme {args.scheme} --eval {EVAL} --test {TEST} --out {PSCHEME_OUT} --layer {layer} --BSvAS {ASvBS} --create_torch"     
    training_CL_model = f"python lib/Squidly_CL_trainer.py --embedding_size {embedding_size} --pair_scheme {PAIR_SCHEME} --metadata {META} --out {PSCHEME_OUT}"
    converting_esm2_to_squidly = f"python lib/esm2squidly.py --model_location {MODEL} --embeddings_dir {EMB} --embedding_size {embedding_size} --out {EMB_REP} --layer {layer} --save_new_pt"
    training_XGBoost = f"python lib/XGBoost_trainer.py --rep_dir {EMB_REP} --meta {META} --eval {EVAL} --test {TEST} --output {PSCHEME_OUT}" 
    training_LSTM = f"python lib/LSTM_trainer.py --contrastive_representations {EMB_REP} --metadata {META} --evaluation_set {EVAL} --test_set {TEST} --hyperparams {HPARAM} --output {LSTMOUT}"

    
    if args.reruns > 1:
        for i in range(args.reruns):
            # make the PSCHEME_OUT directory
            if not os.path.exists(PSCHEME_OUT):
                os.makedirs(PSCHEME_OUT)
            print(f"Rerun {i+1}")
            os.system(filtering)
            # if there's no embeddings dir setup already, then run the following -- ensures we don't generate the same embeddings multiple times
            if not os.path.exists(EMB):
                os.system(extracting_esm)
            os.system(processing_pair_scheme)
            os.system(training_CL_model)
            os.system(converting_esm2_to_squidly)
            os.system(training_XGBoost)
            os.system(training_LSTM)
            # delete the squidly embeddings and pair scheme
            os.rmdir(EMB_REP)
            # move the output to a new directory so we save consecutive runs
            os.rename(PSCHEME_OUT, f"{PSCHEME_OUT[:-1]}_{i+1}/")
    else:
        # make the PSCHEME_OUT directory
        if not os.path.exists(PSCHEME_OUT):
            os.makedirs(PSCHEME_OUT)
        os.system(filtering)
        if not os.path.exists(EMB):
            os.system(extracting_esm)
        os.system(processing_pair_scheme)
        os.system(training_CL_model)
        os.system(converting_esm2_to_squidly)
        os.system(training_XGBoost)
        os.system(training_LSTM)

    # remove the embeddings directory
    os.rmdir(EMB)
    
    return

if __name__ == '__main__':
    main()



