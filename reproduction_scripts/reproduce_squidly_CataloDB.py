# python reproduction_scripts/reproduce_squidly_CataloDB.py --scheme 2 --sample_limit 12000 --esm2_model esm2_t36_3B_UR50D --reruns 1

import argparse
import pathlib
import os
import torch
import datetime
import random
import shutil
random.seed(420)


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract per-token representations and model outputs for sequences in a FASTA file"  # noqa
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
        #raise Exception("No GPU detected")
        print("No GPU detected")
        print("Running on CPU")
    
    # training datasets
    
    dataset_fasta = "datasets/CataloDB/fastas/train_and_test.fasta"
    dataset_metadata = "datasets/CataloDB/All_metadata.tsv"

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
    
    if args.esm2_model == "esm2_t48_15B_UR50D" or "esm2_t48_15B_UR50D" in args.esm2_model:
        layer = 48
        embedding_size = 5120
        model_name = "esm2_t48_15B_UR50D"
    elif args.esm2_model == "esm2_t36_3B_UR50D" or "esm2_t36_3B_UR50D" in args.esm2_model:
        layer = 36
        embedding_size = 2560
        model_name = "esm2_t36_3B_UR50D"
    
        # set up the output directory with todays date and the scheme/dataset
    output_dir = pathlib.Path(f"CataloDB_{args.scheme}_{model_name}_{datetime.datetime.now().strftime('%Y-%m-%d')}")
    if not output_dir.exists():
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")
    
    ESM_MODEL= args.esm2_model
    PSCHEME_OUT=f"{output_dir}/Scheme{args.scheme}_{args.sample_limit}/"
    EMB=f"{output_dir}/embeddings"
    EMB_REP=f"{PSCHEME_OUT}contrastive_rep"
    PAIR_SCHEME=f"{PSCHEME_OUT}/paired_embeddings_dataset.pt"
    META=f"{PSCHEME_OUT}metadata_paired.tsv"
    HPARAM="LSTM_Hyperparameters_bidirectional.json"
    EVAL="datasets/CataloDB/final_test_set_post_structural_filtering.txt"      
    LSTMOUT=f"{PSCHEME_OUT}LSTM/"
    MODEL=f"{PSCHEME_OUT}models/temp_best_model.pt"
    ASvBS="AS"
    ADDITIONAL_TEST="datasets/CataloDB/empty_extra_test_validation.txt"
    #benchmark datasets for the 6 common benchmarks based on unique SCOP families etc EF_families etc
    AEGAN_benchmark_dir = f"datasets/family_specific_embeddings_{model_name}/"
    AEGAN_benchmark_squid = f"{PSCHEME_OUT}family_specific_embeddings_squidly/"
            
    extracting_esm = f"python lib/extract_esm2.py {ESM_MODEL} {dataset_fasta} {EMB} --toks_per_batch 1000 --include per_tok"
    processing_pair_scheme = f"python lib/pair_schemes.py --embedding_dir {EMB} --metadata {dataset_metadata} --sample_limit {args.sample_limit} --scheme {args.scheme} --eval {EVAL} --test {ADDITIONAL_TEST} --out {PSCHEME_OUT} --layer {layer} --BSvAS {ASvBS} --create_torch"     
    training_CL_model = f"python lib/Squidly_CL_trainer.py --embedding_size {embedding_size} --pair_scheme {PAIR_SCHEME} --metadata {META} --out {PSCHEME_OUT}"
    converting_esm2_to_squidly = f"python lib/esm2squidly.py --model_location {MODEL} --embeddings_dir {EMB} --embedding_size {embedding_size} --out {EMB_REP} --layer {layer} --save_new_pt"
    converting_AEGAN_esm2_to_squidly = f"python lib/esm2squidly.py --model_location {MODEL} --embeddings_dir {AEGAN_benchmark_dir} --embedding_size {embedding_size} --out {AEGAN_benchmark_squid} --layer {layer} --save_new_pt"
    training_LSTM = f"python lib/LSTM_trainer.py --contrastive_representations {EMB_REP} --metadata {META} --evaluation_set {EVAL} --test_set {ADDITIONAL_TEST} --benchmark {AEGAN_benchmark_squid} --hyperparams {HPARAM} --output {LSTMOUT}"

    
    if args.reruns > 1:
        for i in range(args.reruns):
            # make the PSCHEME_OUT directory
            if not os.path.exists(PSCHEME_OUT):
                os.makedirs(PSCHEME_OUT)
            print(f"Rerun {i+1}")
            # if there's no embeddings dir setup already, then run the following -- ensures we don't generate the same embeddings multiple times
            if not os.path.exists(EMB):
                os.system(extracting_esm)
            os.system(processing_pair_scheme)
            os.system(training_CL_model)
            os.system(converting_esm2_to_squidly)
            os.system(converting_AEGAN_esm2_to_squidly)
            os.system(training_XGBoost)
            os.system(training_LSTM)
            # delete the squidly embeddings and pair scheme
            shutil.rmtree(EMB_REP)
            # move the output to a new directory so we save consecutive runs
            os.rename(PSCHEME_OUT, f"{PSCHEME_OUT[:-1]}_{i+1}/")
    else:
        # make the PSCHEME_OUT directory
        if not os.path.exists(PSCHEME_OUT):
            os.makedirs(PSCHEME_OUT)
        if not os.path.exists(EMB):
            os.system(extracting_esm)
        os.system(processing_pair_scheme)
        os.system(training_CL_model)
        os.system(converting_esm2_to_squidly)
        os.system(converting_AEGAN_esm2_to_squidly)
        os.system(training_LSTM)

    # remove the embeddings directory
    shutil.rmtree(EMB)
    
    return

if __name__ == '__main__':
    main()



