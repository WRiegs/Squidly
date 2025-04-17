# python /scratch/project/squid/code_modular/reproduce_ESM2_models_AEGAN.py --esm2_model esm2_t36_3B_UR50D --reruns 5

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
        "esm2_model",
        type=str,
        help="Name of the ESM2 model",
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
    
    dataset_fasta = "/scratch/project/squid/AEGAN_extracted_sequences/train_test/uni3175_uni14230_unduplicated.fasta"
    dataset_metadata = "/scratch/project/squid/AEGAN_extracted_sequences/train_test/uni3175_uni14230_unduplicated.tsv"
        
    if args.reruns:
        print(f"Number of reruns: {args.reruns}")
    
    if args.esm2_model == "esm2_t48_15B_UR50D":
        args.esm2_model="/scratch/project/squid/models/ESM2/esm2_t48_15B_UR50D.pt"
        layer = 48
        embedding_size = 5120
        model_name = "esm2_t48_15B_UR50D"
    elif args.esm2_model == "esm2_t36_3B_UR50D":
        layer = 36
        embedding_size = 2560
        model_name = "esm2_t36_3B_UR50D"

    if args.reruns:
        print(f"Number of reruns: {args.reruns}")
        
    # set up the output directory with todays date and the scheme/dataset
    output_dir = pathlib.Path(f"reproduction_runs/esm2/reproducing_ESM2_models_AEGAN_{datetime.datetime.now().strftime('%Y-%m-%d')}")
    if not output_dir.exists():
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")
            
    PSCHEME_OUT=f"{output_dir}/ESM2raw/"
    HPARAM="/scratch/project/squid/code/ESM2_Hyperparameters_bidirectional.json"
    ESM_MODEL= args.esm2_model
    EMB=f"{output_dir}/embeddings"
    EVAL=f"/scratch/project/squid/AEGAN_extracted_sequences/uni3175/uni3175_unduplicated_entries.txt"      
    AEGAN_leaked_entries = "/scratch/project/squid/AEGAN_extracted_sequences/train_test/overlapping_entries.txt"
        
    #filtering = f"python lib/Redundancy_and_eval_set_filtering.py --fasta {dataset_fasta} --out {PSCHEME_OUT} --num_samples {eval_num} --experimental_data {experimental_df} --redundancy_threshold {redundancy_threshold}"
    extracting_esm = f"python lib/extract_esm2.py {ESM_MODEL} {dataset_fasta} {EMB} --toks_per_batch 10 --include per_tok"
    processing_the_torch = f"python lib/ESM2_prep_tensors.py --metadata {dataset_metadata} --emb_dir {EMB} --emb_size {embedding_size} --emb_layer {layer} --max_len {1024} --out {PSCHEME_OUT}"
    LSTM_training = f"python lib/ESM2_LSTM_trainer.py --embeddings {PSCHEME_OUT}padded_tensor.pt --emb_size {embedding_size} --targets {PSCHEME_OUT}padded_targets_index0.pt --metadata {PSCHEME_OUT}metadata_paired.tsv --evaluation_set {EVAL} --output {PSCHEME_OUT} --hyperparams {HPARAM} --leaked_entries {AEGAN_leaked_entries}"

    
    if args.reruns > 1:
        for i in range(args.reruns):
            # make the PSCHEME_OUT directory
            if not os.path.exists(PSCHEME_OUT):
                os.makedirs(PSCHEME_OUT)
            print(f"Rerun {i+1}")
            #os.system(filtering)
            # if there's no embeddings dir setup already, then run the following -- ensures we don't generate the same embeddings multiple times
            if not os.path.exists(EMB):
                if model_name == "esm2_t48_15B_UR50D":
                    # copy it over to the new directory
                    shutil.copytree("/scratch/project/squid/code_modular/dirty_reproducing_AEGAN_benchmark_squidly_scheme_2_esm2_t48_15B_UR50D_2025-01-11/embeddings", EMB)
                else:
                    os.system(extracting_esm)
            os.system(processing_the_torch)
            os.system(LSTM_training)
            # move the output to a new directory so we save consecutive runs
            os.rename(PSCHEME_OUT, f"{PSCHEME_OUT[:-1]}_{i+1}/")
            
            # remove any file starting with 'padded' in the ESM2raw directory
            for file in os.listdir(f"{PSCHEME_OUT[:-1]}_{i+1}/"):
                if file.startswith("padded"):
                    print(f"Removing {file}")
                    os.remove(f"{PSCHEME_OUT[:-1]}_{i+1}/{file}")
    else:
        # make the PSCHEME_OUT directory
        if not os.path.exists(PSCHEME_OUT):
            os.makedirs(PSCHEME_OUT)
        os.system(filtering)
        if not os.path.exists(EMB):
            os.system(extracting_esm)
        os.system(processing_the_torch)
        os.system(LSTM_training)


    # remove the EMB directory
    #os.system(f"rm -r {EMB}")
    

if __name__ == '__main__':
    main()



