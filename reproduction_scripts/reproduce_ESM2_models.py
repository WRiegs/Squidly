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

    if args.reruns:
        print(f"Number of reruns: {args.reruns}")
        
    # set up the output directory with todays date and the scheme/dataset
    output_dir = pathlib.Path(f"reproduction_runs/esm2/reproducing_ESM2_models_dataset_{args.dataset}_{datetime.datetime.now().strftime('%Y-%m-%d')}")
    if not output_dir.exists():
        os.makedirs(output_dir)
    print(f"Output directory: {output_dir}")
        
    redundancy_threshold = 0.9
    experimental_df = "datasets/dataset_1.tsv" # Used to help generate the evaluation set
    eval_num = 250
    
    ESM_MODEL="esm2_t36_3B_UR50D"
    PSCHEME_OUT=f"{output_dir}/ESM2raw/"
    EMB=f"{output_dir}/embeddings"
    EMB_REP=f"{PSCHEME_OUT}contrastive_rep"
    PAIR_SCHEME=f"{PSCHEME_OUT}/paired_embeddings_dataset.pt"
    META=f"/scratch/project/squid/code_modular/reproduction_runs/esm2/metadata_paired.tsv"
    HPARAM="/scratch/project/squid/code/ESM2_Hyperparameters_bidirectional.json"
    EVAL=f"{PSCHEME_OUT}Low30_mmseq_ID_250_exp_subset.txt"
    TEST=f"{PSCHEME_OUT}filt90_TEST_set.txt"
    LSTMOUT=f"{PSCHEME_OUT}LSTM/"
    MODEL=f"{PSCHEME_OUT}models/temp_best_model.pt"
    ASvBS="AS"
    
    
    #parser.add_argument('--metadata', type=str, help='Metadata tsv file from uniprot, containing act sites')
    #parser.add_argument('--emb_dir', type=str, help='Directory containing embeddings', required=True)
    #parser.add_argument('--emb_layer', type=int, help='Layer of embeddings to use', required=True)
    #parser.add_argument('--max_len', type=int, help='Max length of the sequences', required=True)
    #parser.add_argument('--out', type=str, help='Output_tag_file_identifier', required=True)
        
    filtering = f"python lib/Redundancy_and_eval_set_filtering.py --fasta {dataset_fasta} --out {PSCHEME_OUT} --num_samples {eval_num} --experimental_data {experimental_df} --redundancy_threshold {redundancy_threshold}"
    extracting_esm = f"python lib/extract_esm2.py {ESM_MODEL} {PSCHEME_OUT}/0.9filtered_for_redundancy.fasta {EMB} --toks_per_batch 10 --include per_tok"
    processing_the_torch = f"python lib/ESM2_prep_tensors.py --metadata {dataset_metadata} --emb_dir {EMB} --emb_layer {36} --max_len {1024} --out {PSCHEME_OUT}"
    LSTM_training = f"python lib/ESM2_LSTM_trainer.py --embeddings {PSCHEME_OUT}padded_tensor.pt --targets {PSCHEME_OUT}padded_targets_index0.pt --metadata {PSCHEME_OUT}metadata_paired.tsv --evaluation_set {EVAL} --output {PSCHEME_OUT} --hyperparams {HPARAM}"

    
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



