from Bio import PDB
import pickle
import os
import argparse
import glob
from pathlib import Path
from tqdm import tqdm

def load_cif_structure(cif_file):
    parser = PDB.MMCIFParser()
    structure = parser.get_structure("protein", cif_file)
    return structure


def convert_to_pdb(dir):
    # get all the cif files in the directory, and make a pdb file for each and save it
    for file in os.listdir(dir):
        if file.endswith(".cif"):
            structure = load_cif_structure(os.path.join(dir, file))
            io = PDB.PDBIO()
            io.set_structure(structure)
            pdb_file = os.path.join(dir, file.replace(".cif", ".pdb"))
            io.save(pdb_file)


def assign_heatmap(pdb_file_path, heatmap_values, output_file_path):
    # open the PDB, read the lines, replacing the bfactor with the heatmap values in a new file
    with open(pdb_file_path, 'r') as f:
        lines = f.readlines()
        with open(output_file_path, 'w') as f:
            residue_score_index = 0
            for line in lines :
                if line.startswith("ATOM"):
                    # split the line
                    string_line = line
                    line = line.split(' ')
                    # remove empty strings
                    line = list(filter(None, line))
                    residue = line[3]
                    residue_score_index = int(line[5])-1
                    score = heatmap_values[residue_score_index]
                    
                    # Multiply by 100 to scale the score
                    scaled_score = score * 100

                    # Check if the scaled score is less than 1
                    if scaled_score < 1:
                        # Format with 3 decimal places for numbers less than 1, need to do this because of PDB formatting rules. Oongaboonga
                        formatted_score = f"{scaled_score:.3f}"
                    else:
                        # Format with 4 significant figures for other numbers
                        formatted_score = f"{scaled_score:.4g}"

                    string_line = string_line[:60]+' '+formatted_score+string_line[66:]
                    # write the line to the new file
                    f.write(string_line)
                    
                   
                   
def main():
    # load pkl file with the heatmap values
    with open("/scratch/project/squid/kari_seqs/BS_site_probs.pkl", 'rb') as f:
        heatmap_values = pickle.load(f)
        
    print(heatmap_values)    
    # use glob to get all the pdb files in chai
    dir = "/scratch/project/squid/Case_study2/literature A20/protein_pdbs"
    files = glob.glob(dir + "/*.pdb")
    files = ["/scratch/project/squid/kari_seqs/N608.pdb"] # just for kari
    
    for file in tqdm(files):
        entry = Path(file).stem
        entry = entry.split(".")[0]
        heatmap = heatmap_values[entry]
        output_file = file.replace(".pdb", "_heatmap.pdb")
        assign_heatmap(file, heatmap, output_file)
    
    
    
    return


if __name__ == "__main__":
    main()
    