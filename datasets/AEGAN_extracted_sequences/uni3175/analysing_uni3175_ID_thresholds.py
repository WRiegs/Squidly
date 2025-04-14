import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os



def get_AS_pos_from_uniprot(df):
    active_sites = []
    active_site_AA = []
    # iterate through the df and get the active sites
    for index, row in df.iterrows():
        active_site_string = row["Active site"]
        active_site_list = []
        intermediate_list = active_site_string.split(";")
        #iterate through the intermediate list and get the active sites
        for item in intermediate_list:
            if item.startswith("ACT_SITE") or item.startswith(" ACT_SITE"):
                active_site_list.append(int(item.split("ACT_SITE ")[1])-1)
        active_sites.append(active_site_list)
        
        # go through the active site list and extract the AA at the active site position
        AAs_in_active_site = []
        for active_site in active_site_list:
            AA = row["Sequence"][active_site]
            AAs_in_active_site.append(AA)
        active_site_AA.append(AAs_in_active_site)
        
    return active_sites, active_site_AA


def get_EC_TX_from_uniprot(df, tier = 2):
    EC_TX = []
    # iterate through the df and get the ECs up to the 2nd tier
    for index, row in df.iterrows():
        EC_string = row["EC number"]
        EC_list = []
        intermediate_list = EC_string.split(";")
        intermediate_list = [x.strip() for x in intermediate_list]
        intermediate_list = ['.'.join(x.split(".")[0:tier]) for x in intermediate_list]     # get the first 2 tiers of the EC number
        EC_list = list(dict.fromkeys(intermediate_list))    # remove duplicate strings in the list
        EC_TX.append(EC_list)
    return EC_TX

# get performance metrics


def main():
    
    path = "/scratch/project/squid/AEGAN_extracted_sequences/uni3175/big/labeled_test_sequences_results.tsv"
    df = pd.read_csv(path, sep='\t')
    active_sites, active_site_AA = get_AS_pos_from_uniprot(df)
    EC_TX = get_EC_TX_from_uniprot(df)
    
    # add the active sites and EC numbers to the df
    df["Active sites"] = active_sites
    df["Active site AA"] = active_site_AA
    df["EC_TX"] = EC_TX
    
    thresholds = ["0-30", "30-50", "50-80"]
    
    for threshold in thresholds:
        # group the df by the threshold in the Identity Range column
        threshold_df = df[df["Identity Range"] == threshold]
        
        # get the performance metrics by comparing the Squidly_CR_Position to the "Active sites" column
        
        Squidly_CR_Position = threshold_df["Squidly_CR_Position"]
        
        for index, row in threshold_df.iterrows():
            squidly_CR = row["Squidly_CR_Position"]
            # convert the prediction to a list of integers, but if empty, make it an empty list
            # if nan
            if type(squidly_CR) == float:
                squidly_CR = []
            else:
                squidly_CR = [int(x) for x in squidly_CR.split("|")]
            threshold_df.at[index, "Squidly_CR_Position"] = squidly_CR
        
        # get the true positives, false positives and false negatives
        TP = []
        FP = []
        FN = []
        for index, row in threshold_df.iterrows():
            active_sites = row["Active sites"]
            squidly_CR = row["Squidly_CR_Position"]
            TP.append(len(set(active_sites) & set(squidly_CR)))
            FP.append(len(set(squidly_CR) - set(active_sites)))
            FN.append(len(set(active_sites) - set(squidly_CR)))
            
        # calculate the f1 score, precision and recall
        f1 = []
        precision = []
        recall = []
        
        for i in range(len(TP)):
            if TP[i] == 0:
                f1.append(0) #  doesn't make sense
                precision.append(0)
                recall.append(0)
            else:
                precision.append(TP[i]/(TP[i] + FP[i]))
                recall.append(TP[i]/(TP[i] + FN[i]))
                f1.append(2*((precision[i]*recall[i])/(precision[i]+recall[i])))
        
        # now get the overall f1, precision and recall
        overall_f1 = sum(f1)/len(f1)
        overall_precision = sum(precision)/len(precision)
        overall_recall = sum(recall)/len(recall)
        
        # put it in a text file with the heading of the threshold
        with open(f"/scratch/project/squid/AEGAN_extracted_sequences/uni3175/big/{threshold}_performance_metrics.txt", "w") as file:
            file.write(f"Threshold: {threshold}\n")
            file.write(f"F1: {overall_f1}\n")
            file.write(f"Precision: {overall_precision}\n")
            file.write(f"Recall: {overall_recall}\n")
            file.write("\n")
            
            
if __name__ == "__main__":
    main()