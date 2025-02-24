import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
import xgboost as xgb
#from cuml.svm import SVC
#from cuml.neighbors import KNeighborsClassifier
from pathlib import Path
import glob
from tqdm import tqdm


def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Train a model on the TRAST data.')
    parser.add_argument('--rep_dir', type=Path, help='Path to the representations file.')
    parser.add_argument('--meta', type=Path, help='Path to the matadata file.')
    parser.add_argument('--eval', type=Path, help='Path to the eval set txt file describing the list of entries that are evaluation sequences.')
    parser.add_argument('--test', type=str, default=None, help='Path to the file containing the list of sequences to use as a test set')
    parser.add_argument('--output', type=Path, help='Path to the output file.')
    return parser.parse_args()


def get_evaluation_metrics(actuals, predictions):
    # Calculate evaluation metrics
    TP = sum([1 for i in range(len(actuals)) if actuals[i] == 1 and predictions[i] == 1])
    TN = sum([1 for i in range(len(actuals)) if actuals[i] == 0 and predictions[i] == 0])
    FP = sum([1 for i in range(len(actuals)) if actuals[i] == 0 and predictions[i] == 1])
    FN = sum([1 for i in range(len(actuals)) if actuals[i] == 1 and predictions[i] == 0])

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

    return accuracy, precision, recall, f1


def get_cm_plot(actuals, predictions, save_location):
    cm = confusion_matrix(actuals, predictions)
    
    # clear old plots, initialise new one
    plt.clf()
    
    # plot the matrix
    sns.heatmap(cm, annot=True, fmt='g')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_location + 'confusion_matrix.png')
    

def get_ROC_curve(actuals, predictions, save_location):
    fpr, tpr, _ = roc_curve(actuals, predictions)
    plt.clf()
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig(save_location + 'ROC_curve.png')
    
    AUC_ROC = np.trapz(tpr, fpr)
    return AUC_ROC


def get_PR_curve(actuals, predictions, save_location):
    precision, recall, _ = precision_recall_curve(actuals, predictions)
    plt.clf()
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.savefig(save_location + 'PR_curve.png')
    
    AUC_PR = np.trapz(precision, recall)
    return AUC_PR
    
    
def get_best_threshold(actuals, predictions):
    best_threshold = 0
    best_f1 = 0
    for threshold in range(0, 101):
        threshold = threshold / 100
        predictions_binary = [1 if prediction > threshold else 0 for prediction in predictions]
        accuracy, precision, recall, f1 = get_evaluation_metrics(actuals, predictions_binary)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1    

    
def evaluate_model(model, actuals, X_test, save_location):
    # make dir for save_location
    Path(save_location).mkdir(parents=True, exist_ok=True)
    
    predictions = model.predict(X_test)
    
    best_threshold, best_f1 = get_best_threshold(actuals, predictions)
    
    AUC_ROC = get_ROC_curve(actuals, predictions, save_location)
    AUC_PR = get_PR_curve(actuals, predictions, save_location)
    
    # covert the predictions to binary predictions using the threshold
    predictions = [1 if prediction > best_threshold else 0 for prediction in predictions]
    
    print(len(actuals),len(predictions))
    
    results = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})
    results.to_csv(save_location + 'results.tsv', sep='\t', index=False)

    # Save evaluation metrics
    accuracy, precision, recall, f1 = get_evaluation_metrics(actuals, predictions)
    with open(save_location + 'evaluation_metrics.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall: {recall:.4f}\n')
        f.write(f'F1: {f1:.4f}\n')
        f.write(f'best_threshold: {str(best_threshold)}\n')
        f.write(f'AUC_ROC: {AUC_ROC:.4f}\n')
        f.write(f'AUC_PR: {AUC_PR:.4f}\n')
    
    get_cm_plot(actuals, predictions, save_location)



def get_AS_pos_from_uniprot(df):
    active_sites = []
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
    return active_sites


def main():
    args = argparser()
    
    # Load the metadata
    meta = pd.read_csv(args.meta, sep='\t')
    
    print(len(meta))
    
    Active_sites = get_AS_pos_from_uniprot(meta)
    
    print(Active_sites)
    
    # process the Active sites data into lists of integers
    #for i , row in meta.iterrows():
    #    AS = row['Active sites']
    #    if AS == '[]':
    #        meta.at[i, 'Active sites'] = []
    #    else:
    #        AS = AS[1:-1]
    #        AS = AS.split(',')
    #        AS = [x.strip() for x in AS]
    #        AS = [int(x) for x in AS]
    #        meta.at[i, 'Active sites'] = AS
    
    meta['Active sites'] = Active_sites
    
    # load the eval txt
    with open(args.eval, 'r') as f:
        eval_list = f.readlines()
        
    eval_list = [x.strip() for x in eval_list]
    
    print(eval_list)
            
    # Load the representations
    rep_files = glob.glob(str(args.rep_dir) + '/*.pt')
    
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    
    print("Loading the representations...")
    for rep_file in tqdm(rep_files):
        rep = torch.load(rep_file)
        rep = rep['representations'][48]
        # get AS from meta
        id = rep_file.split('/')[-1].split('|')[1]
        
        if id in eval_list:
            rep_meta = meta[meta['Entry'] == id]
            AS = list(rep_meta['Active sites'])[0]
            index = 0
            for x in range(len(rep)):
                X_test.append(rep[x].cpu().detach().numpy())
                if index in AS:
                    y_test.append(1)
                else:
                    y_test.append(0)
                index += 1
        else:
            rep_meta = meta[meta['Entry'] == id]
            AS = list(rep_meta['Active sites'])[0]
            index = 0
            for x in range(len(rep)):
                X_train.append(rep[x].cpu().detach().numpy())
                if index in AS:
                    y_train.append(1)
                else:
                    y_train.append(0)
                index += 1
    
    # turn X_train and X_test into 2D numpy matrix
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    # print shape of X_train
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    
    # prepare the xgboost
    # Convert the dataset into DMatrix, which is a data structure optimized for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set parameters for training with GPU
    params = {
        'tree_method': 'hist', 
        'objective': 'binary:logistic',  # For binary classification
        'eval_metric': 'error',
        'device': 'cuda'
    }

    # Train the model
    bst = xgb.train(params, dtrain, num_boost_round=100)

    evaluate_model(bst, y_test, dtest, str(args.output / '_XGBoost'))
    
    # save the model
    bst.save_model(str(args.output / '_XGBoost/model.json'))
    
    # Initialize and train the SVM classifier using cuML
    #svc = SVC(kernel='rbf', C=1.0)
    #svc.fit(X_train, y_train)
    
    #evaluate_model(svc, y_test, X_test, args.output+'_SVM')
    

if __name__ == '__main__':
    main()