import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse
import wandb
from helper.generate_plot import get_auroc_by_groups, get_labels
from sklearn.preprocessing import LabelEncoder
from datasets.dataloader import prepare_dataloaders
from models.build_model import DenseNet_Model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import pandas as pd

def generate_race_roc_score(weight, race_loader, device, labels, test_df, avg_method):

     test_model = DenseNet_Model(weights=None, out_feature=4)

     race_all_labels, race_all_preds, all_ids = get_labels(race_loader, weight, device, test_model, False)
     num_classes = race_all_preds.shape[1]
     race_all_labels = label_binarize(race_all_labels, classes=list(range(num_classes)))

     auc_records = []
     for i in range(num_classes):
          y_true = race_all_labels[:, i]
          y_pred = race_all_preds[:, i]
          try:
              auc = roc_auc_score(y_true, y_pred, average=avg_method)
          except ValueError:
              auc = float('nan')   
          auc_records.append({
                'Race': labels[i],
                'AUROC': auc,
          })

     all_auc = roc_auc_score(race_all_labels, race_all_preds, average=avg_method)
     auc_records.append({
                'Race': 'all',
                'AUROC': all_auc,
          })

     auc_df = pd.DataFrame(auc_records)
     return auc_df



def generate_tabel(race_weights, race_lung_weights, race_clahe_weights, weights, lung_weights, clahe_weights, device, test_df, base_dir, external_ood_test):

    label_encoder = LabelEncoder()
    test_df["race_encoded"] = label_encoder.fit_transform(test_df["race"])
    race_labels = label_encoder.classes_
    labels = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion"]


    diag_loader = prepare_dataloaders(
                    images_path= test_df["Path"],
                    labels= test_df[labels].values,
                    dataframe= test_df,
                    masked= False,
                    clahe= False,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=True,
                    external_ood_test=external_ood_test
           )

    diag_lung_loader = prepare_dataloaders(
                    images_path= test_df["Path"],
                    labels= test_df[labels].values,
                    dataframe= test_df,
                    masked= True,
                    clahe= False,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=True,
                    external_ood_test=external_ood_test
           )
    diag_clahe_loader = prepare_dataloaders(
                    images_path= test_df["Path"],
                    labels= test_df[labels].values,
                    dataframe= test_df,
                    masked= False,
                    clahe= True,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=True,
                    external_ood_test=external_ood_test
           )

    race_loader = prepare_dataloaders(
                    images_path= test_df["Path"],
                    labels= test_df['race_encoded'].values,
                    dataframe= test_df,
                    masked= False,
                    clahe= False,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=False,
                    external_ood_test=external_ood_test
    )
    race_lung_loader = prepare_dataloaders(
                    images_path= test_df["Path"],
                    labels= test_df['race_encoded'].values,
                    dataframe= test_df,
                    masked= True,
                    clahe= False,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=False,
                    external_ood_test=external_ood_test
    )
    race_clahe_loader = prepare_dataloaders(
                    images_path= test_df["Path"],
                    labels= test_df['race_encoded'].values,
                    dataframe= test_df,
                    masked= False,
                    clahe= True,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=False,
                    external_ood_test=external_ood_test
    )

    race_auc_df = generate_race_roc_score(race_weights, race_loader, device, race_labels, test_df, 'macro')
    race_auc_lung_df = generate_race_roc_score(race_lung_weights, race_lung_loader, device, race_labels, test_df, 'macro')
    race_auc_clahe_df = generate_race_roc_score(race_clahe_weights, race_clahe_loader, device, race_labels, test_df, 'macro')
    
    auc_df = get_auroc_by_groups(diag_loader, weights, device, labels, test_df, 'macro', True)
    auc_lung_df = get_auroc_by_groups(diag_lung_loader, lung_weights, device, labels, test_df, 'macro', True)
    auc_clahe_df = get_auroc_by_groups(diag_clahe_loader, clahe_weights, device, labels, test_df, 'macro', True)

    auc_df['Preprocessing'] = 'Baseline'
    auc_lung_df['Preprocessing'] = 'Lung Masking'
    auc_clahe_df['Preprocessing'] = 'CLAHE'

    disease_auroc_df = pd.concat([auc_df, auc_lung_df, auc_clahe_df], ignore_index=True)

    summary_df = disease_auroc_df.groupby('Race')['AUROC'].agg(
                   Macro_AUROC='mean',
                   Max_Diff_AUROC=lambda x: x.max() - x.min()
               ).reset_index()


    disparities = (disease_auroc_df.groupby(['Preprocessing', 'Disease', 'Race'])['AUROC']
          .agg(lambda x: x.max() - x.min())
          .reset_index(name='Gap')
    )

    avg_disparities = disparities.groupby(['Preprocessing', 'Race'])['Gap'].mean().reset_index()

    
    avg_diff_str = avg_disparities.to_latex(index=False, 
                                  label="tab:avg_auroc_diff_diagnostic_summary")

    diff_str = disparities.to_latex(index=False, 
                                  label="tab:auroc_diff_diagnostic_summary")

    print(diff_str)


    disease_auroc_df = disease_auroc_df.pivot_table(index=['Race', 'Disease'], columns=['Preprocessing'], values='AUROC')
    disease_auroc_df = disease_auroc_df.reset_index()

    race_auc_df['Preprocessing'] = 'Baseline'
    race_auc_lung_df['Preprocessing'] = 'Lung Masking'
    race_auc_clahe_df['Preprocessing'] = 'CLAHE'

    race_auroc_df = pd.concat([race_auc_df, race_auc_lung_df, race_auc_clahe_df], ignore_index=True)
               
    diff_latex_str = summary_df.to_latex(index=False, 
                                  label="tab:auroc_diagnostic_summary")
    
    
    race_latex_str = race_auroc_df.to_latex(index=False, 
                                  caption="Summary of AUROC per Disease and Race groupBy metrics across preprocessing methods", 
                                  label="tab:auroc_race_summary")
    
    if external_ood_test:
        output_diff_path = "/deep_learning/output/Sutariya/main/mimic/avg_diff_auc_result_tabel_external.tex"
        output_path = "/deep_learning/output/Sutariya/main/mimic/daignostic_result_tabel_external.tex"
        race_output_path = "/deep_learning/output/Sutariya/main/mimic/race_encoding_table_external.tex"
    else:
        output_diff_path = "/deep_learning/output/Sutariya/main/mimic/avg_diff_auc_result_tabel.tex"
        output_path = "/deep_learning/output/Sutariya/main/mimic/daignostic_result_tabel.tex"
        race_output_path = "/deep_learning/output/Sutariya/main/mimic/race_encoding_table.tex"


    with open(output_path, "w") as f:
         f.write(diff_latex_str)
    with open(output_diff_path, "w") as f:
         f.write(avg_diff_str)
    with open(race_output_path, "w") as f:
         f.write(race_latex_str)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Arguments")

    parser.add_argument('--random_state', type=int, default=100)
    parser.add_argument('--dataset', type=str, default="mimic")
    parser.add_argument('--external_ood_test', action='store_true')
    args = parser.parse_args()
    print(vars(args))
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = args.random_state
    dataset = args.dataset
    external_ood_test = args.external_ood_test

    name = f"model_baseline_{dataset}_{random_state}"
    if dataset == 'mimic':
        base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    elif dataset == 'chexpert':
        base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
    else:
        raise NotImplementedError

    wandb.init(
        project=f"{dataset}_preprocessing",
        name="Evaluation results tabels",
    )

    if external_ood_test:
        base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
        external_train_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train_clean_dataset.csv"
        external_val_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/validation_clean_dataset.csv"
        external_test_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/test_clean_dataset.csv"
        ex_test_df = pd.read_csv(external_test_path)
        ex_train_df = pd.read_csv(external_train_path)
        ex_val_df = pd.read_csv(external_val_path)

        test_df = pd.concat([ex_val_df, ex_train_df, ex_test_df])
        top_races = test_df["race"].value_counts().index[:4]
        test_df = test_df[test_df["race"].isin(top_races)].copy()
    else:
        test_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
        test_df = pd.read_csv(test_output_path)

    weight = f"/deep_learning/output/Sutariya/main/{dataset}/checkpoints/diagnostic/{name}.pth"
    lung_weight = f"/deep_learning/output/Sutariya/main/{dataset}/checkpoints/diagnostic/mask_preprocessing_{name}.pth"                    
    clahe_weight = f"/deep_learning/output/Sutariya/main/{dataset}/checkpoints/diagnostic/clahe_preprocessing_{name}.pth"

    race_weight = f"/deep_learning/output/Sutariya/main/{dataset}/checkpoints/race/{name}.pth"
    race_lung_weight = f"/deep_learning/output/Sutariya/main/{dataset}/checkpoints/race/mask_preprocessing_{name}.pth"                    
    race_clahe_weight = f"/deep_learning/output/Sutariya/main/{dataset}/checkpoints/race/clahe_preprocessing_{name}.pth"

    generate_tabel(race_weight, race_lung_weight, race_clahe_weight, weight, lung_weight, clahe_weight, device, test_df, base_dir, external_ood_test=external_ood_test)