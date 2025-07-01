from helper.generate_plot import get_auroc_by_groups, get_labels
from sklearn.preprocessing import LabelEncoder
from datasets.dataloader import prepare_mimic_dataloaders
from models.build_model import DenseNet_Model
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
import pandas as pd

def generate_race_roc_score(weight, race_loader, device, labels, test_data, avg_method):

     test_model = DenseNet_Model(weights=None, out_feature=5)

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



def generate_tabel(race_weights, race_lung_weights, race_clahe_weights, weights, lung_weights, clahe_weights, device, test_data, base_dir):

    label_encoder = LabelEncoder()
    top_races = test_data["race"].value_counts().index[:5]
    test_data = test_data[test_data["race"].isin(top_races)].copy()
    test_data["race_encoded"] = label_encoder.fit_transform(test_data["race"])
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


    diag_loader = prepare_mimic_dataloaders(
               test_data["Path"],
               test_data[labels].values,
               test_data,
               masked=False,
               clahe=False,
               base_dir=base_dir,
               shuffle=False,
               is_multilabel=True,
           )

    diag_lung_loader = prepare_mimic_dataloaders(
               test_data["Path"],
               test_data[labels].values,
               test_data,
               masked=True,
               clahe=False,
               base_dir=base_dir,
               shuffle=False,
               is_multilabel=True,
           )
    diag_clahe_loader = prepare_mimic_dataloaders(
               test_data["Path"],
               test_data[labels].values,
               test_data,
               masked=False,
               clahe=True,
               base_dir=base_dir,
               shuffle=False,
               is_multilabel=True,
           )

    race_loader = prepare_mimic_dataloaders(
        test_data["Path"],
        test_data["race_encoded"].values,
        test_data,
        masked=False,
        clahe= False,
        base_dir=base_dir,
        shuffle=False,
        is_multilabel=False,
    )
    race_lung_loader = prepare_mimic_dataloaders(
        test_data["Path"],
        test_data["race_encoded"].values,
        test_data,
        masked=True,
        clahe= False,
        base_dir=base_dir,
        shuffle=False,
        is_multilabel=False,
    )
    race_clahe_loader = prepare_mimic_dataloaders(
        test_data["Path"],
        test_data["race_encoded"].values,
        test_data,
        masked=False,
        clahe= True,
        base_dir=base_dir,
        shuffle=False,
        is_multilabel=False,
    )

    race_auc_df = generate_race_roc_score(race_weights, race_loader, device, race_labels, test_data, 'macro')
    race_auc_lung_df = generate_race_roc_score(race_lung_weights, race_lung_loader, device, race_labels, test_data, 'macro')
    race_auc_clahe_df = generate_race_roc_score(race_clahe_weights, race_clahe_loader, device, race_labels, test_data, 'macro')
    
    auc_df = get_auroc_by_groups(diag_loader, weights, device, labels, test_data, 'macro', True)
    auc_lung_df = get_auroc_by_groups(diag_lung_loader, lung_weights, device, labels, test_data, 'macro', True)
    auc_clahe_df = get_auroc_by_groups(diag_clahe_loader, clahe_weights, device, labels, test_data, 'macro', True)

    auc_df['Preprocessing'] = 'Baseline'
    auc_lung_df['Preprocessing'] = 'Lung Masking'
    auc_clahe_df['Preprocessing'] = 'CLAHE'

    # Combine all
    disease_auroc_df = pd.concat([auc_df, auc_lung_df, auc_clahe_df], ignore_index=True)

    disease_auroc_df = disease_auroc_df.pivot_table(index=['Disease', 'Race'], columns='Preprocessing', values='AUROC')
    disease_auroc_df = disease_auroc_df.reset_index()


    race_auc_df['Preprocessing'] = 'Baseline'
    race_auc_lung_df['Preprocessing'] = 'Lung Masking'
    race_auc_clahe_df['Preprocessing'] = 'CLAHE'

    # Combine all
    race_auroc_df = pd.concat([race_auc_df, race_auc_lung_df, race_auc_clahe_df], ignore_index=True)

    summary_df = race_auroc_df.groupby('Race')['AUROC'].agg(
                   Macro_AUROC='mean',
                   Max_Diff_AUROC=lambda x: x.max() - x.min()
               ).reset_index()
               
    diff_latex_str = summary_df.to_latex(index=False, 
                                  caption="Summary of AUROC per Disease and Race groupBy metrics across preprocessing methods", 
                                  label="tab:auroc_diagnostic_summary")
    output_path = "/deep_learning/output/Sutariya/main/mimic/daignostic_diff_auc_result_tabel.tex"
    with open(output_path, "w") as f:
         f.write(diff_latex_str)

    disease_latex_str = disease_auroc_df.to_latex(index=False, 
                                  caption="Summary of AUROC per Disease and Race groupBy metrics across preprocessing methods", 
                                  label="tab:auroc_diagnostic_summary")
    output_path = "/deep_learning/output/Sutariya/main/mimic/daignostic_result_tabel.tex"
    with open(output_path, "w") as f:
         f.write(disease_latex_str)

    race_latex_str = race_auroc_df.to_latex(index=False, 
                                  caption="Summary of AUROC per Disease and Race groupBy metrics across preprocessing methods", 
                                  label="tab:auroc_diagnostic_summary")
    
    race_output_path = "/deep_learning/output/Sutariya/main/mimic/race_encoding_table.tex"
    with open(race_output_path, "w") as f:
         f.write(race_latex_str)
