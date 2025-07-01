from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torch
import wandb
from datasets.dataloader import prepare_mimic_dataloaders
from models.build_model import DenseNet_Model

def get_labels(test_loader, weight, device, model, multi_label):
    model.eval()
    print(weight)
    model.to(device)
    weights = torch.load(weight,
                        map_location=device,
                        weights_only=True,
                        )

    model.load_state_dict(weights)
    all_preds, all_labels, all_ids = [], [], []

    with torch.no_grad():
        for inputs, labels, idx in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            preds = (
                torch.sigmoid(outputs).detach().cpu().numpy()
                if multi_label
                else torch.softmax(outputs, dim=1).detach().cpu().numpy()
            )

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
            all_ids.extend(idx)

    return (
        np.array(all_labels),
        np.array(all_preds),
        all_ids
    )


# -------------------------
# Generate Strip Plot
# -------------------------

def get_auroc_by_groups(test_loader, weights, device, labels, test_data, avg_method, multi_label):

    model = DenseNet_Model(
                weights=None,
                out_feature=11
            )

    all_labels, all_preds, all_ids = get_labels(test_loader, weights, device, model, multi_label)
    df_preds = pd.DataFrame(all_preds, columns=[f'{label}_pred' for label in labels])
    df_preds['id'] = all_ids
    test_data = test_data.copy()
    test_data['id'] = test_data['Path']
    test_data = test_data.merge(df_preds, on='id', how='inner')

    auc_records = []
    for race_group, group_df in test_data.groupby('race'):
        for label in labels:

            y_true = group_df[label]
            y_pred = group_df[f'{label}_pred']

            try:
                auc = roc_auc_score(y_true, y_pred, average=avg_method)
            except ValueError:
                auc = float('nan')   
            auc_records.append({
                'Disease': label,
                'AUROC': auc,
                'Race': race_group
            })

    for label in labels:
        y_true = test_data[label]
        y_pred = test_data[f'{label}_pred']

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = float('nan')
        auc_records.append({
                'Disease': label,
                'AUROC': auc,
                'Race': 'all'
            })
    auc_df = pd.DataFrame(auc_records)
    return auc_df


def generate_plot(weights, lung_weights, clahe_weights, device, test_data, multi_label, base_dir):

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
        
    auc_df = get_auroc_by_groups(diag_loader, weights, device, labels, test_data, 'macro', multi_label)
    auc_lung_df = get_auroc_by_groups(diag_lung_loader, lung_weights, device, labels, test_data, 'macro', multi_label)
    auc_clahe_df = get_auroc_by_groups(diag_clahe_loader, clahe_weights, device, labels, test_data, 'macro', multi_label)

    diseases = auc_df['Disease'].unique()
    races = auc_df['Race'].unique()
    x_base = np.arange(len(diseases)) * 2
    offset = np.linspace(-0.6, 0.6, len(races))  # Wider spacing for more races

    def plot_aligned(ax, df, title):
        for i, race in enumerate(races):
            x_vals = []
            y_vals = []
            y_errs = []

            for j, disease in enumerate(diseases):
                group = df[(df['Disease'] == disease) & (df['Race'] == race)]
                x_vals.append(x_base[j] + offset[i])
                y_vals.append(group['AUROC'].values[0])
                y_errs.append(0.0)

            ax.errorbar(x_vals, y_vals, yerr=y_errs, fmt='o', label=race, capsize=2)

        ax.set_title(title)
        ax.set_ylim(0.5, 1.0)
        ax.set_xticks(x_base)
        ax.set_xticklabels(diseases, rotation=45, ha='right')
        ax.grid(True, axis='y')

    # Create plots
    fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True, sharey=True)

    plot_aligned(axs[0], auc_df, "Baseline Model - AUROC")
    plot_aligned(axs[1], auc_lung_df, "Baseline with Lung Masking - AUROC")
    plot_aligned(axs[2], auc_clahe_df, "Baseline with CLAHE - AUROC")

    axs[0].legend(title='Race', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    wandb.log({"Per-Disease AUC by Race (All Methods)": wandb.Image(fig)})
    plt.show()