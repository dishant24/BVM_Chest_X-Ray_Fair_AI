from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torch
from sklearn.utils import resample
import wandb
from datasets.dataloader import prepare_dataloaders
from models.build_model import DenseNet_Model

def get_labels(test_loader, weight, device, model, multi_label):
    model.eval()
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
    df_true = pd.DataFrame(all_labels, columns=[f'{label}_true' for label in labels])
    df_preds['id'] = all_ids
    df_true['id'] = all_ids
    test_data = test_data.copy()
    test_data['id'] = test_data['Path']
    test_data = test_data.merge(df_true, on='id', how='inner')
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
                'Race': race_group,
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
                'Race': 'all',
            })
    auc_df = pd.DataFrame(auc_records)
    return auc_df, test_data


def get_bootstrap_auc(test_loader, weights, device, labels, test_data, multi_label, n_bootstraps=100, ci=0.95, seed=42):
    np.random.seed(seed)
    model = DenseNet_Model(
                weights=None,
                out_feature=11
            )

    all_labels, all_preds, all_ids = get_labels(test_loader, weights, device, model, multi_label)
    df_preds = pd.DataFrame(all_preds, columns=[f'{label}_pred' for label in labels])
    df_true = pd.DataFrame(all_labels, columns=[f'{label}_true' for label in labels])
    df_preds['id'] = all_ids
    df_true['id'] = all_ids
    test_data = test_data.copy()
    test_data['id'] = test_data['Path']
    test_data = test_data.merge(df_true, on='id', how='inner')
    test_data = test_data.merge(df_preds, on='id', how='inner')

    auc_records = []
    for race_group, group_df in test_data.groupby('race'):
        for label in labels:

            y_true = group_df[label]
            y_pred = group_df[f'{label}_pred']
            bootstrapped_scores = []

            for _ in range(n_bootstraps):
                indices = resample(np.arange(len(y_true)))
                score = roc_auc_score(y_true.iloc[indices], y_pred.iloc[indices])
                bootstrapped_scores.append(score)

            sorted_scores = np.sort(bootstrapped_scores)
            lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
            upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
            error = np.array([lower, upper])
            auc = sorted_scores.mean()
            auc_records.append({
                'Disease': label,
                'AUROC': auc,
                'Race': race_group,
                'error': error
            })

    for label in labels:
        y_true = test_data[label]
        y_pred = test_data[f'{label}_pred']
        bootstrapped_scores = []

        for _ in range(n_bootstraps):

            indices = resample(np.arange(len(y_true)))
            score = roc_auc_score(y_true.iloc[indices], y_pred.iloc[indices])
            bootstrapped_scores.append(score)

        sorted_scores = np.sort(bootstrapped_scores)
        lower = np.percentile(sorted_scores, (1 - ci) / 2 * 100)
        upper = np.percentile(sorted_scores, (1 + ci) / 2 * 100)
        error = np.array([lower, upper])
        auc = sorted_scores.mean()
        auc_records.append({
            'Disease': label,
            'AUROC': auc,
            'Race': 'all',
            'error': error
        })

    auc_df = pd.DataFrame(auc_records)
    print(auc_df)
    
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

    diag_loader = prepare_dataloaders(
               test_data["Path"],
               test_data[labels].values,
               test_data,
               masked=False,
               clahe=False,
               base_dir=base_dir,
               shuffle=False,
               is_multilabel=True,
           )

    diag_lung_loader = prepare_dataloaders(
               test_data["Path"],
               test_data[labels].values,
               test_data,
               masked=True,
               clahe=False,
               base_dir=base_dir,
               shuffle=False,
               is_multilabel=True,
           )
    diag_clahe_loader = prepare_dataloaders(
               test_data["Path"],
               test_data[labels].values,
               test_data,
               masked=False,
               clahe=True,
               base_dir=base_dir,
               shuffle=False,
               is_multilabel=True,
           )
        
    auc_df = get_bootstrap_auc(diag_loader, weights, device, labels, test_data, multi_label)
    auc_lung_df = get_bootstrap_auc(diag_lung_loader, lung_weights, device, labels, test_data, multi_label)
    auc_clahe_df = get_bootstrap_auc(diag_clahe_loader, clahe_weights, device, labels, test_data, multi_label)

    diseases = auc_df['Disease'].unique()
    races = auc_df['Race'].unique()
    x_base = np.arange(len(diseases)) * 2
    offset = np.linspace(-0.6, 0.6, len(races))
    def plot_aligned(ax, df, title):
        for i, race in enumerate(races):
            x_vals = []
            y_vals = []
            errors = []

            for j, disease in enumerate(diseases):
                group = df[(df['Disease'] == disease) & (df['Race'] == race)]
                x_vals.append(x_base[j] + offset[i])
                y = group['AUROC'].values[0]
                y_vals.append(y)

                lower, upper = group['error'].values[0]
                errors.append([y - lower, upper - y])

            errors_np = np.array(errors).T
            ax.errorbar(x_vals, y_vals, yerr=errors_np, fmt='o', label=race, capsize=2)

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