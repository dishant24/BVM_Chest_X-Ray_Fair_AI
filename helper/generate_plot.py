from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
import torch
import wandb

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

def get_auroc_by_groups(test_loader, weights, device, model, labels, test_data, avg_method, multi_label):

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
                'ROC AUC': auc,
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
                'ROC AUC': auc,
                'Race': 'all'
            })
    auc_df = pd.DataFrame(auc_records)
    return auc_df


def generate_strip_plot(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, model, labels, test_data, multi_label):

        
    auc_df = get_auroc_by_groups(test_loader, weights, device, model, labels, test_data, 'macro', multi_label)
    auc_lung_df = get_auroc_by_groups(lung_test_loader, lung_weights, device, model, labels, test_data, 'macro', multi_label)
    auc_clahe_df = get_auroc_by_groups(clahe_test_loader, clahe_weights, device, model, labels, test_data, 'macro', multi_label)

    fig, axs = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

    # First plot: Normal
    sns.stripplot(data=auc_df, x='Disease', y='ROC AUC', hue='Race', size=8, jitter=True, ax=axs[0])
    axs[0].set_title("Baseline Model - ROC AUC")
    axs[0].tick_params(axis='x', rotation=45)

    # Second plot: Lung
    sns.stripplot(data=auc_lung_df, x='Disease', y='ROC AUC', hue='Race', size=8, jitter=True, ax=axs[1])
    axs[1].set_title("Baseline with lung masking - ROC AUC")
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].legend_.remove()

    # Third plot: CLAHE
    sns.stripplot(data=auc_clahe_df, x='Disease', y='ROC AUC', hue='Race', size=8, jitter=True, ax=axs[2])
    axs[2].set_title("Baseline with Clahe processing - ROC AUC")
    axs[2].tick_params(axis='x', rotation=45)
    axs[2].legend_.remove()

    # Add one shared legend (from last plot)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Race", loc='upper right', bbox_to_anchor=(1.12, 0.98))

    plt.tight_layout()

    # Log the full figure to W&B
    wandb.log({"Per-Disease AUC by Race (All Methods)": wandb.Image(fig)})

    plt.show()