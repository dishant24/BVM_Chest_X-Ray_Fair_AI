import sys
import os
from typing import Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
from sklearn.utils import resample
import wandb
from datasets.dataloader import prepare_dataloaders
from models.build_model import DenseNet_Model
import argparse

def get_labels(test_loader: torch.utils.data.DataLoader, weight: str, device:torch.device , model: torch.nn.Module, multi_label: bool = True) -> Tuple:
    """
    Obtains predictions and true labels from a given PyTorch model on a test dataset.

    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        DataLoader providing batches of test images and labels.
    weight : str
        Path to the saved model weights file.
    device : torch.device
        Device on which to perform inference (CPU or GPU).
    model : torch.nn.Module
        The trained model to evaluate.
    multi_label : bool
        Whether the task is multi-label classification (True) or single-label classification (False).

    Returns
    -------
    tuple
        A tuple containing:
        - all_labels: numpy array of true labels.
        - all_preds: numpy array of predicted probabilities or scores.
        - all_ids: list of indices or identifiers corresponding to predictions.
    """
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


def get_auroc_by_groups(test_loader:torch.utils.data.DataLoader, weights:str, device:torch.device, labels: list[str], test_df: pd.DataFrame, avg_method: str, multi_label:bool = True)-> pd.DataFrame:

    """
    Computes the AUROC (Area Under ROC Curve) scores for each disease label within different race groups,
    as well as overall AUROC scores across all races. Uses model predictions obtained from a test DataLoader.

    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    weights : str
        Path to the model weights file.
    device : torch.device
        Device on which to perform inference (CPU/GPU).
    labels : list of str
        List of disease label names.
    test_df : pd.DataFrame
        Test dataset DataFrame containing metadata including 'race' and ground truth labels.
    avg_method : str
        Averaging method parameter for sklearn's roc_auc_score (e.g., 'macro', 'weighted').
    multi_label : bool
        Whether the task is multi-label classification.

    Returns
    -------
    pd.DataFrame
        DataFrame with AUROC scores per disease per race group and overall.
    """

    model = DenseNet_Model(
                weights=None,
                out_feature=11
            )

    all_labels, all_preds, all_ids = get_labels(test_loader, weights, device, model, multi_label)
    df_preds = pd.DataFrame(all_preds, columns=[f'{label}_pred' for label in labels])
    df_true = pd.DataFrame(all_labels, columns=[f'{label}_true' for label in labels])
    df_preds['id'] = all_ids
    df_true['id'] = all_ids
    test_df = test_df.copy()
    test_df['id'] = test_df['Path']
    test_df = test_df.merge(df_true, on='id', how='inner')
    test_df = test_df.merge(df_preds, on='id', how='inner')

    auc_records = []
    for race_group, group_df in test_df.groupby('race'):
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
        y_true = test_df[label]
        y_pred = test_df[f'{label}_pred']

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
    return auc_df


def get_bootstrap_auc(test_loader:torch.utils.data.DataLoader, weights:str, device: torch.device, labels : list[str], test_df:pd.DataFrame, multi_label:bool =True, n_bootstraps: int=100, ci: int=0.95, seed:int =42)-> pd.DataFrame:

    """
    Calculates bootstrap confidence intervals for AUROC scores per disease label grouped by race and overall.
    Uses model predictions from a test DataLoader with bootstrapping for statistical estimation.

    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        DataLoader providing test data.
    weights : str
        Path to model weights to load.
    device : torch.device
        Device for model inference.
    labels : list of str
        Disease label names.
    test_df : pd.DataFrame
        Test dataset metadata including ground truth labels and race information.
    multi_label : bool
        Whether task is multi-label classification.
    n_bootstraps : int, optional
        Number of bootstrap samples to generate (default 100).
    ci : float, optional
        Confidence interval level (default 0.95).
    seed : int, optional
        Random seed for reproducibility (default 42).

    Returns
    -------
    pd.DataFrame
        DataFrame with AUROC means and confidence intervals for each disease and race group, plus overall.
    """

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
    test_df = test_df.copy()
    test_df['id'] = test_df['Path']
    test_df = test_df.merge(df_true, on='id', how='inner')
    test_df = test_df.merge(df_preds, on='id', how='inner')

    auc_records = []
    for race_group, group_df in test_df.groupby('race'):
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
        y_true = test_df[label]
        y_pred = test_df[f'{label}_pred']
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

def generate_plot(weights:str, lung_weights:str, crop_lung_weights:str, clahe_weights:str, device:torch.device, testing_df:pd.DataFrame, multi_label:bool = True, base_dir:str = None, external_ood_test:bool = False)-> None:

    """
    Generates comparative AUROC plots by race for different preprocessing methods/models on test data.
    Prepares DataLoaders with different masking or enhancement options, computes bootstrap AUROCs,
    and plots them with error bars showing confidence intervals.

    Parameters
    ----------
    weights : str
        Path to baseline model weights.
    lung_weights : str
        Path to model weights with lung masking.
    crop_lung_weights : str
        Path to model weights with cropped lung masking.
    clahe_weights : str
        Path to model weights with CLAHE enhancement.
    device : torch.device
        Device for running inference (CPU/GPU).
    testing_df : pd.DataFrame
        Test dataset DataFrame with image paths, labels, and metadata including race.
    multi_label : bool
        Whether the task is multi-label classification.
    base_dir : str
        Base directory for image file paths.
    external_ood_test : bool
        Flag indicating whether external out-of-distribution test.

    Returns
    -------
    None
    """

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
                    images_path= testing_df["Path"],
                    labels= testing_df[labels].values,
                    dataframe= testing_df,
                    masked= False,
                    clahe= False,
                    crop_masked= False,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test=external_ood_test
           )

    diag_lung_loader = prepare_dataloaders(
                    images_path= testing_df["Path"],
                    labels= testing_df[labels].values,
                    dataframe= testing_df,
                    masked= True,
                    clahe= False,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test=external_ood_test
           )

    diag_crop_lung_loader = prepare_dataloaders(
                    images_path= testing_df["Path"],
                    labels= testing_df[labels].values,
                    dataframe= testing_df,
                    masked= False,
                    clahe= False,
                    crop_masked= True,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test=external_ood_test
           )

    diag_clahe_loader = prepare_dataloaders(
                    images_path= testing_df["Path"],
                    labels= testing_df[labels].values,
                    dataframe= testing_df,
                    masked= False,
                    clahe= True,
                    crop_masked= False,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test=external_ood_test
           )
        
    auc_df = get_bootstrap_auc(diag_loader, weights, device, labels, test_df, multi_label)
    auc_lung_df = get_bootstrap_auc(diag_lung_loader, lung_weights, device, labels, test_df, multi_label)
    auc_crop_lung_df = get_bootstrap_auc(diag_crop_lung_loader, crop_lung_weights, device, labels, test_df, multi_label)
    auc_clahe_df = get_bootstrap_auc(diag_clahe_loader, clahe_weights, device, labels, test_df, multi_label)

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
        ax.set_ylim(0.4, 1.0)
        ax.set_xticks(x_base)
        ax.set_xticklabels(diseases, rotation=45, ha='right')
        ax.grid(True, axis='y')


    # Create plots
    fig, axs = plt.subplots(4, 1, figsize=(18, 16), sharex=True, sharey=True)

    plot_aligned(axs[0], auc_df, "Baseline Model - AUROC")
    plot_aligned(axs[1], auc_lung_df, "Baseline with Lung Masking - AUROC")
    plot_aligned(axs[2], auc_crop_lung_df, "Baseline with Crop Lung Masking - AUROC")
    plot_aligned(axs[3], auc_clahe_df, "Baseline with CLAHE - AUROC")

    axs[0].legend(title='Race', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    wandb.log({"Per-Disease AUC by Race (All Methods)": wandb.Image(fig)})
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing Arguments")
    parser.add_argument('--random_state', type=int, default=100)
    parser.add_argument('--task', type=str, choices=["diagnostic", "race"], default="diagnostic")
    parser.add_argument('--dataset', type=str, default="mimic")
    parser.add_argument('--external_ood_test', action='store_true')

    args = parser.parse_args()
    print(vars(args))
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = args.random_state
    task = args.task
    dataset = args.dataset
    external_ood_test = args.external_ood_test

    name = f"model_baseline_{dataset}_{random_state}"
    
    if task == 'diagnostic':
        multi_label = True
    elif task == 'race':
        multi_label = False
    else:
        raise NotImplementedError

    wandb.init(
        project=f"{dataset}_preprocessing_{task}",
        name="Evaluation results",
    )

    if external_ood_test:
        external_total_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/chexpert_total_dataset.csv"
        base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
        test_df = pd.read_csv(external_total_path)
    else:
        base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
        test_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
        test_df = pd.read_csv(test_output_path)

    baseline_model_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/{name}.pth"

    lung_model_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/mask_preprocessing_{name}.pth"

    crop_lung_model_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/crop_mask_preprocessing_{name}.pth"                    

    clahe_model_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/clahe_preprocessing_{name}.pth"

    generate_plot(
                weights=baseline_model_path,
                lung_weights=lung_model_path,
                crop_lung_weights=crop_lung_model_path,
                clahe_weights=clahe_model_path,
                device=device,
                testing_df=test_df,
                multi_label=multi_label,
                base_dir=base_dir,
                external_ood_test=external_ood_test
                )
