import numpy as np
import torch
import pandas as pd
import torch.utils.data.dataloader
import torchvision
import wandb

from datasets.dataloader import prepare_dataloaders
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, roc_curve, confusion_matrix

from models.build_model import DenseNet_Model
from helper.log import log_roc_auc
from typing import List, Optional

from data_preprocessing.process_dataset import get_group_by_data
import torch
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.frozen import FrozenEstimator
from sklearn.base import BaseEstimator
from sklearn.isotonic import IsotonicRegression

from tqdm import tqdm

def model_calibration(weight_path, device, val_loader, test_loader, class_names):
    model = DenseNet_Model(weights=None, out_feature=11)
    weights = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    val_probs, val_labels = [], []
    test_probs, test_labels = [], []

    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = torch.sigmoid(model(inputs)).cpu().numpy()
            val_probs.append(logits)
            val_labels.append(labels.cpu().numpy())

        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logits = torch.sigmoid(model(inputs)).cpu().numpy()
            test_probs.append(logits)
            test_labels.append(labels.cpu().numpy())

    val_probs = np.vstack(val_probs)
    val_labels = np.vstack(val_labels)
    test_probs = np.vstack(test_probs)
    test_labels = np.vstack(test_labels)

    calibrators = []
    for i in range(val_labels.shape[1]):
        ir = IsotonicRegression(out_of_bounds='clip')
        ir.fit(val_probs[:, i], val_labels[:, i])
        calibrators.append(ir)

    # Predict calibrated probabilities on test set
    calib_test_prob = {}
    for i in range(test_labels.shape[1]):
        prob = calibrators[i].transform(test_probs[:, i])
        calib_test_prob[i] = prob
        auc = roc_auc_score(test_labels[:, i], prob)
        print(f"Calibrated ROC-AUC for class {i}: {auc:.4f}")

    # Use sklearn's calibration_curve to get expected vs. predicted probabilities
    plt.figure(figsize=(10, 8))

    for i in range(test_labels.shape[1]):  # Loop over 11 classes
        frac_pos, mean_pred = calibration_curve(
            test_labels[:, i], calib_test_prob[i], n_bins=5, strategy='quantile'
        )
        plt.plot(mean_pred, frac_pos, marker='o', label=class_names[i], alpha=0.6)

    # Plot perfect calibration line
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Plot aesthetics
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Isotonic Calibration Curves")
    plt.legend(loc="lower right", fontsize=8, ncol=2)
    plt.grid(True)
    plt.tight_layout()

    # Log to Weights & Biases
    wandb.log({"Isotonic Calibration Curves": wandb.Image(plt)})
    plt.close()


def model_eval_metrics_saving(
    dataframe: pd.DataFrame,
    model: torch.nn.Module,
    original_labels: List[str],
    dataset: str,
    name: str,
    masked:bool=False,
    clahe:bool=False,
    crop_masked:bool=False,
    base_dir: str =None,
    device: Optional[torch.device] =None,
    multi_label: bool =True,
    is_groupby:bool =False,
    external_ood_test:bool = False,
):

    torch.backends.cudnn.benchmark = True
    model.to(device)
    model.eval()
    label_name = [
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
        "Pleural Effusion",
    ]

    test_loader = prepare_dataloaders(
                            images_path= dataframe["Path"],
                            labels= dataframe[original_labels].values,
                            dataframe= dataframe,
                            masked= masked,
                            clahe= clahe,
                            crop_masked= crop_masked,
                            base_dir= base_dir,
                            shuffle=False,
                            is_multilabel=multi_label,
                            external_ood_test =external_ood_test
                        )

    all_test_labels, all_test_preds, idx = [], [], []

    with torch.no_grad():
        for inputs, labels, p_id in test_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            outputs = model(inputs)
            preds = (
                torch.sigmoid(outputs).detach().cpu().numpy()
                if multi_label
                else torch.softmax(outputs, dim=1).detach().cpu().numpy()
            )
            
            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds)
            idx.extend(p_id)

    all_test_preds_np = np.array(all_test_preds)
    all_test_labels_np = np.array(all_test_labels)
    n_samples, n_classes = all_test_preds_np.shape

    best_thresholds = []
    for i in range(n_classes):
        y_true = all_test_labels_np[:, i]
        y_pred_proba = all_test_preds_np[:, i]
    
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_thresholds.append(best_threshold)
    if not clahe and not masked:
        df_threshold = pd.DataFrame(best_thresholds, columns=["best_thresholds"])
        df_threshold.to_csv(f"/deep_learning/output/Sutariya/main/{dataset}/evaluation_files/{name}_thresholds.csv", index=False)

    aurocs = []
    fprs = []
    threshold_df = pd.read_csv(f"/deep_learning/output/Sutariya/main/{dataset}/evaluation_files/{name}_thresholds.csv")
    thresholds = threshold_df['best_thresholds'].values
    for i in range(n_classes):
        y_true = all_test_labels_np[:, i]
        y_pred_proba = all_test_preds_np[:, i]
        y_pred = (y_pred_proba >= thresholds[i]).astype(int)
        auroc = roc_auc_score(
                y_true, y_pred_proba)
        aurocs.append(auroc)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp/(fp+tn)
        fprs.append(fpr)

    df_fpr = pd.DataFrame(fprs, columns=['FPR'])
    df_disease = pd.DataFrame(original_labels, columns=['Disease'])
    df_auc = pd.DataFrame(aurocs, columns=['AUROC'])
    df_metric = pd.concat([df_fpr, df_auc, df_disease], axis=1)

    df_preds = pd.DataFrame(all_test_preds, columns=[f'{label}_pred' for label in label_name])
    df_true = pd.DataFrame(all_test_labels, columns=[f'{label}_true' for label in label_name])
    df_preds['id'] = idx
    df_true['id'] = idx
    dataframe = dataframe.copy()
    dataframe['id'] = dataframe['Path']
    dataframe = dataframe.merge(df_true, on='id', how='inner')
    dataframe = dataframe.merge(df_preds, on='id', how='inner')

    df_metric.to_csv(f'/deep_learning/output/Sutariya/main/{dataset}/evaluation_files/{name}_FPR_metrics.csv', index=False)
    dataframe.to_csv(f'/deep_learning/output/Sutariya/main/{dataset}/evaluation_files/{name}_prob_metrics.csv', index=False)

    if is_groupby:
        race_groupby_dataset = get_group_by_data(dataframe, "race")
        for group in race_groupby_dataset.keys():
            group_data = race_groupby_dataset[group]
            assert not group_data.duplicated("subject_id").any(), f"Duplicate subject_ids in group {group}"
            assert not group_data.duplicated("Path").any(), f"Duplicate image paths in group {group}"

            test_loader = prepare_dataloaders(
                    images_path= group_data["Path"],
                    labels= group_data[original_labels].values,
                    dataframe= group_data,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test= external_ood_test
            )

            group_preds, group_labels = [], []
            with torch.no_grad():
                for inputs, labels, _ in test_loader:
                    inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs).cpu().numpy() if multi_label else torch.softmax(outputs, dim=1).cpu().numpy()
                    group_preds.extend(preds)
                    group_labels.extend(labels.cpu().numpy())

            group_preds_np = np.array(group_preds)
            group_labels_np = np.array(group_labels)
            n_samples, n_classes = group_preds_np.shape
            threshold_df = pd.read_csv(f"/deep_learning/output/Sutariya/main/{dataset}/evaluation_files/{name}_thresholds.csv")
            thresholds = threshold_df['best_thresholds'].values

            group_fprs, group_aurocs = [], []
            for i in range(n_classes):
                y_true = group_labels_np[:, i]
                y_pred_prob = group_preds_np[:, i]
                y_pred = (y_pred_prob >= thresholds[i]).astype(int)
                auroc = roc_auc_score(y_true, y_pred_prob)
                group_aurocs.append(auroc)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

                fpr = fp / (fp + tn + 0.000001) 
                group_fprs.append(fpr)

            # Save group-level metrics
            df_group_metrics = pd.DataFrame({
                "AUROC": group_aurocs,
                "FPR": group_fprs
            }, index=original_labels)

            if group == 'hisp/lat/SA':
                group = 'Hispanic'
            
            df_group_metrics.to_csv(
                f'/deep_learning/output/Sutariya/main/{dataset}/evaluation_files/{name}_{group}_metrics.csv'
            )

            


def model_testing(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    dataframe: pd.DataFrame,
    original_labels: List[str],
    masked: bool,
    clahe: bool,
    task: str,
    crop_masked: bool,
    name:str,
    base_dir:str,
    device: Optional[torch.device] =None,
    multi_label: bool =True,
    is_groupby: bool = False,
    external_ood_test:bool = False,
):
    """
    Evaluates a multi-label classification model on a test dataframe.

    Args:
    - test_loader (DataLoader): DataLoader for test data.
    - model (nn.Module): Trained model.
    - device (torch.device): Device to run inference on (CPU or GPU).

    Returns:
    - auc_roc (float): ROC-AUC score for the test dataframe.
    """

    model.to(device)
    torch.backends.cudnn.benchmark = True
    model.eval()

    all_test_labels, all_test_preds = [], []
    # race = dataframe['race']
    # view_position = dataframe['ViewPosition']
    # image_path = dataframe['Path']
    # subject_id, study_id = dataframe['subject_id'], dataframe['study_id']


    with torch.no_grad():
        for inputs, labels, _ in test_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            if not multi_label:
                labels = torch.argmax(labels, dim=1).long()
            outputs = model(inputs)
            preds = (
                torch.sigmoid(outputs).detach().cpu().numpy()
                if multi_label
                else torch.softmax(outputs, dim=1).detach().cpu().numpy()
            )

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(preds)

    if multi_label:
        auc_roc_test = roc_auc_score(
            all_test_labels, all_test_preds, average="macro"
        )
        test_preds_binary = (np.array(all_test_preds) > 0.4).astype(int)
        test_acc = accuracy_score(all_test_labels, test_preds_binary)
    else:
        auc_roc_test = roc_auc_score(
            all_test_labels, all_test_preds, average="macro", multi_class="ovo"
        )
        test_pred_classes = np.argmax(all_test_preds, axis=1)
        test_acc = accuracy_score(all_test_labels, test_pred_classes)

    
    if external_ood_test:
        wandb.log({"External Testing macro ROC_AUC_Score": auc_roc_test})
        print(f"External Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}")
        log_roc_auc(
        y_true= all_test_labels,
        y_scores= all_test_preds,
        labels= original_labels,
        task= task,
        log_name=f"External Testing macro ROC-AUC for {task}",
        multilabel=multi_label,
        group_name=None,
    )
    else:
        wandb.log({"Testing macro ROC_AUC_Score": auc_roc_test})
        print(f"Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}")
        log_roc_auc(
        y_true= all_test_labels,
        y_scores= all_test_preds,
        labels= original_labels,
        task= task,
        log_name=f"Testing macro ROC-AUC for {task}",
        multilabel=multi_label,
        group_name=None,
    )
    if is_groupby:
        if task == 'diagnostic':
            race_groupby_dataset = get_group_by_data(dataframe, "race")
            for group in race_groupby_dataset.keys():
                group_data = race_groupby_dataset[group]
                assert not group_data.duplicated("subject_id").any(), f"Duplicate subject_ids in group {group}"
                assert not group_data.duplicated("Path").any(), f"Duplicate image paths in group {group}"

                test_loader = prepare_dataloaders(
                    images_path= group_data["Path"],
                    labels= group_data[original_labels].values,
                    dataframe= group_data,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test = external_ood_test,
                )

                group_preds, group_labels = [], []
                with torch.no_grad():
                    for inputs, labels, _ in test_loader:
                        inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                        outputs = model(inputs)
                        preds = torch.sigmoid(outputs).cpu().numpy() if multi_label else torch.softmax(outputs, dim=1).cpu().numpy()
                        group_preds.extend(preds)
                        group_labels.extend(labels.cpu().numpy())

                if multi_label:
                    auc_roc_test = roc_auc_score(
                        group_labels, group_preds, average="weighted"
                    )
                    test_preds_binary = (np.array(group_preds) > 0.4).astype(int)
                    test_acc = accuracy_score(group_labels, test_preds_binary)
                else:
                    auc_roc_test = roc_auc_score(
                        group_labels, group_preds, average="weighted", multi_class="ovo"
                    )
                    test_pred_classes = np.argmax(group_preds, axis=1)
                    test_acc = accuracy_score(group_labels, test_pred_classes)
                if external_ood_test:
                    wandb.log({f"{group} External Testing macro ROC_AUC_Score": auc_roc_test})
                    print(f"{group} External Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}")
                    log_roc_auc(
                    y_true= group_labels,
                    y_scores= group_preds,
                    labels= original_labels,
                    task= task,
                    log_name=f"External Testing {task} macro ROC-AUC for {group}",
                    multilabel=multi_label,
                    group_name=group,
                )
                else:
                    wandb.log({f"{group} Testing macro ROC_AUC_Score": auc_roc_test})
                    print(f"{group} Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}")
                    log_roc_auc(
                    y_true= group_labels,
                    y_scores= group_preds,
                    labels= original_labels,
                    task= task,
                    log_name=f"Testing {task} macro ROC-AUC for {group}",
                    multilabel=multi_label,
                    group_name=group,
                )
        else:
            raise AssertionError("Couldn't find race groupby on race prediction")
