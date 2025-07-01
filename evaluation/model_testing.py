import numpy as np
import torch
import pandas as pd
import wandb

from datasets.dataloader import prepare_mimic_dataloaders, prepare_chexpert_dataloaders
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
            test_labels[:, i], calib_test_prob[i], n_bins=10, strategy='quantile'
        )
        plt.plot(mean_pred, frac_pos, marker='o', label=class_names[i])

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


def model_testing_metrics_eval(
    dataset:pd.DataFrame,
    model: torch.nn.Module,
    original_labels: List[str],
    task: str,
    name,
    masked:bool=False,
    clahe:bool=False,
    device: Optional[torch.device] =None,
    multi_label: bool =True,
    group_name: str =None,
    threshold_finding: bool=False,
    metrics_saving: bool=False,
    threshold_file_path: str=None,
    is_groupby_testing:bool =False
):

    model.to(device)
    model.eval()
    label = [
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

    top_races = dataset["race"].value_counts().index[:5]
    dataset = dataset[dataset["race"].isin(top_races)].copy()

    if is_groupby_testing:
        if threshold_file_path is None:
            raise AssertionError("Threshold path must be define to calculate and save metrics")
        # Group-wise evaluation using the same thresholds
        race_groupby_dataset = get_group_by_data(dataset, "race")
        for group in race_groupby_dataset.keys():
            group_data = race_groupby_dataset[group]
            assert not group_data.duplicated("subject_id").any(), f"Duplicate subject_ids in group {group}"
            assert not group_data.duplicated("Path").any(), f"Duplicate image paths in group {group}"

            test_loader = prepare_mimic_dataloaders(
                group_data["Path"],
                group_data[label].values,
                group_data,
                masked,
                clahe,
                base_dir="MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/",
                shuffle=False,  
                is_multilabel=multi_label
            )

            group_preds, group_labels = [], []
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    preds = torch.sigmoid(outputs).cpu().numpy() if multi_label else torch.softmax(outputs, dim=1).cpu().numpy()
                    group_preds.extend(preds)
                    group_labels.extend(labels.cpu().numpy())

            group_preds_np = np.array(group_preds)
            group_labels_np = np.array(group_labels)
            n_samples, n_classes = group_preds_np.shape
            threshold_df = pd.read_csv(threshold_file_path)
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

            print(f"Testing AUROC Score for {group} is : {np.mean(group_aurocs)}")
            

            df_group_metrics.to_csv(
                f'/deep_learning/output/Sutariya/main/mimic/evaluation_files/{name}_{group}_metrics.csv'
            )
    else:
        test_loader = prepare_chexpert_dataloaders(
                            dataset["Path"],
                            dataset[label].values,
                            dataset,
                            masked,
                            base_dir="MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/",
                            shuffle=False,
                            is_multilabel=multi_label,
                        )

        all_test_labels, all_test_preds = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

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
                all_test_labels, all_test_preds, average="weighted"
            )
            test_preds_binary = (np.array(all_test_preds) > 0.4).astype(int)
            test_acc = accuracy_score(all_test_labels, test_preds_binary)
        else:
            auc_roc_test = roc_auc_score(
                all_test_labels, all_test_preds, average="weighted", multi_class="ovo"
            )
            test_pred_classes = np.argmax(all_test_preds, axis=1)
            test_acc = accuracy_score(all_test_labels, test_pred_classes)

        all_test_preds_np = np.array(all_test_preds)
        all_test_labels_np = np.array(all_test_labels)
        n_samples, n_classes = all_test_preds_np.shape

        if threshold_finding:
            best_thresholds = []
            for i in range(n_classes):
                y_true = all_test_labels_np[:, i]
                y_pred_proba = all_test_preds_np[:, i]
            
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
                
                f1 = 2 * (precision * recall) / (precision + recall)
                
                best_idx = np.argmax(f1)
                best_threshold = thresholds[best_idx]
                best_thresholds.append(best_threshold)
            df_threshold = pd.DataFrame(best_thresholds, columns=["best_thresholds"])
            df_threshold.to_csv(threshold_file_path, index=False)

        if metrics_saving:
            if threshold_file_path is None:
               raise AssertionError("Threshold path must be define to calculate and save metrics")
            aurocs = []
            fprs = []
            threshold_df = pd.read_csv(threshold_file_path)
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

            df_race = dataset[["race"]].reset_index(drop=True)
            df_fpr = pd.DataFrame.from_dict(fprs, orient="index")
            df_auc = pd.DataFrame.from_dict(aurocs, orient="index")
            df_preds = pd.DataFrame(all_test_preds_np, columns=[f"logit_{label}" for label in original_labels])
            df_labels = pd.DataFrame(all_test_labels_np, columns=[f"label_{label}" for label in original_labels])

            df_combined = pd.concat([df_preds, df_labels, df_race], axis=1)
            df_metric = pd.concat([df_fpr, df_auc])

            df_metric.to_csv(f'/deep_learning/output/Sutariya/main/mimic/evaluation_files/{name}_metrics.csv', index=False)
            df_combined.to_csv(f'/deep_learning/output/Sutariya/main/mimic/evaluation_files/{name}_predictions.csv', index=False)


def model_testing(
    model: torch.nn.Module,
    dataset: pd.DataFrame,
    original_labels: List[str],
    masked: bool,
    clahe: bool,
    task: str,
    name:str,
    base_dir:str,
    device: Optional[torch.device] =None,
    multi_label: bool =True,
    is_groupby: bool = False,
    group_name: str =None,
):
    """
    Evaluates a multi-label classification model on a test dataset.

    Args:
    - test_loader (DataLoader): DataLoader for test data.
    - model (nn.Module): Trained model.
    - device (torch.device): Device to run inference on (CPU or GPU).

    Returns:
    - auc_roc (float): ROC-AUC score for the test dataset.
    """

    model.to(device)
    model.eval()

    all_test_labels, all_test_preds = [], []
    # race = dataset['race']
    # view_position = dataset['ViewPosition']
    # image_path = dataset['Path']
    # subject_id, study_id = dataset['subject_id'], dataset['study_id']

    if not is_groupby:
        test_loader = prepare_mimic_dataloaders(
                        dataset["Path"],
                        dataset[original_labels].values,
                        dataset,
                        masked,
                        clahe,
                        base_dir,
                        shuffle=False,
                        is_multilabel=multi_label)

        with torch.no_grad():
            for inputs, labels, _ in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

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
                all_test_labels, all_test_preds, average="weighted"
            )
            test_preds_binary = (np.array(all_test_preds) > 0.4).astype(int)
            test_acc = accuracy_score(all_test_labels, test_preds_binary)
        else:
            auc_roc_test = roc_auc_score(
                all_test_labels, all_test_preds, average="weighted", multi_class="ovo"
            )
            test_pred_classes = np.argmax(all_test_preds, axis=1)
            test_acc = accuracy_score(all_test_labels, test_pred_classes)

        # df_preds = pd.DataFrame(all_test_preds_np, columns=[f"logit_{label}" for label in original_labels])
        # df_labels = pd.DataFrame(all_test_labels_np, columns=[f"label_{label}" for label in original_labels])

        # df_group_metrics = pd.DataFrame({
        #             "race": race,
        #             "view_position":view_position,
        #             "subject_id":subject_id,
        #             "study_id":study_id,
        #             "image_path":image_path
        #         })
        # df_all = pd.concat([df_preds, df_labels, df_group_metrics], axis=1)

        # df_all.to_csv(
        #             f'/deep_learning/output/Sutariya/main/mimic/evaluation_files/{name}_metrics.csv'
        #         )
        log_roc_auc(
            all_test_labels,
            all_test_preds,
            original_labels,
            task,
            log_name=f"Testing ROC-AUC for {task} {group_name}",
            multilabel=multi_label,
            group_name=group_name,
        )
        wandb.log(
            {"Testing ROC_AUC_Score": auc_roc_test}
        ) if group_name is None else wandb.log(
            {f"{group_name} Testing ROC_AUC_Score": auc_roc_test}
        )
        # log_confusion_matrix(all_test_labels, all_test_preds, log_name="Testing Confusion Matrix")
        print(
            f"Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}"
        )

    else:
        if task == 'diagnostic':
            race_groupby_dataset = get_group_by_data(dataset, "race")
            for group in race_groupby_dataset.keys():
                group_data = race_groupby_dataset[group]
                assert not group_data.duplicated("subject_id").any(), f"Duplicate subject_ids in group {group}"
                assert not group_data.duplicated("Path").any(), f"Duplicate image paths in group {group}"

                test_loader = prepare_mimic_dataloaders(
                    group_data["Path"],
                    group_data[original_labels].values,
                    group_data,
                    masked,
                    clahe,
                    base_dir="MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/",
                    shuffle=False,  
                    is_multilabel=multi_label
                )

                group_preds, group_labels = [], []
                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
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
                log_roc_auc(
                    group_labels,
                    group_preds,
                    original_labels,
                    task,
                    log_name=f"Testing ROC-AUC for {task} {group_name}",
                    multilabel=multi_label,
                    group_name=group_name,
                )
                wandb.log(
                    {"Testing ROC_AUC_Score": auc_roc_test}
                ) if group_name is None else wandb.log(
                    {f"{group_name} Testing ROC_AUC_Score": auc_roc_test}
                )
                print(
                    f"Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}"
                )
        else:
            raise AssertionError("Couldn't find race groupby on race prediction")
