import numpy as np
import torch
import pandas as pd
import wandb

from datasets.dataloader import prepare_mimic_dataloaders, prepare_chexpert_dataloaders
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_auc_score, roc_curve, confusion_matrix

from models.build_model import DenseNet_Model
from helper.log import log_roc_auc
from typing import List, Optional
from sklearn.frozen import FrozenEstimator
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

from data_preprocessing.process_dataset import get_group_by_data
import torch
import numpy as np
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from sklearn.frozen import FrozenEstimator
from sklearn.base import BaseEstimator
from tqdm import tqdm
from sklearn.utils.validation import check_is_fitted


class DenseNetSklearnWrapper(BaseEstimator):
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
    
def log_calibration_curve(y_true, y_prob, label_name, num_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=num_bins)
    plt.figure(figsize=(4, 4))
    plt.plot(prob_pred, prob_true, marker='o', label=f'{label_name}')
    plt.plot([0, 1], [0, 1], '--k', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration Curve: {label_name}')
    plt.legend()
    wandb.log({f"calibration/{label_name}": wandb.Image(plt)})
    plt.close()
    
def calibrate_model(model, val_data, val_labels, method="isotonic"):
    calibrators = []
    wrapped_model = DenseNetSklearnWrapper(model)
    wrapped_model.fit(val_data) 

    for i in range(val_labels.shape[1]):
        base_clf = SingleLabelWrapper(wrapped_model, i)
        base_clf.fit(val_data, val_labels[:, i])
        calib = CalibratedClassifierCV(base_clf, method=method, cv="prefit")
        calib.fit(val_data, val_labels[:, i])
        calibrators.append(calib)

    return calibrators

def predict_calibrated(calibrators, test_data):
    calibrated_probs = np.column_stack([
        calibrators[i].predict_proba(test_data)[:, 1] for i in range(len(calibrators))
    ])
    return calibrated_probs

class SingleLabelWrapper(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(self, multi_output_model, label_idx):
        self.multi_output_model = multi_output_model
        self.label_idx = label_idx

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.is_fitted_ = True
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "is_fitted_")
        probs = self.multi_output_model.predict_proba(X)
        label_probs = probs[:, self.label_idx]
        return np.column_stack([1 - label_probs, label_probs])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
    
def get_logits_and_labels(model, dataloader, device):
    all_logits, all_labels = [], []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            logits = model(x_batch).cpu()
            all_logits.append(logits)
            all_labels.append(y_batch.cpu())

    return torch.cat(all_logits).numpy(), torch.cat(all_labels).numpy()

def calibrate_and_predict_dense_net(
    model_class,
    model_path,
    val_loader,
    test_loader,
    device,
    num_labels=11,
    method="isotonic"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(weights=None, out_feature=num_labels)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)

    # Extract logits (before sigmoid) from validation and test sets
    val_logits, val_labels = get_logits_and_labels(model, val_loader, device)
    test_logits, test_labels = get_logits_and_labels(model, test_loader, device)

    # Wrap model with sklearn interface
    wrapped_model = DenseNetSklearnWrapper(model, device=device)
    wrapped_model.fit(val_logits)  # dummy fit to satisfy sklearn API

    # Calibrate using sklearn (per label)
    calibrators = calibrate_model(wrapped_model, val_logits, val_labels, method=method)

    # Predict calibrated probabilities on test set
    calibrated_probs = predict_calibrated(calibrators, test_logits)
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(test_labels, calibrated_probs, average="macro")
    print(f"Calibrated macro ROC-AUC: {auc:.4f}")

    log_calibration_curve(test_labels, calibrated_probs, "lung_calibration_prob")


def model_testing_metrics_eval(
    dataset:pd.DataFrame,
    model: torch.nn.Module,
    original_labels: List[str],
    task: str,
    name,
    masked:bool=False,
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
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    dataset: pd.DataFrame,
    original_labels: List[str],
    task: str,
    name,
    device: Optional[torch.device] =None,
    multi_label: bool =True,
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
    no_finding_labels = dataset['No Finding'].values
    race = dataset['race']
    view_position = dataset['ViewPosition']
    image_path = dataset['Path']
    subject_id, study_id = dataset['subject_id'], dataset['study_id']

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

    df_preds = pd.DataFrame(all_test_preds_np, columns=[f"logit_{label}" for label in original_labels])
    df_labels = pd.DataFrame(all_test_labels_np, columns=[f"label_{label}" for label in original_labels])

    df_group_metrics = pd.DataFrame({
                "race": race,
                "view_position":view_position,
                "subject_id":subject_id,
                "study_id":study_id,
                "image_path":image_path
            })
    df_all = pd.concat([df_preds, df_labels, df_group_metrics], axis=1)

    df_all.to_csv(
                f'/deep_learning/output/Sutariya/main/mimic/evaluation_files/{name}_metrics.csv'
            )
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

def calibration_testing(test_dataset, val_dataset, device, model_path, masked, base_dir, multi_label=True):


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
        "Pleural Effusion",
    ]
    val_loader = prepare_mimic_dataloaders(
                        val_dataset["Path"],
                        val_dataset[labels].values,
                        val_dataset,
                        masked,
                        base_dir,
                        shuffle=False,
                        is_multilabel=multi_label)

    test_model = DenseNet_Model(weights=None, out_feature=11)
    test_model.eval()
    test_model.to(device)
    weights = torch.load(
                        model_path,
                        map_location=device, weights_only=True
                    )
    test_model.load_state_dict(weights)

    def collect_logits_labels(model, val_loader, device, multi_label=multi_label):
        

        all_logits, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)

                logits = (
                    torch.sigmoid(outputs).cpu().numpy() if multi_label
                    else torch.softmax(outputs, dim=1).cpu().numpy()
                )
                all_logits.extend(logits)
                all_labels.extend(labels.cpu().numpy())

        return np.array(all_logits), np.array(all_labels)

    val_logits, val_labels = collect_logits_labels(test_model, val_loader, device, multi_label=multi_label)

    # Step 2: Create frozen estimator and calibrate it
    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(test_model), method='isotonic')
    calibrated_clf.fit(val_logits, val_labels)

    
    top_races = test_dataset["race"].value_counts().index[:5]
    dataset = test_dataset[test_dataset["race"].isin(top_races)].copy()
    race_groupby_dataset = get_group_by_data(dataset, "race")
    for group in race_groupby_dataset.keys():
        group_data = race_groupby_dataset[group]
        assert not group_data.duplicated("subject_id").any(), f"Duplicate subject_ids in group {group}"
        assert not group_data.duplicated("Path").any(), f"Duplicate image paths in group {group}"

        test_loader = prepare_mimic_dataloaders(
            group_data["Path"],
            group_data[labels].values,
            group_data,
            masked,
            base_dir="MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/",
            shuffle=False,  
            is_multilabel=multi_label
        )

        test_logits, test_labels = collect_logits_labels(test_model, test_loader, device, multi_label=multi_label)

        # Get calibrated probabilities
        calibrated_probs = calibrated_clf.predict_proba(test_logits)

        # Use these calibrated_probs for evaluation
        auc_roc = roc_auc_score(test_labels, calibrated_probs)

        print(f"Calibrated ROC-AUCfor group {group} : {auc_roc}")
