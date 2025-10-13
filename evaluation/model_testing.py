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
    Evaluates a multi-label (or single-label) classification model on a test dataset and logs performance metrics.
    If 'is_groupby' is True and the task is 'diagnostic', further evaluates model performance per group 
    stratified by race and logs group-specific metrics.

    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    model : torch.nn.Module
        Trained classification model.
    dataframe : pd.DataFrame
        DataFrame containing metadata and ground truth labels for the test dataset.
    original_labels : List[str]
        List of label names corresponding to model outputs.
    masked : bool
        Whether lung mask preprocessing was applied.
    clahe : bool
        Whether CLAHE enhancement was applied.
    task : str
        Task identifier (e.g., 'diagnostic').
    crop_masked : bool
        Whether cropping based on masks was applied.
    name : str
        Name identifier for the evaluation run.
    base_dir : str
        Base directory for images.
    device : Optional[torch.device], optional
        Device to run inference on (CPU or GPU). Default is None.
    multi_label : bool, optional
        Indicates multilabel classification if True, single-label otherwise (default True).
    is_groupby : bool, optional
        If True, evaluates metrics by groups stratified by 'race' (default False).
    external_ood_test : bool, optional
        Indicates evaluation on external out-of-distribution test data (default False).

    Returns
    -------
    None
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
