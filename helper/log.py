import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import wandb
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from typing import List, Union

def log_roc_auc(
    y_true : List[Union[int, float]],
    y_scores: List[Union[int, float]],
    labels : List[str]=None,
    task: str =None,
    multilabel: bool =True,
    log_name: str ="roc_auc_curve",
    group_name: str =None,
)-> None:
    """
    Plots the ROC curve for multi-label or multi-class classification.

    Args:
    - y_true (np.array): True labels.
    - y_scores (np.array): Predicted probabilities.
    - labels (list): List of label names (optional if task is provided).
    - task (str): 'diagnostic', 'race', etc.
    - multilabel (bool): True if multi-label classification.
    - log_name (str): Log title.
    - group_name (str): Optional subgroup label for title.
    """

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # Fallback to default labels if none passed
    if labels is None:
        labels = np.arange(0, y_scores.shape[1])

    fig, ax = plt.subplots(figsize=(7, 7))

    if not multilabel:
        num_classes = y_scores.shape[1]
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.2f})")
    else:
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(
        f"ROC Curve {task}"
        if group_name is None
        else f"ROC Curve {task} of {group_name}"
    )
    ax.legend(loc="lower right", fontsize=8 if num_classes > 10 else 10)

    wandb.log({log_name: wandb.Image(fig)})
    plt.close(fig)


def log_confusion_matrix(y_true: List[Union[int, float]], y_pred: List[Union[int, float]], multilabel: bool =True, log_name: str ="confusion_matrix"):
    """
    Plots the confusion matrix for multi-label or multi-class classification.

    Args:
    - y_true (np.array): True labels.
        * If multilabel=False: multi-hot encoded (multi-label).
        * If multilabel=True: single-label integer encoded.
    - y_pred (np.array): Predicted labels.
    - multilabel (bool): Set True for multi-class, False for multi-label.
    - log_name (str): Name for logging.

    Returns:
    - None
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if not multilabel:
        # Multi-class confusion matrix
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Confusion Matrix (Multi-Class)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    else:
        # Multi-label confusion matrix (per-class)
        num_classes = y_true.shape[1]
        fig, axes = plt.subplots(1, num_classes, figsize=(20, 12))

        for i in range(num_classes):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", ax=axes[i])
            axes[i].set_title(f"Class {i}")
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

    plt.tight_layout()
    wandb.log({log_name: wandb.Image(fig)})
    plt.close(fig)
