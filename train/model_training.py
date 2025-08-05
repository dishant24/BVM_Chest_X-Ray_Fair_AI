import copy

import numpy as np
import torch
import wandb
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from torch.optim.swa_utils import SWALR, AveragedModel

from helper.log import log_roc_auc
from typing import Callable, List, Optional
from helper.early_stop import EarlyStopperByAUC, EarlyStopperByLoss

def model_training(
    model : torch.nn.Module,
    train_loader : torch.utils.data.DataLoader,
    val_loader : torch.utils.data.DataLoader,
    loss_function : Callable ,
    tasks : str,
    actual_labels : List[str],
    num_epochs: int = 10,
    device : Optional[torch.device] =None ,
    multi_label: bool = True,
) -> None:
    """
    Trains a model for either multi-label or multi-class classification.

    Args:
    - model (nn.Module): The neural network model.
    - train_loader (DataLoader): Training data loader.
    - val_loader (DataLoader): Validation data loader.
    - num_epochs (int): Number of training epochs.
    - device (torch.device): Device to train on (CPU or GPU).
    - multi_label (bool): Whether the task is multi-label (default: True).

    Returns:
    - None
    """
    model = model.to(device)
    base_optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0001, weight_decay=0.001
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        base_optimizer,T_max=num_epochs, eta_min=10e-6)
    early_stopper = EarlyStopperByAUC(patience=5)

    best_val_auc = 0
    best_model_weights = copy.deepcopy(model.state_dict())

    torch.backends.cudnn.benchmark = True

    for epoch in range(num_epochs):
        ### === Training Phase === ###
        model.train()
        train_loss = 0.0
        all_train_labels, all_train_preds = [], []

        for inputs, labels, _ in train_loader:
            inputs, labels = inputs.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            base_optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            base_optimizer.step()
            train_loss += loss.detach()

            preds = (
                torch.sigmoid(outputs).detach().cpu().numpy()
                if multi_label
                else torch.softmax(outputs, dim=1).detach().cpu().numpy()
            )
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)

        train_loss /= len(train_loader)
        scheduler.step()

        ### === Validation Phase === ###
        model.eval()
        val_loss = 0.0
        all_val_labels, all_val_preds = [], []
        with torch.no_grad():
            for inputs, labels, _ in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.detach()
                preds = (
                    torch.sigmoid(outputs).detach().cpu().numpy()
                    if multi_label
                    else torch.softmax(outputs, dim=1).detach().cpu().numpy()
                )
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds)
        val_loss /= len(val_loader)

        # Compute AUC-ROC and Accuracy
        if multi_label:
            auc_roc_train = roc_auc_score(
                all_train_labels, all_train_preds, average="weighted"
            )
            auc_roc_val = roc_auc_score(
                all_val_labels, all_val_preds, average="weighted"
            )
            train_preds_binary = (np.array(all_train_preds) > 0.4).astype(int)
            val_preds_binary = (np.array(all_val_preds) > 0.4).astype(int)
            train_acc = accuracy_score(
                all_train_labels, train_preds_binary
            )
            val_acc = accuracy_score(all_val_labels, val_preds_binary)
        else:
            auc_roc_train = roc_auc_score(
                all_train_labels, all_train_preds, average="weighted", multi_class="ovr"
            )
            auc_roc_val = roc_auc_score(
                all_val_labels, all_val_preds, average="weighted", multi_class="ovr"
            )
            train_pred_classes = np.argmax(all_train_preds, axis=1)
            val_pred_classes = np.argmax(all_val_preds, axis=1)
            train_acc = accuracy_score(
                all_train_labels, train_pred_classes
            )
            val_acc = accuracy_score(all_val_labels, val_pred_classes)

        wandb.log({"Training AUC": auc_roc_train, "Validation AUC": auc_roc_val})
        log_roc_auc(
            y_true= all_train_labels,
            y_scores= all_train_preds,
            labels= actual_labels,
            task= tasks,
            log_name=f"Training macro ROC-AUC for {tasks}",
            multilabel=multi_label,
            group_name=None,
        )
        log_roc_auc(
            y_true= all_val_labels,
            y_scores= all_val_preds,
            labels= actual_labels,
            task= tasks,
            log_name=f"Validation macro ROC-AUC for {tasks}",
            multilabel=multi_label,
            group_name=None,
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], "
            f"Train AUC: {auc_roc_train:.4f}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
            f"Val AUC: {auc_roc_val:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}"
        )

        if auc_roc_val > best_val_auc:
            best_val_auc = auc_roc_val
            best_model_weights = copy.deepcopy(model.state_dict())

        if early_stopper.early_stop(auc_roc_val):
            print("Early stopping triggered.")
            best_model_weights = copy.deepcopy(model.state_dict())
            break

    # Restore best model weights if early stopped
    model.load_state_dict(best_model_weights)
    
    print("Training complete.")

