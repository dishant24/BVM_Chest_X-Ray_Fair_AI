import torch
from helper.early_stop import EarlyStopper
import copy
import numpy as np
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import wandb
from helper.log import log_roc_auc

def model_training(model, train_loader, val_loader, loss_function, tasks, actual_labels, num_epochs: int=10, device=None, multi_label: bool=True, is_swa: bool= True):
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
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, mode='min', factor=0.20, patience=5)
    early_stopper = EarlyStopper(patience=6)

    # SWA will be initialized just before starting SWA training
    if is_swa:
        swa_model = None  
        swa_scheduler = None
        swa_start_epoch = max(num_epochs - 3, 0)  # SWA starts in last 2 epochs

    best_model_weights = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    early_stopped = False

    for epoch in range(num_epochs):
        ### === Training Phase === ###
        model.train()
        train_loss = 0.0
        all_train_labels, all_train_preds = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            base_optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            base_optimizer.step()
            train_loss += loss.item()

            preds = torch.sigmoid(outputs).detach().cpu().numpy() if multi_label else torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)

        train_loss /= len(train_loader)
        if epoch == swa_start_epoch and is_swa == True:
            print("Initializing SWA Model...")
            swa_model = AveragedModel(model)  # Initialize with latest trained weights
            swa_model = swa_model.to(device)
            swa_scheduler = SWALR(base_optimizer, anneal_strategy="cos", anneal_epochs=2, swa_lr=0.001)

        ### === Validation Phase === ###
        model.eval()
        val_loss = 0.0
        all_val_labels, all_val_preds = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()
                preds = torch.sigmoid(outputs).detach().cpu().numpy() if multi_label else torch.softmax(outputs, dim=1).detach().cpu().numpy()
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds)
        val_loss /= len(val_loader)

        # Compute AUC-ROC and Accuracy
        if multi_label:
            auc_roc_train = roc_auc_score(all_train_labels, all_train_preds, average="weighted")
            auc_roc_val = roc_auc_score(all_val_labels, all_val_preds, average="weighted")
            train_preds_binary = (np.array(all_train_preds) > 0.4).astype(int)
            val_preds_binary = (np.array(all_val_preds) > 0.4).astype(int)
            train_acc = f1_score(all_train_labels, train_preds_binary, average='weighted')
            val_acc = f1_score(all_val_labels, val_preds_binary, average='weighted')
        else:
            auc_roc_train = roc_auc_score(all_train_labels, all_train_preds, average='weighted', multi_class='ovr')
            auc_roc_val = roc_auc_score(all_val_labels, all_val_preds, average='weighted', multi_class='ovr')
            train_pred_classes = np.argmax(all_train_preds, axis=1)
            val_pred_classes = np.argmax(all_val_preds, axis=1)
            train_acc = f1_score(all_train_labels, train_pred_classes, average='weighted')
            val_acc = f1_score(all_val_labels, val_pred_classes, average='weighted')

        wandb.log({"Training AUC": auc_roc_train, "Validation AUC": auc_roc_val})
        log_roc_auc(all_train_labels, all_train_preds, actual_labels, tasks, log_name=f"{tasks} Training ROC", multilabel=multi_label, group_name=None)
        log_roc_auc(all_val_labels, all_val_preds, actual_labels, tasks, log_name=f"{tasks} Validation ROC", multilabel=multi_label, group_name=None)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train AUC: {auc_roc_train:.4f}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
              f"Val AUC: {auc_roc_val:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        # Apply SWA if within SWA start phase
        if epoch > swa_start_epoch and is_swa == True:
            swa_model.update_parameters(model)  # Ensure SWA model is actually updated
            swa_scheduler.step()
        else:    
            scheduler.step(val_loss)

        # Save best model weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())

        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered.")
            best_model_weights = copy.deepcopy(model.state_dict())
            early_stopped = True
            break


    # Restore best model weights if early stopped
    model.load_state_dict(best_model_weights)

    # Apply SWA only if training wasn't early stopped
    if not early_stopped and is_swa == True:
        print("Applying SWA...")
        state_dict = swa_model.state_dict()
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k != "n_averaged"}  # Remove prefix & ignore "n_averaged"
        model.load_state_dict(new_state_dict)

    print("Training complete.")
