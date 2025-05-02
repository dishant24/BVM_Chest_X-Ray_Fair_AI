import torch
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from helper.log import log_roc_auc
import numpy as np
import wandb

def model_testing(test_loader, model, original_labels, task,device=None, multi_label=True, group_name=None):
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
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  
            preds = torch.sigmoid(outputs).detach().cpu().numpy() if multi_label else torch.softmax(outputs, dim=1).detach().cpu().numpy()

            all_test_labels.extend(labels.cpu().numpy())  
            all_test_preds.extend(preds) 

    if multi_label:
        auc_roc_test = roc_auc_score(all_test_labels, all_test_preds, average="weighted")
        test_preds_binary = (np.array(all_test_preds) > 0.4).astype(int)
        test_acc = accuracy_score(all_test_labels, test_preds_binary)
    else:
        auc_roc_test = roc_auc_score(all_test_labels, all_test_preds, average='weighted', multi_class='ovo')
        test_pred_classes = np.argmax(all_test_preds, axis=1)
        test_acc = accuracy_score(all_test_labels, test_pred_classes)
    
    log_roc_auc(all_test_labels, all_test_preds, original_labels, task,log_name=f'Testing ROC-AUC for {task} {group_name}', multilabel=multi_label, group_name=group_name)
    wandb.log({"Testing ROC_AUC_Score": auc_roc_test})
    #log_confusion_matrix(all_test_labels, all_test_preds, log_name="Testing Confusion Matrix")
    print(f"Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}")
