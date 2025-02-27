import pandas as pd
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.io import read_image
from tqdm import tqdm
import torchvision
import torch.nn as nn
from skimage import exposure
from skimage.exposure import match_histograms
torch.cuda.empty_cache()
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score, accuracy_score
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb



def file_load(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

def select_most_positive_sample(group):

    disease_columns = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture'
    ]
    
    group['positive_count'] = group[disease_columns].sum(axis=1)
    
    positive_cases = group[group['positive_count'] > 0]
    
    if not positive_cases.empty:

        selected_sample = positive_cases.loc[positive_cases['positive_count'].idxmax()]
    else:
        selected_sample = group.sample(n=1).iloc[0]
    
    return selected_sample

def sampling_datasets(training_dataset):

    training_dataset = training_dataset.groupby('subject_id', group_keys=False).apply(select_most_positive_sample)
    training_dataset.drop(columns=['positive_count'], inplace=True, errors='ignore')
    
    return training_dataset


def merge_dataframe(training_data, demographic_data):
    path = training_data['Path']
    patientid = []
    for i in path:
        id = i.split(sep='/')[2]
        id = id.replace("patient", "")
        patientid.append(float(id))

    temp_patient = pd.DataFrame(patientid,columns=['patient_id'])
    training_data = training_data.reset_index(drop=True)
    training_data['subject_id'] = temp_patient['patient_id']
    training_data_merge = training_data.merge(demographic_data, on='subject_id')
    return training_data_merge


def cleaning_datasets(traning_dataset):

    traning_dataset[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']] = (traning_dataset[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']].fillna(0.0) == 1.0).astype(int)

    #Select only Frontal View
    traning_dataset = traning_dataset[traning_dataset['Frontal/Lateral'] == 'Frontal']

    return traning_dataset

def split_and_save_datasets(dataset, train_path='train.csv', val_path='val.csv', val_size=0.1, random_seed=42):

    train_data, val_data = train_test_split(dataset, test_size=val_size, random_state=random_seed)
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)


def store_diagnostic_images_labels(training_data_merge, path):
    
    data_images = torch.load(path)
    data_labels = training_data_merge[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values
    data_labels = torch.tensor(data_labels, dtype=torch.float32)
    return data_images, data_labels


def store_race_images_labels(training_data_merge, path):
    """
    Loads images and processes race labels for multi-class classification.

    Args:
    - training_data_merge (DataFrame): Contains image paths and race labels.

    Returns:
    - data_images (List[Tensor]): List of image tensors.
    - data_labels (Tensor): Tensor of race class labels.
    """
    label_encoder = LabelEncoder()
    data_images = torch.load(path)
    # Keep only top 5 race categories
    top_races = training_data_merge['race'].value_counts().index[:5]
    training_data_merge = training_data_merge[training_data_merge['race'].isin(top_races)]

    # Convert race categories into categorical integer labels
    training_data_merge['race_encoded'] = label_encoder.fit_transform(training_data_merge['race'])
    data_labels = torch.tensor(training_data_merge['race_encoded'].values, dtype=torch.long)

    return data_images, data_labels

def save_image_tensor(dataset, save_path):
    if not os.path.exists(save_path):
        data_images = []
        paths = tqdm(dataset['Path'], desc="Loading images")
        for path in paths:
            full_path = '/deep_learning/output/Sutariya/chexpert' + '/' + str(path)
            img = read_image(full_path)
            data_images.append(img)
            paths.set_postfix({'Loaded': len(data_images)})
        torch.save(data_images, save_path)
    else:
        print(f'File {save_path} already exists. Skipping save.')

        
class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)

        return img, label


def prepare_dataloaders(data_images, labels, sampler=None, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
        transforms.RandomResizedCrop((200,200), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(contrast=(0.8, 1.2)),
        transforms.RandomRotation(20),
        transforms.Lambda(lambda i: i/255),
        transforms.Lambda(lambda i: i.to(torch.float32)),
        transforms.Normalize(mean=[0.5062, 0.5062, 0.5062], std=[0.2873, 0.2873, 0.2873])
    ])

    dataset = MyDataset(data_images,labels,transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=shuffle, sampler=sampler)

    return data_loader

class DenseNet_Model(nn.Module):
     def __init__(self, weights, out_feature):
          super().__init__()
          self.weight = weights
          self.out_feature = out_feature
          self.encoder = torchvision.models.densenet121(weights=weights)
          self.relu = nn.ReLU()
          self.clf = nn.Linear(1000, out_feature)

     def encode(self, x):
          return self.encoder(x)

     def forward(self, x):
          z = self.encode(x)
          z = self.relu(z)
          return self.clf(z)


def log_roc_auc(y_true, y_scores, multiclass=True, log_name="roc_auc_curve"):
    """
    Plots the ROC curve for multi-label or multi-class classification.

    Args:
    - y_true (np.array): True labels.
        * If multiclass=False: multi-hot encoded (multi-label).
        * If multiclass=True: single-label integer encoded.
    - y_scores (np.array): Model's predicted probabilities.
    - multiclass (bool): Set True for multi-class, False for multi-label.
    - log_name (str): Name for logging.

    Returns:
    - None
    """
    labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    fig, ax = plt.subplots(figsize=(7, 7))

    if multiclass:
        # Binarize y_true for one-vs-rest ROC curve
        num_classes = y_scores.shape[1]
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.2f})")
    
    else:  # Multi-label case
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

    # Add diagonal line
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve" + (" (Multi-Class)" if multiclass else " (Multi-Label)"))

    # Adjust legend size
    ax.legend(loc="lower right", fontsize=8 if num_classes > 10 else 10)

    wandb.log({log_name: wandb.Image(fig)})
    plt.close(fig)


def log_confusion_matrix(y_true, y_pred, multiclass=True, log_name="confusion_matrix"):
    """s
    Plots the confusion matrix for multi-label or multi-class classification.

    Args:
    - y_true (np.array): True labels.
        * If multiclass=False: multi-hot encoded (multi-label).
        * If multiclass=True: single-label integer encoded.
    - y_pred (np.array): Predicted labels.
    - multiclass (bool): Set True for multi-class, False for multi-label.
    - log_name (str): Name for logging.

    Returns:
    - None
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if multiclass:
        # Multi-class confusion matrix
        y_pred = np.argmax(y_pred, axis=1)
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6,6))
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
            axes[i].set_title(f'Class {i}')
            axes[i].set_xlabel("Predicted")
            axes[i].set_ylabel("Actual")

    plt.tight_layout()
    wandb.log({log_name: wandb.Image(fig)})
    plt.close(fig)


class EarlyStopper:
    def __init__(self, patience=1):
        self.patience = patience
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= self.min_validation_loss:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
        


def model_training(model, train_loader, val_loader, loss_function, num_epochs=10, device=None, multi_label=True):
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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)
    early_stopper = EarlyStopper(patience=5)
    
    # SWA Setup
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(optimizer, anneal_strategy="cos", anneal_epochs=3, swa_lr=0.005)
    swa_start_epoch = max(num_epochs - 3, 0)

    best_model_weights = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        ### === Training Phase === ###
        model.train()
        train_loss = 0.0
        all_train_labels, all_train_preds = [], []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Convert predictions
            preds = torch.sigmoid(outputs).detach().cpu().numpy() if multi_label else torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)
        train_loss /= len(train_loader)

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
        scheduler.step(val_loss)
        
        # Compute Metrics
        auc_roc_train = roc_auc_score(all_train_labels, all_train_preds, average="weighted")
        auc_roc_val = roc_auc_score(all_val_labels, all_val_preds, average="weighted")

        # Apply SWA in the last 3 epochs or if early stopping is about to trigger
        if epoch >= swa_start_epoch:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        
        # Print Metrics
        print(f"Epoch [{epoch+1}/{num_epochs}], Train AUC: {auc_roc_train:.4f}, Train Loss: {train_loss:.4f}, Val AUC: {auc_roc_val:.4f}, Val Loss: {val_loss:.4f}")

        # Early Stopping Check
        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered. Switching to SWA.")
            break

    # Load best model before switching to SWA
    model.load_state_dict(best_model_weights)
    swa_model.train()  

    with torch.no_grad():
        for inputs, _ in train_loader:
            inputs = inputs.to(device)
            _ = swa_model(inputs) 

    # Switch to SWA model for final evaluation
    model = swa_model
    print("Training complete...")


def testing_model(test_loader, model, device=None, multi_label= True):
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
    
    loader = tqdm(test_loader, desc="Testing Model")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)  
            if multi_label:
                preds = torch.sigmoid(outputs).detach().cpu().numpy()
            else:
                preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            all_test_labels.extend(labels.cpu().numpy())  
            all_test_preds.extend(preds) 

    if multi_label:
        auc_roc_test = roc_auc_score(all_test_labels, all_test_preds, average="macro")
        test_preds_binary = (np.array(all_test_preds) > 0.4).astype(int)
        test_acc = accuracy_score(all_test_labels, test_preds_binary)
    else:
        auc_roc_test = roc_auc_score(all_test_labels, all_test_preds, average='macro', multi_class='ovo')
        test_pred_classes = np.argmax(all_test_preds, axis=1)
        test_acc = accuracy_score(all_test_labels, test_pred_classes)
    
    log_roc_auc(all_test_labels, all_test_preds, log_name='Testing ROC-AUC')
    wandb.log({"Testing ROC_AUC_Score": auc_roc_test})
    log_confusion_matrix(all_test_labels, all_test_preds, log_name="Testing Confusion Matrix")
    print(f"Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}")

def model_build_transfer(model):

    state_dict = torch.load("diagnostic_model.pth", map_location=device)
    state_dict.pop("clf.weight", None)
    state_dict.pop("clf.bias", None)
    
    model.load_state_dict(state_dict, strict=False)

    for params in model.encoder.parameters():
        params.requires_grad = False

    return model

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    separate_split = True

    if not os.path.exists("/deep_learning/output/Sutariya/chexpert/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/chexpert/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/Sutariya/chexpert/wandb")

    wandb.init(
    project="cxr_preprocessing",
    dir="/deep_learning/output/Sutariya/chexpert/wandb",
    config={
    "learning_rate": 0.001,
    "Task": "diagnostic",
    "Uncertain Labels" : "-1 = 1, NAN = 0",
    "epochs": 30,
    "Augmentation": 'Yes',
    "optimiser": "SGD",
    "SWA":'Yes',
    "architecture": "DenseNet121",
    "dataset": "CheXpert",
    "Standardization": 'Yes'
    }
    )

    # Paths to the output files
    train_output_path = '/deep_learning/output/Sutariya/chexpert/train_clean_dataset.csv'
    val_output_path = '/deep_learning/output/Sutariya/chexpert/validation_clean_dataset.csv'

    # Run the code block only if the output files do not exist
    if not (os.path.exists(train_output_path) and os.path.exists(val_output_path)):
        
        # Input file paths
        training_file_path = '/deep_learning/output/Sutariya/chexpert/train.csv'
        demographic_data_path = '/deep_learning/output/Sutariya/chexpert/demographics_CXP.csv'
        
        # Load the data
        training_data = file_load(training_file_path)
        demographic_data = file_load(demographic_data_path)

        # Merge and clean the data
        training_data_merge = merge_dataframe(training_data, demographic_data)
        training_data_clean = cleaning_datasets(training_data_merge)

        # Sample the data and split
        training_dataset = sampling_datasets(training_data_clean)
        split_and_save_datasets(training_dataset, 
                                train_path=train_output_path, 
                                val_path=val_output_path)
    else:
        print(f'Files {train_output_path} && {val_output_path} already exists. Skipping save.')

    training_dataset = pd.read_csv(train_output_path)
    validation_dataset = pd.read_csv(val_output_path)
    
    save_image_tensor(dataset=training_dataset, save_path='/deep_learning/output/Sutariya/chexpert/train_images_tensor.pt')
    save_image_tensor(dataset=validation_dataset, save_path='/deep_learning/output/Sutariya/chexpert/validation_images_tensor.pt')

    train_data_images,train_labels = store_diagnostic_images_labels(training_dataset, '/deep_learning/output/Sutariya/chexpert/train_images_tensor.pt')
    val_data_images, val_labels = store_diagnostic_images_labels(validation_dataset, '/deep_learning/output/Sutariya/chexpert/validation_images_tensor.pt')
    train_loader = prepare_dataloaders(train_data_images, train_labels, shuffle=True)
    val_loader = prepare_dataloaders(val_data_images, val_labels)
    
    criterion = nn.BCEWithLogitsLoss()
    model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=14)

    model_training(model, train_loader, val_loader, criterion, 30, device=device, multi_label=True)

    torch.save(model.state_dict(), 'daignostic_model_swa.pth')

    # test_file_path =  r'..\..\datasets\CheXpert-v1.0-small\valid.csv'
    # demographic_data_path = r'..\..\datasets\CheXpert-v1.0-small\demographics_CXP.csv'
    # testing_data = file_load(test_file_path)
    # demographic_data = file_load(demographic_data_path)

    # testing_dataset = merge_dataframe(testing_data, demographic_data)
    # testing_dataset = cleaning_datasets(testing_dataset)
    # testing_dataset = sampling_datasets(testing_dataset)
    # test_data_images, test_labels = store_race_images_labels(testing_dataset)
    # test_loader = prepare_dataloaders(test_data_images,test_labels, None, shuffle=False)

    # print(np.unique(test_labels, return_counts=True))

    # weights = torch.load('race_model.pth', device)
    # test_model = DenseNet_Model(weights, 5)

    # testing_model(test_loader,test_model, device, multi_label=False)

