import pandas as pd
import numpy as np
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

    #Replace -1 value with 1
    traning_dataset[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']] = traning_dataset[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']].replace(-1, 1)

    #Select only Frontal View
    traning_dataset = traning_dataset[traning_dataset['Frontal/Lateral'] == 'Frontal']

    traning_dataset.fillna(0, inplace=True)

    return traning_dataset


def split_and_save_datasets(dataset, train_path='train.csv', val_path='val.csv', val_size=0.1, random_seed=42):

    train_data, val_data = train_test_split(dataset, test_size=val_size, random_state=random_seed)
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)


def store_diagnostic_images_labels(training_data_merge):
    data_images = []
    paths = tqdm(training_data_merge['Path'], desc="Loading images")
    for path in paths:
        full_path = '../../datasets' + '/' + str(path)
        img = read_image(full_path)
        data_images.append(img)
        paths.set_postfix({'Loaded': len(data_images)})

    data_labels = training_data_merge[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']].values
    data_labels = torch.tensor(data_labels, dtype=torch.float32)
    return data_images, data_labels


def store_race_images_labels(training_data_merge):
    """
    Loads images and processes race labels for multi-class classification.

    Args:
    - training_data_merge (DataFrame): Contains image paths and race labels.

    Returns:
    - data_images (List[Tensor]): List of image tensors.
    - data_labels (Tensor): Tensor of race class labels.
    """
    data_images = []
    label_encoder = LabelEncoder()

    # Keep only top 5 race categories
    top_races = training_data_merge['race'].value_counts().index[:5]
    training_data_merge = training_data_merge[training_data_merge['race'].isin(top_races)]

    paths = tqdm(training_data_merge['Path'], desc="Loading images")
    
    for path in paths:
        full_path =  '../../datasets' + '/' + str(path)
        img = read_image(full_path)
        data_images.append(img)
        paths.set_postfix({'Loaded': len(data_images)})

    # Convert race categories into categorical integer labels
    training_data_merge['race_encoded'] = label_encoder.fit_transform(training_data_merge['race'])
    data_labels = torch.tensor(training_data_merge['race_encoded'].values, dtype=torch.long)

    return data_images, data_labels


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


def prepare_dataloaders(data_images, labels, sampler= None, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
        transforms.RandomResizedCrop((256,256), scale=(0.7, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
          self.layer1 = torch.nn.Linear(1000, 512)
          self.clf = nn.Linear(512, out_feature)

     
     def encode(self, x):
          return self.encoder(x)

     def forward(self, x):
          z = self.encode(x)
          z = self.relu(z)
          z = self.layer1(z)
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
            ax.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")
    
    else:  # Multi-label case
        num_classes = y_true.shape[1]
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f"Class {i} (AUC = {roc_auc:.2f})")

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    early_stopper = EarlyStopper(patience=5)

    for epoch in range(num_epochs):
        ### === Training Phase === ###
        model.train()
        train_loss = 0.0
        all_train_labels, all_train_preds = [], []

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_loop):
            inputs, labels = inputs.to(device), labels.to(device)
            assert isinstance(inputs, torch.Tensor), "inputs must be a torch.Tensor"
            assert isinstance(labels, torch.Tensor), "labels must be a torch.Tensor"
            assert inputs.dim() >= 2, f"inputs must have at least 2 dimensions, got {inputs.shape}"
            assert labels.dim() >= 2, f"labels must have at least 2 dimensions, got {labels.shape}"

            optimizer.zero_grad()

            outputs = model(inputs)
            assert outputs.shape[0] == labels.shape[0], "Batch size mismatch"
            
            loss = loss_function(outputs, labels)  # Compute loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Convert predictions
            if multi_label:
                preds = torch.sigmoid(outputs).detach().cpu().numpy()  # Multi-label: Use sigmoid
            else:
                preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()  # Multi-class: Use softmax

            all_train_labels.extend(labels.cpu().numpy())
            all_train_preds.extend(preds)
            train_loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)

        ### === Validation Phase === ###
        model.eval()
        val_loss = 0.0
        all_val_labels, all_val_preds = [], []

        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_loop):
                inputs, labels = inputs.to(device), labels.to(device)
                assert isinstance(inputs, torch.Tensor), "inputs must be a torch.Tensor"
                assert isinstance(labels, torch.Tensor), "labels must be a torch.Tensor"
                assert inputs.dim() >= 2, f"inputs must have at least 2 dimensions, got {inputs.shape}"
                assert labels.dim() >= 2, f"labels must have at least 2 dimensions, got {labels.shape}"

                
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                val_loss += loss.item()

                if multi_label:
                    preds = torch.sigmoid(outputs).detach().cpu().numpy()  # Multi-label
                else:
                    preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()  # Multi-class

                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(preds)
                val_loop.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Compute AUC-ROC (Multi-label) or Accuracy (Multi-class)
        if multi_label:
            auc_roc_train = roc_auc_score(all_train_labels, all_train_preds, average="weighted")
            auc_roc_val = roc_auc_score(all_val_labels, all_val_preds, average="weighted")
        else:
            auc_roc_train = roc_auc_score(all_train_labels, all_train_preds, average='weighted', multi_class='ovr')
            auc_roc_val = roc_auc_score(all_val_labels, all_val_preds, average='weighted', multi_class='ovr')

        # Log results
        wandb.log({"Training AUC": auc_roc_train, "Validation AUC": auc_roc_val})
        log_roc_auc(all_train_labels, all_train_preds, log_name="Training ROC")
        log_roc_auc(all_val_labels, all_val_preds, log_name="Validation ROC")
        log_confusion_matrix(all_train_labels, all_train_preds, log_name="Training Confusion Matrix")
        log_confusion_matrix(all_val_labels, all_val_preds, log_name="Validation Confusion Matrix")

        # Compute F1 Score (Thresholding required for multi-label)
        if multi_label:
            train_preds_binary = (np.array(all_train_preds) > 0.4).astype(int)
            val_preds_binary = (np.array(all_val_preds) > 0.4).astype(int)
            train_acc = f1_score(all_train_labels, train_preds_binary, average='weighted')
            val_acc = f1_score(all_val_labels, val_preds_binary, average='weighted')
        else:
            train_pred_classes = np.argmax(all_train_preds, axis=1)
            val_pred_classes = np.argmax(all_val_preds, axis=1)
            train_acc = f1_score(all_train_labels, train_pred_classes, average='weighted')
            val_acc = f1_score(all_val_labels, val_pred_classes, average='weighted')

        print(f"Epoch [{epoch+1}/{num_epochs:.4f}], Train AUC: {auc_roc_train:.4f}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val AUC: {auc_roc_val:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping
        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered.")
            break


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

    for params in model.layer1.parameters():
        params.requires_grad = False

    return model

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb.init(
    # project="cxr_preprocessing",
    # config={
    # "learning_rate": 0.001,
    # "Task": "Race",
    # "epochs": 20,
    # "Augmentation": 'Yes',
    # "optimiser": "AdamW",
    # "architecture": "DenseNet121",
    # "dataset": "CheXpert",
    # "Standardization": 'Yes'
    # }
    # )

    # training_file_path =  r'..\..\datasets\CheXpert-v1.0-small\train.csv'
    # demographic_data_path = r'..\..\datasets\CheXpert-v1.0-small\demographics_CXP.csv'
    # training_data = file_load(training_file_path)
    # demographic_data = file_load(demographic_data_path)

    # training_data_merge = merge_dataframe(training_data, demographic_data)

    # training_data_merge = cleaning_datasets(training_data_merge)

    # training_dataset = sampling_datasets(training_data_merge)

    # split_and_save_datasets(training_dataset,train_path='../../datasets/train_clean_dataset.csv', val_path='../../datasets/validation_clean_dataset.csv')
    
    # training_dataset = pd.read_csv('../../datasets/train_clean_dataset.csv')
    # validation_dataset = pd.read_csv('../../datasets/validation_clean_dataset.csv')
    # train_data_images,train_labels = store_race_images_labels(training_dataset)
    # val_data_images, val_labels = store_race_images_labels(validation_dataset)
    # print(np.unique(train_labels, return_counts=True)) 
    # print(np.unique(val_labels, return_counts=True)) 

    # class_counts = np.bincount(train_labels)
    # weights = 1.0 / class_counts

    # train_sample_weights = weights[train_labels]
    # train_sampler = WeightedRandomSampler(train_sample_weights, len(train_sample_weights))
    # val_sample_weights = weights[val_labels]
    # val_sampler = WeightedRandomSampler(val_sample_weights, len(val_sample_weights))
    # criterion = nn.CrossEntropyLoss()
    
    # train_loader = prepare_dataloaders(train_data_images,train_labels, train_sampler, shuffle=False)
    # val_loader = prepare_dataloaders(val_data_images, val_labels, val_sampler, shuffle=False)

    # model = DenseNet_Model(weights=None, out_feature=5)
    # model = model_build_transfer(model)

    # model_training(model, train_loader, val_loader, criterion,  20, device=device, multi_label=False)

    # torch.save(model.state_dict(), 'race_model.pth')

    test_file_path =  r'..\..\datasets\CheXpert-v1.0-small\valid.csv'
    demographic_data_path = r'..\..\datasets\CheXpert-v1.0-small\demographics_CXP.csv'
    testing_data = file_load(test_file_path)
    demographic_data = file_load(demographic_data_path)

    testing_dataset = merge_dataframe(testing_data, demographic_data)
    testing_dataset = cleaning_datasets(testing_dataset)
    testing_dataset = sampling_datasets(testing_dataset)
    test_data_images, test_labels = store_race_images_labels(testing_dataset)
    test_loader = prepare_dataloaders(test_data_images,test_labels, None, shuffle=False)

    print(np.unique(test_labels, return_counts=True))

    # weights = torch.load('race_model.pth', device)
    # test_model = DenseNet_Model(weights, 5)

    # testing_model(test_loader,test_model, device, multi_label=False)
    