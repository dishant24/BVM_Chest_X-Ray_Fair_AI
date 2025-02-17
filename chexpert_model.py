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
from sklearn.metrics import roc_curve, auc, roc_auc_score,  f1_score
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
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
    data_images = []
    training_data_merge = training_data_merge[training_data_merge['race'].isin(training_data_merge['race'].value_counts()[:5].index)]
    paths = tqdm(training_data_merge['Path'], desc="Loading images")
    for path in paths:
        full_path = '../../datasets' + '/' + str(path)
        img = read_image(full_path)
        data_images.append(img)
        paths.set_postfix({'Loaded': len(data_images)})
    training_data_merge = pd.get_dummies(training_data_merge, columns=['race'], dtype=float)
    data_labels = training_data_merge[['race_Asian', 'race_Black', 'race_Other', 'race_Unknown', 'race_White']].values
    data_labels = torch.tensor(data_labels, dtype=torch.float32) 
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


def prepare_dataloaders(data_images, labels,shuffle=True):
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
    data_loader = DataLoader(dataset, batch_size=16, shuffle=shuffle)

    return data_loader

class DenseNet_Model(nn.Module):
     def __init__(self, weights, out_feature):
          super().__init__()
          self.weight = weights
          self.out_feature = out_feature
          self.encoder = torchvision.models.densenet121(weights=weights)
          self.clf = nn.Linear(1000, out_feature)

     
     def encode(self, x):
          return self.encoder(x)

     def forward(self, x):
          z = self.encode(x)
          return self.clf(z)

def log_roc_auc(y_true, y_scores, log_name="roc_auc_curve"):
    """
    Plots the ROC curve for multi-label classification.

    Args:
    - y_true (np.array): True labels (multi-hot encoded).
    - y_scores (np.array): Model's predicted probabilities.
    - log_name (str): Name for logging.

    Returns:
    - None
    """
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    num_classes = y_true.shape[1]
    fig, ax = plt.subplots(figsize=(7, 7))

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
    ax.set_title("Multi-Label ROC Curve")

    # Adjust legend size
    if num_classes > 10:
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.legend(loc="lower right", fontsize=10)

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



def model_training(model, train_loader, val_loader, num_epochs=10, device=None):
    """
    Trains a multi-label classification model using BCEWithLogitsLoss.
    
    Args:
    - model (nn.Module): The neural network model.
    - train_loader (DataLoader): Training data loader.
    - val_loader (DataLoader): Validation data loader.
    - num_epochs (int): Number of training epochs.
    - device (torch.device): Device to train on (CPU or GPU).
    """

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=6)
    early_stopper = EarlyStopper(patience=5)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        ### === Training Phase === ###

        model.train()
        train_loss = 0.0
        all_train_labels, all_train_preds = [], []

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)
        for batch_idx, (inputs, tr_labels) in enumerate(train_loop):
            
            inputs, tr_labels = inputs.to(device), tr_labels.to(device)
            assert isinstance(inputs, torch.Tensor), "inputs must be a torch.Tensor"
            assert isinstance(tr_labels, torch.Tensor), "labels must be a torch.Tensor"
            assert inputs.dim() >= 2, f"inputs must have at least 2 dimensions, got {inputs.shape}"
            assert tr_labels.dim() >= 2, f"labels must have at least 2 dimensions, got {tr_labels.shape}"
            assert torch.all((tr_labels >= 0) & (tr_labels <= 1)), "Labels must be binary (0 or 1)"
            assert torch.all((inputs >= -1.76) & (inputs <= 1.69)), "Inputs must be normalized in the range [-1.76, 1.69]"

            optimizer.zero_grad()
            outputs = model(inputs)
            assert outputs.shape == tr_labels.shape , "Output diemension does not match with label diemension"

            loss = criterion(outputs, tr_labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            tr_preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_train_labels.extend(tr_labels.cpu().numpy())
            all_train_preds.extend(tr_preds)
            train_loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        
        ### === Validation Phase === ###
        model.eval()
        val_loss = 0.0
        all_val_labels, all_val_preds = [], []

        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)
        with torch.no_grad():
            for batch_idx, (inputs, vl_labels) in enumerate(val_loop):
            
                inputs, vl_labels = inputs.to(device), vl_labels.to(device)
                assert isinstance(inputs, torch.Tensor), "inputs must be a torch.Tensor"
                assert isinstance(vl_labels, torch.Tensor), "labels must be a torch.Tensor"
                assert inputs.dim() >= 2, f"inputs must have at least 2 dimensions, got {inputs.shape}"
                assert vl_labels.dim() >= 2, f"labels must have at least 2 dimensions, got {vl_labels.shape}"
                assert torch.all((vl_labels >= 0) & (vl_labels <= 1)), "Labels must be binary (0 or 1)"
                assert torch.all((inputs >= -1.76) & (inputs <= 1.69)), "Inputs must be normalized in the range [-1.76, 1.69]"

                outputs = model(inputs)
                assert outputs.shape == vl_labels.shape , "Output diemension does not match with label diemension"
                
                loss = criterion(outputs, vl_labels)
                val_loss += loss.item()

                vl_preds = torch.sigmoid(outputs).detach().cpu().numpy()
                all_val_labels.extend(vl_labels.cpu().numpy())
                all_val_preds.extend(vl_preds)

                val_loop.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Compute AUC-ROC
        auc_roc_train = roc_auc_score(all_train_labels, all_train_preds)
        auc_roc_val = roc_auc_score(all_val_labels, all_val_preds)

        wandb.log({"Training AUC: ": auc_roc_train, "Validation AUC: ": auc_roc_val})
        log_roc_auc(all_train_labels, all_train_preds)
        log_roc_auc(all_val_labels, all_val_preds)

        # Compute Accuracy
        train_preds_binary = (np.array(all_train_preds) > 0.5).astype(int)
        val_preds_binary = (np.array(all_val_preds) > 0.5).astype(int)

        train_acc = f1_score(all_train_labels, train_preds_binary, average='macro')
        val_acc = f1_score(all_val_labels, val_preds_binary, average='macro')

        print(f"Epoch [{epoch+1}/{num_epochs}], Train AUC: {auc_roc_train:.4f}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, Val AUC: {auc_roc_val:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")

        if early_stopper.early_stop(val_loss):
            print("Early stopping triggered.")
            break


def testing_model(test_loader, model, device=None):
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
            preds = torch.sigmoid(outputs).detach().cpu().numpy() 

            all_test_labels.extend(labels.cpu().numpy())  
            all_test_preds.extend(preds) 

    auc_roc = roc_auc_score(all_test_labels, all_test_preds, average='macro')  
    test_preds_binary = (np.array(all_test_preds) > 0.5).astype(int)
    test_accuracy = f1_score(all_test_labels, test_preds_binary,  average='macro')
    log_roc_auc(all_test_labels, all_test_preds, log_name='Testing ROC-AUC')
    wandb.log({"Testing ROC_AUC_Score": auc_roc})

    print(f"Test ROC-AUC Score: {auc_roc:.4f}, Testing Accuracy Score: {test_accuracy:.4f}")
    return auc_roc


if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
    project="cxr_preprocessing",
    config={
    "learning_rate": 0.001,
    "epochs": 20,
    "Augmentation": 'Yes',
    "optimiser": "AdamW",
    "architecture": "DenseNet121",
    "dataset": "CheXpert",
    "Standardization": 'Yes'
    }
    )

    # training_file_path =  r'..\..\datasets\CheXpert-v1.0-small\train.csv'
    # demographic_data_path = r'..\..\datasets\CheXpert-v1.0-small\demographics_CXP.csv'
    # training_data = file_load(training_file_path)
    # demographic_data = file_load(demographic_data_path)

    # training_data_merge = merge_dataframe(training_data, demographic_data)

    # training_data_merge = cleaning_datasets(training_data_merge)

    # training_dataset = sampling_datasets(training_data_merge)

    # split_and_save_datasets(training_dataset,train_path='../../datasets/train_clean_dataset.csv', val_path='../../datasets/validation_clean_dataset.csv')
    
    training_dataset = pd.read_csv('../../datasets/train_clean_dataset.csv')
    validation_dataset = pd.read_csv('../../datasets/validation_clean_dataset.csv')
    train_data_images,train_labels = store_diagnostic_images_labels(training_dataset)
    val_data_images, val_lables = store_diagnostic_images_labels(validation_dataset)
    train_loader = prepare_dataloaders(train_data_images,train_labels, shuffle=True)
    val_loader = prepare_dataloaders(val_data_images, val_lables,shuffle=False)

    model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=13)

    model_training(model, train_loader, val_loader, 20, device=device)

    torch.save(model.state_dict(), 'diagnostic_model.pth')

    #testing_model(test_loader,test_model,'cpu')
    