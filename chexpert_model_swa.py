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
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb



torch.cuda.empty_cache()
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


# Select the single subject_id per patient which has most positive disease 
def sampling_datasets(training_dataset):

    training_dataset = training_dataset.groupby('subject_id', group_keys=False).apply(select_most_positive_sample)
    training_dataset.drop(columns=['positive_count'], inplace=True, errors='ignore')
    
    return training_dataset


# Merge the data with demographic data
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
    'Support Devices']].fillna(0.0) == 1.0).astype(int)  # In The limits of fair medical imaging paper they treat uncertain label as negative and fill NA with 0.

    #Select only Frontal View 
    traning_dataset = traning_dataset[traning_dataset['Frontal/Lateral'] == 'Frontal']

    return traning_dataset

def split_and_save_datasets(dataset, train_path='train.csv', val_path='val.csv', val_size=0.1, random_seed=42):

    train_data, val_data = train_test_split(dataset, test_size=val_size, random_state=random_seed)
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)



def store_diagnostic_images_labels(dataset, path, device):
    """
    Loads images and processes diagnostic labels.

    Args:
    - training_data_merge (DataFrame): Contains image paths and diagnostic labels.

    Returns:
    - data_images (List[Tensor]): List of image tensors.
    - data_labels (Tensor): Tensor of diagnostic class labels.
    """

    
    if not os.path.exists(path):
        data_images = []
        paths = tqdm(dataset['Path'], desc="Loading images")
        for path in paths:
            full_path = '/deep_learning/output/Sutariya/chexpert' + '/' + str(path)
            img = read_image(full_path)
            data_images.append(img)
            paths.set_postfix({'Loaded': len(data_images)})
        torch.save(data_images, path)
    else:
        data_images = torch.load(path, map_location=device, weights_only=True)

    data_labels = training_data_merge[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values
    data_labels = torch.tensor(data_labels, dtype=torch.float32)
    
    return data_images, data_labels


def store_race_images_labels(dataset, save_path, device):
    """
    Loads images and processes race labels for multi-class classification.

    Args:
    - training_data_merge (DataFrame): Contains image paths and race labels.

    Returns:
    - data_images (List[Tensor]): List of image tensors.
    - data_labels (Tensor): Tensor of race class labels.
    """
    label_encoder = LabelEncoder()

    # Keep only top 3 race categories
    top_races = dataset['race'].value_counts().index[:3]
    dataset = dataset[dataset['race'].isin(top_races)].copy()

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
        data_images = torch.load(save_path, map_location=device, weights_only=True)

    dataset['race_encoded'] = label_encoder.fit_transform(dataset['race'])
    data_labels = torch.tensor(dataset['race_encoded'].values, dtype=torch.long)
    
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


def prepare_dataloaders(data_images, labels, sampler=None, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
        transforms.Lambda(lambda i: i/255),
        transforms.Normalize(mean=[0.5062, 0.5062, 0.5062], std=[0.2873, 0.2873, 0.2873]), # Adapt to own standard deviation and mean to Chexpert
        transforms.Lambda(lambda i: i.to(torch.float32)),
        transforms.RandomResizedCrop((224,224), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(contrast=(0.7, 1.2)) # Randomly change the brightness, contrast, saturation and hue of an image
    ])

    dataset = MyDataset(data_images,labels,transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=shuffle, sampler=sampler)

    return data_loader

class DenseNet_Model(nn.Module):
     def __init__(self, weights, out_feature):
          super().__init__()
          self.weight = weights
          self.out_feature = out_feature
          self.encoder = torchvision.models.densenet121(weights=weights) # Adapt the architecture to initial paper: The limits of fair medical imaging and almost all other papers
          self.relu = nn.ReLU()
          self.clf = nn.Linear(1000, out_feature)

     def encode(self, x):
          return self.encoder(x)

     def forward(self, x):
          z = self.encode(x)
          z = self.relu(z)
          return self.clf(z)

def log_roc_auc(y_true, y_scores, multilabel=True, log_name="roc_auc_curve", task_diagnostic=True):
    """
    Plots the ROC curve for multi-label or multi-class classification.

    Args:
    - y_true (np.array): True labels.
        * If multilabel=False: multi-hot encoded (multi-label).
        * If multilabel=True: single-label integer encoded.
    - y_scores (np.array): Model's predicted probabilities.
    - multilabel (bool): Set True for multi-class, False for multi-label.
    - log_name (str): Name for logging.

    Returns:
    - None
    """
    if task_diagnostic:
        labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']
    else:
        labels = demographic_data['race'].value_counts().index.values
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    fig, ax = plt.subplots(figsize=(7, 7))

    if not multilabel:
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
    ax.set_title("ROC Curve" + (" (Multi-Class)" if multilabel else " (Multi-Label)"))

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
    base_optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(base_optimizer, mode='min', factor=0.25, patience=5)
    early_stopper = EarlyStopper(patience=6)

    # SWA will be initialized just before starting SWA training
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

        if epoch == swa_start_epoch:
            print("Initializing SWA Model...")
            swa_model = AveragedModel(model)  # Initialize with latest trained weights
            swa_model = swa_model.to(device)
            swa_scheduler = SWALR(base_optimizer, anneal_strategy="cos", anneal_epochs=2, swa_lr=0.0001)

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
        log_roc_auc(all_train_labels, all_train_preds, log_name="Training ROC", multiclass=False, task_diagnostic=False)
        log_roc_auc(all_val_labels, all_val_preds, log_name="Validation ROC", multiclass=False, task_diagnostic=False)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train AUC: {auc_roc_train:.4f}, Train Acc: {train_acc:.4f}, Train Loss: {train_loss:.4f}, "
              f"Val AUC: {auc_roc_val:.4f}, Val Acc: {val_acc:.4f}, Val Loss: {val_loss:.4f}")
        
        # Apply SWA if within SWA start phase
        if epoch >= swa_start_epoch:
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
    if not early_stopped:
        print("Applying SWA...")
        state_dict = swa_model.state_dict()
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if k != "n_averaged"}  # Remove prefix & ignore "n_averaged"
        model.load_state_dict(new_state_dict)


    print("Training complete.")


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
    
    log_roc_auc(all_test_labels, all_test_preds, log_name='Testing ROC-AUC', multiclass=False)
    wandb.log({"Testing ROC_AUC_Score": auc_roc_test})
    #log_confusion_matrix(all_test_labels, all_test_preds, log_name="Testing Confusion Matrix")
    print(f"Test ROC-AUC Score: {auc_roc_test:.4f}, Testing Accuracy Score: {test_acc:.4f}")


# Help to build model for race prediction training
def model_build_race(path, model, device):

    state_dict = torch.load(path, map_location=device, weights_only=True)
    state_dict.pop("clf.weight", None)
    state_dict.pop("clf.bias", None)
    
    model.load_state_dict(state_dict, strict=False)

    for params in model.encoder.parameters():
        params.requires_grad = False

    return model
 

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 10
    training = True
    task_diagnostic = False

    if not os.path.exists("/deep_learning/output/Sutariya/chexpert/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/chexpert/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/Sutariya/chexpert/wandb")

    wandb.init(
    project="cxr_preprocessing_race",
    dir="/deep_learning/output/Sutariya/chexpert/wandb",
    config={
    "learning_rate": 0.0001,
    "Task": "diagnostic" if task_diagnostic else "Race",
    "save_model_file_name" : f'daignostic_model_swa_{random_state}' if task_diagnostic else f'race_model_swa_{random_state}',
    "Uncertain Labels" : "-1 = 0, NAN = 0",
    "epochs": 20,
    "Augmentation": 'Yes',
    "optimiser": "AdamW",
    "SWA":'Yes',
    "architecture": "DenseNet121",
    "dataset": "CheXpert",
    "Standardization": 'Yes'
    })

    if training:
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

        if task_diagnostic:
            train_data_images,train_labels = store_diagnostic_images_labels(training_dataset, '/deep_learning/output/Sutariya/chexpert/train_images_tensor.pt', device)
            val_data_images, val_labels = store_diagnostic_images_labels(validation_dataset, '/deep_learning/output/Sutariya/chexpert/validation_images_tensor.pt', device)  
            criterion = nn.BCEWithLogitsLoss()
            model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=14)
        else:
            train_data_images,train_labels = store_race_images_labels(training_dataset, '/deep_learning/output/Sutariya/chexpert/train_images_tensor.pt', device)
            val_data_images, val_labels = store_race_images_labels(validation_dataset, '/deep_learning/output/Sutariya/chexpert/validation_images_tensor.pt', device)
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=3)
            model = model_build_race('/deep_learning/output/Sutariya/chexpert/daignostic_model_swa_140.pth', base_model, device)

        train_data_shuffel_images, train_shuffel_labels = shuffle(train_data_images, train_labels, random_state=random_state)
        val_data_shuffel_images, val_shuffel_labels = shuffle(val_data_images, val_labels, random_state=random_state)
        train_loader = prepare_dataloaders(train_data_shuffel_images, train_shuffel_labels, shuffle=True)
        val_loader = prepare_dataloaders(val_data_shuffel_images, val_shuffel_labels, shuffle=True)

        model_training(model, train_loader, val_loader, criterion, 20, device=device, multi_label=True)
        
        torch.save(model.state_dict(), f'daignostic_model_swa_{random_state}.pth') if task_diagnostic else torch.save(model.state_dict(), f'race_model_swa_{random_state}.pth') 
    
    else:

        test_file_path =  '/deep_learning/output/Sutariya/chexpert/valid.csv'
        demographic_data_path = '/deep_learning/output/Sutariya/chexpert/demographics_CXP.csv'
        testing_data = file_load(test_file_path)
        demographic_data = file_load(demographic_data_path)

        testing_dataset = merge_dataframe(testing_data, demographic_data)
        testing_dataset = cleaning_datasets(testing_dataset)
        if task_diagnostic:
            test_data_images, test_labels = store_diagnostic_images_labels(testing_dataset, '/deep_learning/output/Sutariya/chexpert/test_images_tensor.pt', device)
            weights = torch.load('daignostic_model_swa_1.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=14)
        else:
            test_data_images, test_labels = store_race_images_labels(testing_dataset, '/deep_learning/output/Sutariya/chexpert/test_images_tensor.pt', device)
            weights = torch.load('race_model_swa_1.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=3)

        test_loader = prepare_dataloaders(test_data_images,test_labels, None, shuffle=False)
        test_model.load_state_dict(weights)

        testing_model(test_loader,test_model, device, multi_label=True)

