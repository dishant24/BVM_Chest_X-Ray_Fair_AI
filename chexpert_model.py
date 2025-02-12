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
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
import wandb


torch.cuda.empty_cache()

def file_load(training_file_path, demographic_data_path):
     training_data = pd.read_csv(training_file_path)
     demographic_data = pd.read_csv(demographic_data_path)
     return training_data, demographic_data

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
     data_labels = torch.argmax(data_labels, dim=1)

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
    data_labels = data_labels.argmax(dim=1) 

    return data_images, data_labels



class MyDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform 
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = self.data[idx] / 255.0
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)

        return img, label


def prepare_dataloaders(data_images, labels):
     transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.to(torch.float32)),
        transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

     dataset = MyDataset(data_images,labels,transform)

     train_size = int(0.7 * len(dataset)) 
     val_size = int(0.10 * len(dataset))  
     test_size = len(dataset) - train_size - val_size  
     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

     return train_loader, val_loader, test_loader

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



def log_roc_auc(y_true, y_scores, log=True, log_name="roc_auc_curve"):
    y_scores = np.array(y_scores)
    classes = np.unique(y_true) 
    y_true_bin = label_binarize(y_true, classes=classes)
    total_roc = 0

    fig_width = 5 + min(len(classes) // 3, 5)
    fig_height = 5 + min(len(classes) // 5, 5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i, class_label in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        total_roc += roc_auc
        ax.plot(fpr, tpr, lw=2, label=f"Class {class_label} (AUC = {roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-Class ROC Curve")

    if len(classes) > 10:
        ax.legend(loc="lower right", fontsize=8)
    else:
        ax.legend(loc="lower right", fontsize=10)

    if log:
        wandb.log({log_name: wandb.Image(fig)})
    else:
        print(f"{log_name} : {total_roc/len(classes)}")

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
        elif validation_loss > (self.min_validation_loss):
            self.counter += 1
            if self.counter > self.patience:
                return True
        return False


def model_training(model, train_loader, val_loader, num_epochs=10, device=None, is_binary=False):
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    early_stopper = EarlyStopper(patience=3)
    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    all_train_labels, all_train_preds = [], []
    all_val_labels, all_val_preds = [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)
        for inputs, tr_labels in train_loop:
            inputs, tr_labels = inputs.to(device), tr_labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs).to(device)

            # Convert predictions & labels
            if is_binary:
                tr_labels = tr_labels.unsqueeze(dim=1)
                tr_preds = torch.sigmoid(outputs).detach().cpu().numpy()
            else:
                tr_preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            all_train_labels.extend(tr_labels.cpu().numpy())
            all_train_preds.extend(tr_preds)


            loss = criterion(outputs, tr_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            train_loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)
        with torch.no_grad():
            for inputs, vl_labels in val_loop:
                inputs, vl_labels = inputs.to(device), vl_labels.to(device)

                outputs = model(inputs).to(device)
                if is_binary:
                    vl_labels = vl_labels.unsqueeze(dim=1)
                    vl_preds = torch.sigmoid(outputs).detach().cpu().numpy()
                else:
                    vl_preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()

                all_val_labels.extend(vl_labels.cpu().numpy())
                all_val_preds.extend(vl_preds)

                loss = criterion(outputs, vl_labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if early_stopper.early_stop(val_loss):
            break

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        log_roc_auc(all_train_labels, all_train_preds, log=True, log_name='Training ROC-AUC')
        log_roc_auc(all_val_labels, all_val_preds, log=True, log_name='Validation ROC-AUC')

    log_roc_auc(all_train_labels, all_train_preds, log=False, log_name='Training ROC-AUC')
    log_roc_auc(all_val_labels, all_val_preds, log=False, log_name='Validation ROC-AUC')


def testing_model(test_loader, model, device=None, is_binary=False):
    model.to(device)
    model.eval()

    all_test_labels, all_test_preds = [], []

    loader = tqdm(test_loader)
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            if is_binary:
                labels = labels.view(-1, 1)
                predicted = torch.sigmoid(outputs).detach().cpu().numpy()
            else:
                predicted = torch.softmax(outputs, dim=1).detach().cpu().numpy()

            all_test_labels.extend(labels.cpu().numpy())
            all_test_preds.extend(predicted)

    log_roc_auc(all_test_labels, all_test_preds, log=True, log_name='Testing ROC-AUC')
    log_roc_auc(all_test_labels, all_test_preds, log=False, log_name='Testing ROC-AUC')


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
    project="cxr_preprocessing",
    config={
    "learning_rate": 0.0001,
    "epochs": 20,
    "Augmentation": 'No',
    "architecture": "DenseNet121",
    "dataset": "CheXpert",
    "Standardization": 'No'
    }
    )

    training_file_path =  r'..\..\datasets\CheXpert-v1.0-small\train.csv'
    demographic_data_path = r'..\..\datasets\CheXpert-v1.0-small\demographics_CXP.csv'
    training_data, demographic_data = file_load(training_file_path, demographic_data_path)
    training_data_merge = merge_dataframe(training_data, demographic_data)

    training_dataset = sampling_datasets(training_data_merge)
    training_dataset = cleaning_datasets(training_dataset)
    
    data_images,labels = store_diagnostic_images_labels(training_dataset)
    train_loader, val_loader, test_loader = prepare_dataloaders(data_images,labels)
    model = DenseNet_Model(weights=None, out_feature=13)

    model_training(model,train_loader,val_loader,20,is_binary=False, device=device)

    torch.save(model.state_dict(), 'diagnostic_model.pth')

    #testing_model(test_loader,test_model,'cpu')
    