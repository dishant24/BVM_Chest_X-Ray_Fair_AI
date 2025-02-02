import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.io import read_image
from tqdm import tqdm
import torchvision
import torch.nn as nn


torch.cuda.empty_cache()

def file_load(training_file_path, demographic_data_path):
     training_data = pd.read_csv(training_file_path)
     demographic_data = pd.read_csv(demographic_data_path)
     return training_data, demographic_data

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
     training_data.drop_duplicates(subset=['subject_id'], inplace=True, keep='first')
     training_data_merge = training_data.merge(demographic_data, on='subject_id')
     training_data_merge['No Finding'].fillna(value=0,inplace=True)

     return training_data_merge

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

def store_images_labels(training_data_merge):
     data_images = []
     paths = tqdm(training_data_merge['Path'], desc="Loading images")
     for path in paths:
          full_path = '../datasets' + '/' + str(path)
          img = read_image(full_path)
          data_images.append(img)
          paths.set_postfix({'Loaded': len(data_images)})

     data_labels = training_data_merge['No Finding'].values
     data_labels = torch.tensor(data_labels, dtype=torch.float32)

     return data_images, data_labels

def prepare_dataloaders(data_images, labels):
     transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda x: x.to(torch.float32))
])
     dataset = MyDataset(data_images,labels,transform)

     train_size = int(0.7 * len(dataset)) 
     val_size = int(0.10 * len(dataset))  
     test_size = len(dataset) - train_size - val_size  
     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
     train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
     val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

     return train_loader, val_loader, test_loader

def model_building(in_feature, out_feature, weights=None):

     model = torchvision.models.densenet121(weights=weights)
     model.features.conv0 = nn.Conv2d(in_feature, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
     model.classifier = nn.Linear(in_features=1024, out_features=out_feature)

     return model

def model_training(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None):
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Training", leave=False)
        for inputs, labels in train_loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs).to(device)
            loss = criterion(outputs, labels.float().view(-1, 1).to(device))
        
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float().to(device)
            correct += (predicted == labels.view(-1, 1)).sum().item()
            total += labels.size(0)
            
            train_loop.set_postfix(loss=loss.item(), acc=correct / total)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_loop = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Validation", leave=False)
        with torch.no_grad():
            for inputs, labels in val_loop:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels.float().view(-1, 1).to(device))

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float().to(device)
                val_correct += (predicted == labels.view(-1, 1)).sum().item()
                val_total += labels.size(0)
                
                val_loop.set_postfix(loss=loss.item(), acc=val_correct / val_total)
        
        val_loss /= len(val_loader)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")



if __name__ == '__main__':

    print("hello")
    training_file_path =  r'..\datasets\CheXpert-v1.0-small\train.csv'
    demographic_data_path = r'..\datasets\CheXpert-v1.0-small\demographics_CXP.csv'
    training_data, demographic_data = file_load(training_file_path, demographic_data_path)
    training_data_merge = merge_dataframe(training_data, demographic_data)

    data_images,labels = store_images_labels(training_data_merge)
    train_loader, val_loader, test_loader = prepare_dataloaders(data_images,labels)
    model = model_building(in_feature=1, out_feature=1)

    lr = 0.0001
    loss = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(),lr)

    model_training(model,train_loader,val_loader,loss,optimiser,10)
    print("end")
