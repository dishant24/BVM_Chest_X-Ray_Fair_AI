import pandas as pd
import numpy as np
import copy
import os
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
torch.cuda.empty_cache()
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import wandb
import sys
import torchvision
import torch.nn as nn
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.process_dataset import add_demographic_data, merge_file_path,  cleaning_datasets, sampling_datasets, get_group_by_data
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from models.build_model import DenseNet_Model, model_transfer_learning
from datasets.dataloader import prepare_dataloaders
from train.model_training import model_training
from evaluation.model_testing import model_testing

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 20
    training = True
    task = 'race'
    is_groupby = False
    test_output_path, train_output_path = None, None
    dataset = 'mimic'

    if not os.path.exists("/deep_learning/output/Sutariya/main/mimic/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/main/mimic/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/Sutariya/main/mimic/wandb")

    wandb.init(
    project=f"mimic-preprocessing-diagnostic_groupby_{task}" if is_groupby else f"mimic-preprocessing-diagnostic_{task}",
    dir="/deep_learning/output/Sutariya/main/mimic/wandb",
    config={
    "learning_rate": 0.0001,
    "Task": task,
    "save_model_file_name" : f'{task}_model_swa_{random_state}',
    "Uncertain Labels" : "-1 = 0, NAN = 0",
    "epochs": 20,
    "Augmentation": 'Yes',
    "optimiser": "AdamW",
    "SWA":'No',
    "architecture": "DenseNet121",
    "dataset": dataset,
    "Standardization": 'Yes'
    })

    if training:
        # Paths to the output files
        train_output_path = '/deep_learning/output/Sutariya/main/mimic/dataset/train_clean_dataset.csv'
        test_output_path = '/deep_learning/output/Sutariya/main/mimic/dataset/testing_clean_dataset.csv'

        # Run the code block only if the output files do not exist
        if not (os.path.exists(train_output_path) and os.path.exists(test_output_path)):
            
            # Input file paths
            training_file_path = 'MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz'
            demographic_data_path = 'MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/admissions.csv.gz'
            
            # Merge and clean the data
            training_data_merge = add_demographic_data(training_file_path, demographic_data_path)
            training_data_clean = cleaning_datasets(training_data_merge)
            sampling_training_dataset = sampling_datasets(training_data_clean)
            print(len(sampling_training_dataset))
            training_data_path_merge = merge_file_path('MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES.txt', sampling_training_dataset)
            print(len(training_data_path_merge))  

            split_and_save_datasets(training_data_path_merge, 
                                    train_path=train_output_path, 
                                    val_path=test_output_path)
        else:
            print(f'Files {train_output_path} && {test_output_path} already exists. Skipping save.')

        training_dataset = pd.read_csv(train_output_path)
        train_data, val_data = train_test_split(training_dataset, test_size=0.1, random_state=random_state)

        label_encoder = LabelEncoder()
        if task == 'diagnostic':
            labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
            train_loader = prepare_dataloaders(train_data['file_path'], train_data[labels].values, True, dataset=dataset)
            val_loader = prepare_dataloaders(val_data['file_path'], val_data[labels].values, True, dataset=dataset)

            criterion = nn.BCEWithLogitsLoss()
            model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=14)
        elif task == 'race':
            top_races = train_data['race'].value_counts().index[:5]
            train_data = train_data[train_data['race'].isin(top_races)].copy()
            val_data = val_data[val_data['race'].isin(top_races)].copy()
            train_data['race_encoded'] = label_encoder.fit_transform(train_data['race'])
            val_data['race_encoded'] = label_encoder.transform(val_data['race'])
            train_loader = prepare_dataloaders(train_data['file_path'], train_data['race_encoded'].values, task, shuffle=True, dataset=dataset)
            val_loader = prepare_dataloaders(val_data['file_path'], val_data['race_encoded'].values, task,shuffle=True, dataset=dataset)
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            model = model_transfer_learning('/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_daignostic_model_10.pth', base_model, device)
        elif task == 'ethnicity':
            train_data['ethnicity_encoded'] = label_encoder.fit_transform(train_data['ethnicity'])
            val_data['ethnicity_encoded'] = label_encoder.transform(val_data['ethnicity'])
            train_loader = prepare_dataloaders(train_data['Path'], train_data['ethnicity_encoded'].values, task,shuffle=True, dataset=dataset)
            val_loader = prepare_dataloaders(val_data['Path'], val_data['ethnicity_encoded'].values, task,shuffle=True, dataset=dataset)
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=4)
            model = model_transfer_learning('/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_daignostic_model_10.pth', base_model, device)

        model_training(model, train_loader, val_loader, criterion, task, 20, device=device, multi_label=False, is_swa=True)
        
        torch.save(model.state_dict(), f'/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_{task}_model_{random_state}.pth')
    
    else:
        testing_data = pd.read_csv(test_output_path)

        if task =='diagnostic':
            labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']
            test_loader = prepare_dataloaders(testing_data['file_path'], testing_data[labels].values, task,True, dataset=dataset)
            weights = torch.load('daignostic_model_80.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=14)
        elif task == 'race':
            top_races = testing_data['race'].value_counts().index[:5]
            testing_data = testing_data[testing_data['race'].isin(top_races)].copy()
            testing_data['race_encoded'] = label_encoder.fit_transform(testing_data['race'])
            test_loader = prepare_dataloaders(testing_data['file_path'], testing_data['race_encoded'].values, task, shuffle=True, dataset=dataset)
            weights = torch.load('race_model_50.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=3)
        elif task == 'ethnicity':
            testing_data['ethnicity_encoded'] = label_encoder.fit_transform(testing_data['ethnicity'])
            test_loader = prepare_dataloaders(testing_data['Path'], testing_data['ethnicity_encoded'].values, task, shuffle=True, dataset=dataset)
            weights = torch.load('ethnicity_model_50.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=3)

        test_model.load_state_dict(weights)
        testing_model(test_loader,test_model, task, device, multi_label=True)


