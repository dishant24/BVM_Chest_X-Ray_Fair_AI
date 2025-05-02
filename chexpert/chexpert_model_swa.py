import pandas as pd
import os
import sys
import numpy as np
import torch
import torchvision
import torch.nn as nn
torch.cuda.empty_cache()
from sklearn.utils import shuffle
import wandb
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.process_dataset import merge_dataframe, cleaning_datasets, sampling_datasets, get_group_by_data
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from models.build_model import DenseNet_Model, model_transfer_learning
from datasets.dataloader import prepare_chexpert_dataloaders
from train.model_training import model_training
from evaluation.model_testing import model_testing


torch.cuda.empty_cache()

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 10
    epoch = 20
    training = True
    task = 'diagnostic'
    dataset = 'chexpert'
    is_groupby = False
    multi_label = True


    if not os.path.exists("/deep_learning/output/Sutariya/main/chexpert/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/main/chexpert/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/Sutariya/main/chexpert/wandb")

    wandb.init(
    project=f"cxr_preprocessing_{task}" if not is_groupby else f"cxr_preprocessing_groupby_{task}",
    dir="/deep_learning/output/Sutariya/main/chexpert/wandb",
    name = f"no_finding_model_training_{random_state}",
    config={
    "learning_rate": 0.0001,
    "Task": task,
    "save_model_file_name" : f'{task}_model_swa_{random_state}',
    "Uncertain Labels" : "-1 = 0, NAN = 0",
    "epochs": 20,
    "Augmentation": 'Yes',
    "optimiser": "AdamW",
    "SWA":'Yes',
    "architecture": "DenseNet121",
    "dataset": dataset,
    "Standardization": 'Yes'
    })

    training_file_path = '/deep_learning/output/Sutariya/main/chexpert/dataset/train_dataset.csv'
    demographic_data_path = '/deep_learning/output/Sutariya/main/chexpert/dataset/demographics_CXP.csv'
    train_output_path = '/deep_learning/output/Sutariya/main/chexpert/dataset/train_clean_dataset.csv'
    val_output_path = '/deep_learning/output/Sutariya/main/chexpert/dataset/validation_clean_dataset.csv'

    if training:
        if not (os.path.exists(train_output_path) and os.path.exists(val_output_path)):
            training_data = pd.read_csv(training_file_path)
            demographic_data = pd.read_csv(demographic_data_path)
            training_data_merge = merge_dataframe(training_data, demographic_data)
            training_data_clean = cleaning_datasets(training_data_merge)
            training_dataset = sampling_datasets(training_data_clean)
            if not (os.path.exists(training_file_path)):
                split_train_test_data(training_dataset, N=60)
            train_data = pd.read_csv(training_file_path)
            split_and_save_datasets(train_data, train_output_path, val_output_path)
        else:
            print(f'Files {train_output_path} && {val_output_path} already exists. Skipping save.')

        training_dataset = pd.read_csv(train_output_path)
        validation_dataset = pd.read_csv(val_output_path)

        label_encoder = LabelEncoder()
        if task == 'diagnostic':
            labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
     'Support Devices']
            criterion = nn.BCEWithLogitsLoss()
            # if you want to train single diagnostic label uncomment below  2 line and comment above 4 line
            # labels = ['No Finding']
            # criterion = nn.BCEWithLogitsLoss()
            train_loader = prepare_chexpert_dataloaders(training_dataset['Path'], training_dataset[labels].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset',shuffle=True, is_multilabel=multi_label)
            val_loader = prepare_chexpert_dataloaders(validation_dataset['Path'], validation_dataset[labels].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=True, is_multilabel=multi_label)  
            
            model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V2, out_feature=14)
        elif task == 'race':
            top_races = training_dataset['race'].value_counts().index[:5]
            training_dataset = training_dataset[training_dataset['race'].isin(top_races)].copy()
            validation_dataset = validation_dataset[validation_dataset['race'].isin(top_races)].copy()
            training_dataset['race_encoded'] = label_encoder.fit_transform(training_dataset['race'])
            validation_dataset['race_encoded'] = label_encoder.transform(validation_dataset['race'])
            train_loader = prepare_chexpert_dataloaders(training_dataset['Path'], training_dataset['race_encoded'].values, None,base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=True, is_multilabel=multi_label)
            val_loader = prepare_chexpert_dataloaders(validation_dataset['Path'], validation_dataset['race_encoded'].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=True, is_multilabel=multi_label)
            criterion = nn.CrossEntropyLoss()
            model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            # Comment below if you want to train race model
            # base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=3)
            # model = model_transfer_learning('/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/model_swa_10.pth', base_model, device)
        elif task == 'ethnicity':
            training_dataset['ethnicity_encoded'] = label_encoder.fit_transform(training_dataset['ethnicity'])
            validation_dataset['ethnicity_encoded'] = label_encoder.transform(validation_dataset['ethnicity'])
            train_loader = prepare_chexpert_dataloaders(training_dataset['Path'], training_dataset['ethnicity_encoded'].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=True, is_multilabel=multi_label)
            val_loader = prepare_chexpert_dataloaders(validation_dataset['Path'], validation_dataset['ethnicity_encoded'].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset',  shuffle=True, is_multilabel=multi_label)
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=4)
            model = model_transfer_learning('/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/model_swa_10.pth', base_model, device)   
        else:
            print("Task value should be in diagnostic or race or ethnicity...")

        model_training(model, train_loader, val_loader, criterion, task, labels, epoch, device=device, multi_label=multi_label, is_swa=True)
        
        torch.save(model.state_dict(), f'/deep_learning/output/Sutariya/main/chexpert/checkpoint/{task}/model_swa_{random_state}.pth')
    
    else:
        if is_groupby:
            training_data = pd.read_csv('/deep_learning/output/Sutariya/main/chexpert/dataset/train.csv')
            demographic_data = pd.read_csv(demographic_data_path)
            all_dataset = merge_dataframe(training_data, demographic_data)
            test_dataset = pd.read_csv("/deep_learning/output/Sutariya/main/chexpert/dataset/test_dataset.csv")
            ethnic_groupby_dataset = get_group_by_data(test_dataset, 'ethnicity')
            labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
     'Support Devices']

            subject_to_ethnicity = dict(zip(all_dataset['subject_id'], all_dataset['ethnicity']))

            assert not test_dataset.duplicated('subject_id').any(), "Duplicate subject_ids found in test_dataset"
            assert not test_dataset.duplicated('Path').any(), "Duplicate image paths found in test_dataset"

            for idx, row in test_dataset.iterrows():
                sid = row['subject_id']
                test_ethnicity = row['ethnicity']
                
                assert sid in subject_to_ethnicity, f"subject_id {sid} not found in all_dataset"
                assert subject_to_ethnicity[sid] == test_ethnicity, \
                    f"Ethnicity mismatch for subject_id {sid}: test_dataset has '{test_ethnicity}', all_dataset has '{subject_to_ethnicity[sid]}'"

            for group in ethnic_groupby_dataset.keys():
                assert not ethnic_groupby_dataset[group].duplicated('subject_id').any(), f"Duplicate subject_ids in group {group}"
                assert not ethnic_groupby_dataset[group].duplicated('Path').any(), f"Duplicate image paths in group {group}"
                test_loader = prepare_chexpert_dataloaders(ethnic_groupby_dataset[group]['Path'], ethnic_groupby_dataset[group][labels].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=True, is_multilabel=multi_label)
                weights = torch.load('/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/model_swa_10.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=14)
                test_model.load_state_dict(weights)
                model_testing(test_loader, test_model, labels, task, device, multi_label=multi_label, group_name=group)
        else:
            test_file_path =  '/deep_learning/output/Sutariya/main/chexpert/dataset/valid.csv'
            demographic_data_path = '/deep_learning/output/Sutariya/main/chexpert/dataset/demographics_CXP.csv'
            testing_data = pd.read_csv(test_file_path)
            demographic_data = pd.read_csv(demographic_data_path)

            testing_dataset = merge_dataframe(testing_data, demographic_data)
            testing_dataset = cleaning_datasets(testing_dataset)
            if task == 'diagnostic':
                labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
     'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
     'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
     'Support Devices']
                test_loader = prepare_chexpert_dataloaders(testing_dataset['Path'], testing_dataset[labels].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=False, is_multilabel=multi_label)
                weights = torch.load('daignostic_model_swa_1.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=14)
            elif task == 'race':
                top_races = testing_dataset['race'].value_counts().index[:5]
                testing_dataset = testing_dataset[testing_dataset['race'].isin(top_races)].copy()
                testing_dataset['race_encoded'] = label_encoder.fit_transform(testing_dataset['race'])
                test_loader = prepare_chexpert_dataloaders(testing_dataset['Path'], testing_dataset['race_encoded'].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=False, is_multilabel=multi_label)
                weights = torch.load('race_model_swa_1.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=3)
            elif task == 'ethnicity':
                testing_dataset['ethnicity_encoded'] = label_encoder.fit_transform(testing_dataset['ethnicity'])
                test_loader = prepare_chexpert_dataloaders(testing_dataset['Path'], testing_dataset['ethnicity_encoded'].values, None, base_dir='/deep_learning/output/Sutariya/main/chexpert/dataset', shuffle=False,is_multilabel=multi_label)
                weights = torch.load('race_model_swa_1.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=4)
            else:
                print("Task value should be in diagnostic or race or ethnicity...")

            test_model.load_state_dict(weights)

            model_testing(test_loader,test_model, labels, device, multi_label=multi_label)

