import pandas as pd
import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
torch.cuda.empty_cache()
from sklearn.utils import shuffle
import wandb
from data_preprocessing.process_dataset import merge_dataframe, cleaning_datasets, sampling_datasets, get_group_by_data
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from datasets.split_store_dataset import load_and_store_diagnostic_images_labels, load_and_store_ethnic_images_labels, load_and_store_race_images_labels
from models.build_model import DenseNet_Model, model_transfer_learning
from datasets.dataloader import prepare_dataloaders
from train.model_training import model_training
from evaluation.model_testing import model_testing


torch.cuda.empty_cache()

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 10
    epoch = 20
    training = True
    task = 'ethnicity'
    is_groupby = False


    if not os.path.exists("/deep_learning/output/Sutariya/chexpert/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/chexpert/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/Sutariya/chexpert/wandb")

    wandb.init(
    project=f"cxr_preprocessing_{task}",
    dir="/deep_learning/output/Sutariya/chexpert/wandb",
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
    "dataset": "CheXpert",
    "Standardization": 'Yes'
    })

    training_file_path = '/deep_learning/output/Sutariya/chexpert/train_dataset.csv'
    demographic_data_path = '/deep_learning/output/Sutariya/chexpert/demographics_CXP.csv'
    train_output_path = '/deep_learning/output/Sutariya/chexpert/train_clean_dataset.csv'
    val_output_path = '/deep_learning/output/Sutariya/chexpert/validation_clean_dataset.csv'

    if training:
        if not (os.path.exists(train_output_path) and os.path.exists(val_output_path)):
            training_data = pd.read_csv(training_file_path)
            demographic_data = pd.read_csv(demographic_data_path)
            training_data_merge = merge_dataframe(training_data, demographic_data)
            training_data_clean = cleaning_datasets(training_data_merge)
            training_dataset = sampling_datasets(training_data_clean)
            if not (os.path.exists(training_file_path)):
                split_train_test_data(training_dataset, N=50)
            train_data = pd.read_csv(training_file_path)
            split_and_save_datasets(train_data, train_output_path, val_output_path)
        else:
            print(f'Files {train_output_path} && {val_output_path} already exists. Skipping save.')

        training_dataset = pd.read_csv(train_output_path)
        validation_dataset = pd.read_csv(val_output_path)

        if task == 'diagnostic':
            train_data_images,train_labels = load_and_store_diagnostic_images_labels(training_dataset, '/deep_learning/output/Sutariya/chexpert/data/diagnostic/train_images_tensor.pt', device)
            val_data_images, val_labels = load_and_store_diagnostic_images_labels(validation_dataset, '/deep_learning/output/Sutariya/chexpert/data/diagnostic/validation_images_tensor.pt', device)  
            criterion = nn.BCEWithLogitsLoss()
            model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=14)
        elif task == 'race':
            train_data_images,train_labels = load_and_store_race_images_labels(training_dataset, '/deep_learning/output/Sutariya/chexpert/data/race/train_images_tensor.pt', device)
            val_data_images, val_labels = load_and_store_race_images_labels(validation_dataset, '/deep_learning/output/Sutariya/chexpert/data/race/validation_images_tensor.pt', device)
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=3)
            model = model_transfer_learning('/deep_learning/output/Sutariya/chexpert/model_checkpoints/daignostic_model_swa_140.pth', base_model, device)
        elif task == 'ethnicity':
            train_data_images,train_labels = load_and_store_ethnic_images_labels(training_dataset, '/deep_learning/output/Sutariya/chexpert/data/ethnicity/train_images_tensor.pt', device)
            val_data_images, val_labels = load_and_store_ethnic_images_labels(validation_dataset, '/deep_learning/output/Sutariya/chexpert/data/ethnicity/validation_images_tensor.pt', device)
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=4)
            model = model_transfer_learning('/deep_learning/output/Sutariya/chexpert/model_checkpoints/daignostic_model_swa_140.pth', base_model, device)   
        else:
            print("Task value should be in diagnostic or race or ethnicity...")
        train_data_shuffel_images, train_shuffel_labels = shuffle(train_data_images, train_labels, random_state=random_state)
        val_data_shuffel_images, val_shuffel_labels = shuffle(val_data_images, val_labels, random_state=random_state)
        train_loader = prepare_dataloaders(train_data_shuffel_images, train_shuffel_labels, shuffle=True)
        val_loader = prepare_dataloaders(val_data_shuffel_images, val_shuffel_labels, shuffle=True)

        model_training(model, train_loader, val_loader, criterion, epoch, device=device, multi_label=False)
        
        torch.save(model.state_dict(), f'/deep_learning/output/Sutariya/chexpert/checkpoint/{task}/model_swa_{random_state}.pth') 
    
    else:
        if is_groupby:
            if not (os.path.exists(train_output_path) and os.path.exists(val_output_path)):
                training_data = pd.read_csv(training_file_path)
                demographic_data = pd.read_csv(demographic_data_path)
                training_data_merge = merge_dataframe(training_data, demographic_data)
                training_data_clean = cleaning_datasets(training_data_merge)
                training_dataset = sampling_datasets(training_data_clean)

                split_and_save_datasets(training_dataset, train_output_path, val_output_path)
            else:
                print(f'Files {train_output_path} && {val_output_path} already exists. Skipping save.')

            training_dataset = pd.read_csv(train_output_path)
            validation_dataset = pd.read_csv(val_output_path)
            dataset = training_dataset.concat(training_dataset, validation_dataset, copy=True)
            ethnic_groupby_dataset = get_group_by_data(dataset, 'ethnicity')
            
            for group in ethnic_groupby_dataset.keys():
                save_dir  = f'/deep_learning/output/Sutariya/chexpert/data/ethnicity/groupby/{group}'
                os.makedirs(save_dir, exist_ok=True)
                test_data_images,test_labels = load_and_store_diagnostic_images_labels(ethnic_groupby_dataset[group], str(save_dir+'/'+'train_images_tensor.pt'), device)
                test_data_shuffel_images, test_shuffel_labels = shuffle(test_data_images, test_labels, random_state=random_state)
                test_loader = prepare_dataloaders(test_data_shuffel_images, test_shuffel_labels, shuffle=True)
                weights = torch.load('/deep_learning/output/Sutariya/chexpert/model_checkpoints/daignostic_model_swa_140.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=4)
                test_model.load_state_dict(weights)
                model_testing(test_loader, test_model, device, multi_label=True, group_name=group)
        else:
            test_file_path =  '/deep_learning/output/Sutariya/chexpert/valid.csv'
            demographic_data_path = '/deep_learning/output/Sutariya/chexpert/demographics_CXP.csv'
            testing_data = pd.read_csv(test_file_path)
            demographic_data = pd.read_csv(demographic_data_path)

            testing_dataset = merge_dataframe(testing_data, demographic_data)
            testing_dataset = cleaning_datasets(testing_dataset)
            if task == 'diagnostic':
                test_data_images, test_labels = load_and_store_diagnostic_images_labels(testing_dataset, '/deep_learning/output/Sutariya/chexpert/data/diagnostic/test_images_tensor.pt', device)
                weights = torch.load('daignostic_model_swa_1.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=14)
            elif task == 'race':
                test_data_images, test_labels = load_and_store_race_images_labels(testing_dataset, '/deep_learning/output/Sutariya/chexpert/data/race/test_images_tensor.pt', device)
                weights = torch.load('race_model_swa_1.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=3)
            elif task == 'ethnicity':
                test_data_images, test_labels = load_and_store_ethnic_images_labels(testing_dataset, '/deep_learning/output/Sutariya/chexpert/data/ethnicity/test_images_tensor.pt', device)
                weights = torch.load('race_model_swa_1.pth', map_location=device, weights_only=True)
                test_model = DenseNet_Model(weights=None, out_feature=4)
            else:
                print("Task value should be in diagnostic or race or ethnicity...")

            test_loader = prepare_dataloaders(test_data_images,test_labels, None, shuffle=False)
            test_model.load_state_dict(weights)

            model_testing(test_loader,test_model, device, multi_label=True)

