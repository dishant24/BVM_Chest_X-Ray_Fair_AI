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
from data_preprocessing.process_dataset import add_demographic_data, add_metadata, merge_file_path_and_add_dicom_id, add_lung_mask_dataset, cleaning_datasets, sampling_datasets, get_group_by_data
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from models.build_model import DenseNet_Model, model_transfer_learning
from datasets.dataloader import prepare_mimic_dataloaders
from train.model_training import model_training
from evaluation.model_testing import model_testing
from datasets.data import MyDataset
import concurrent.futures
from tqdm import tqdm

if __name__ == '__main__':
    
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 30
    training = True
    task = 'diagnostic'
    is_groupby = False
    dataset = 'mimic'
    masked = True
    multi_label = True
    base_dir= 'MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/'
    name = f'model_{task}_with_mask_{random_state}' if masked else f'model_{task}_{random_state}'
    label_encoder = LabelEncoder()
    # n_jobs = 16  # Number of parallel workers
    # import os
    # import pandas as pd
    # from PIL import Image
    # from joblib import Parallel, delayed
    # from tqdm import tqdm
    
    # Paths to the output files
    train_output_path = '/deep_learning/output/Sutariya/main/mimic/dataset/train_mask_clean_dataset.csv' if masked else '/deep_learning/output/Sutariya/main/mimic/dataset/train_clean_dataset.csv'
    valid_output_path = '/deep_learning/output/Sutariya/main/mimic/dataset/test_mask_clean_dataset.csv' if masked else '/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv'

    # Input file paths
    meta_file_path = 'MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz'
    train_file_path = '/deep_learning/output/Sutariya/main/mimic/dataset/train_dataset.csv'
    test_file_path = '/deep_learning/output/Sutariya/main/mimic/dataset/test_dataset.csv'
    demographic_data_path = 'MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/admissions.csv.gz'
    all_dataset_path = 'MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz'

    if not os.path.exists("/deep_learning/output/Sutariya/main/mimic/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/main/mimic/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath("/deep_learning/output/Sutariya/main/mimic/wandb")

    wandb.init(
    project=f"mimic_preprocessing_groupby_{task}" if is_groupby else f"mimic_preprocessing_{task}",
    name = f'{name}',
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

    

    # # Fast image check function
    # def is_bad_image(path):
    #     try:
    #         with Image.open(path) as img:
    #             img.verify()  # Only checks validity, does not load the full image
    #         return None
    #     except:
    #         return path

    # # Parallel check function
    # def check_image_paths_in_parallel(image_paths, n_jobs=16):
    #     results = Parallel(n_jobs=n_jobs)(
    #         delayed(is_bad_image)(path) for path in tqdm(image_paths)
    #     )
    #     return [r for r in results if r is not None]

    # training_dataset = pd.read_csv(train_output_path)
    # testing_dataset = pd.read_csv(test_file_path)
    # labels = ['Pneumonia']

    # # Get full paths
    # train_paths = [os.path.join(base_dir, p) for p in training_dataset['file_path']]
    # val_paths = [os.path.join(base_dir, p) for p in testing_dataset['file_path']]

    # # Run parallel check
    # print("Checking train dataset...")
    # bad_train_images = check_image_paths_in_parallel(train_paths, n_jobs=n_jobs)

    # print("Checking validation dataset...")
    # bad_val_images = check_image_paths_in_parallel(val_paths, n_jobs=n_jobs)

    # # Output
    # print("\n❌ Bad train images:")
    # for path in bad_train_images:
    #     print(path)

    # print("\n❌ Bad validation images:")
    # for path in bad_val_images:
    #     print(path)
    if not (os.path.exists(train_output_path) and os.path.exists(valid_output_path)):
        
        # Merge and clean the data
        total_data_merge = add_demographic_data(all_dataset_path, demographic_data_path)
        total_data_merge = add_metadata(total_data_merge, meta_file_path)
        total_data_clean = cleaning_datasets(total_data_merge, False)
        sampling_total_dataset = sampling_datasets(total_data_clean)
        if masked:
            total_data_path_merge = merge_file_path_and_add_dicom_id('MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES.txt', sampling_total_dataset)
            sampling_total_dataset = add_lung_mask_dataset(total_data_path_merge) 

        if not (os.path.exists(train_file_path)) and not (os.path.exists(test_file_path)):
            split_train_test_data(sampling_total_dataset, 20, train_file_path, test_file_path, 'race')
        else:
            print("Data is already sampled spit into train and test")
        train_data = pd.read_csv(train_file_path)
        split_and_save_datasets(train_data, 
                                train_path=train_output_path, 
                                val_path=valid_output_path,
                                val_size=0.05,
                                random_seed=random_state)
        print("Splitting Completed...")
    else:
        print(f'Files {train_output_path} && {valid_output_path} already exists. Skipping save...')


    
    if training:

        training_dataset = pd.read_csv(train_output_path)
        
        train_data, val_data = train_test_split(training_dataset, test_size=0.1, random_state=random_state)
        
        if task == 'diagnostic':
            labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion']
            train_loader = prepare_mimic_dataloaders(train_data['file_path'], train_data[labels].values, train_data, masked, base_dir,shuffle=True, is_multilabel=multi_label)
            val_loader = prepare_mimic_dataloaders(val_data['file_path'], val_data[labels].values, val_data, masked, base_dir,shuffle=True, is_multilabel=multi_label)
            criterion = nn.BCEWithLogitsLoss()
            model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=11)
        elif task == 'race':
            top_races = train_data['race'].value_counts().index[:5]
            train_data = train_data[train_data['race'].isin(top_races)].copy()
            val_data = val_data[val_data['race'].isin(top_races)].copy()
            train_data['race_encoded'] = label_encoder.fit_transform(train_data['race'])
            val_data['race_encoded'] = label_encoder.transform(val_data['race'])
            train_loader = prepare_mimic_dataloaders(train_data['file_path'], train_data['race_encoded'].values, train_data, base_dir,shuffle=True, is_multilabel=multi_label)
            val_loader = prepare_mimic_dataloaders(val_data['file_path'], val_data['race_encoded'].values, val_data, base_dir,shuffle=True, is_multilabel=multi_label)
            criterion = nn.CrossEntropyLoss()
            model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            # If you want to use transfer learning and want to diagnostic latent represetation preserve then uncoomment below lines of code
            # base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            # model = model_transfer_learning('/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_daignostic_model_10.pth', base_model, device)
        elif task == 'ethnicity':
            train_data['ethnicity_encoded'] = label_encoder.fit_transform(train_data['ethnicity'])
            val_data['ethnicity_encoded'] = label_encoder.transform(val_data['ethnicity'])
            train_loader = prepare_mimic_dataloaders(train_data['Path'], train_data['ethnicity_encoded'].values, train_data, base_dir,shuffle=True, is_multilabel=multi_label)
            val_loader = prepare_mimic_dataloaders(val_data['Path'], val_data['ethnicity_encoded'].values, val_data, base_dir,shuffle=True, is_multilabel=multi_label)
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=4)
            model = model_transfer_learning('/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_daignostic_model_10.pth', base_model, device)

        model_training(model, train_loader, val_loader, criterion, task, labels, 20, device=device, multi_label=multi_label, is_swa=True)
        
        torch.save(model.state_dict(), f'/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_{name}.pth')
    
    else:
        testing_data = pd.read_csv(valid_output_path)

        if task =='diagnostic':
            labels = ['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion']
            test_loader = prepare_mimic_dataloaders(testing_data['file_path'], testing_data[labels].values, testing_data, base_dir, shuffle=False, is_multilabel=multi_label)
            weights = torch.load('/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_mask_diagnostic_model_20.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=11)
        elif task == 'race':
            top_races = testing_data['race'].value_counts().index[:5]
            testing_data = testing_data[testing_data['race'].isin(top_races)].copy()
            testing_data['race_encoded'] = label_encoder.fit_transform(testing_data['race'])
            test_loader = prepare_mimic_dataloaders(testing_data['file_path'], testing_data['race_encoded'].values, testing_data, base_dir,shuffle=False, is_multilabel=multi_label)
            weights = torch.load('race_model_50.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=3)
        elif task == 'ethnicity':
            testing_data['ethnicity_encoded'] = label_encoder.fit_transform(testing_data['ethnicity'])
            test_loader = prepare_mimic_dataloaders(testing_data['Path'], testing_data['ethnicity_encoded'].values, testing_data, base_dir,shuffle=False, is_multilabel=multi_label)
            weights = torch.load('ethnicity_model_50.pth', map_location=device, weights_only=True)
            test_model = DenseNet_Model(weights=None, out_feature=3)

        test_model.load_state_dict(weights)
        model_testing(test_loader, test_model, labels,  task, device, multi_label=multi_label)


