import os
import sys
import pandas as pd
import torch
import torchvision
import wandb
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.process_dataset import (
    add_demographic_data,
    add_lung_mask_mimic_dataset,
    add_metadata,
    cleaning_datasets,
    merge_file_path_and_add_dicom_id,
    sampling_datasets,
    merge_dataframe,
    add_lung_mask_chexpert_dataset
)

from datasets.dataloader import prepare_dataloaders
from datasets.split_store_dataset import split_train_test_data
from evaluation.model_testing import model_testing
from models.build_model import DenseNet_Model, model_transfer_learning
from train.model_training import model_training
from helper.losses import LabelSmoothingLoss
from sklearn.preprocessing import OneHotEncoder
import argparse

if __name__ == "__main__":
    # Define all the use taken inputs here
    parser = argparse.ArgumentParser(description="Training and Testing Arguments")
    parser.add_argument('--random_state', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--task', type=str, choices=["diagnostic", "race"], default="diagnostic")
    parser.add_argument('--dataset', type=str, default="mimic")
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--clahe', action='store_true')
    parser.add_argument('--crop_masked', action='store_true')
    parser.add_argument('--is_groupby', action='store_true')
    parser.add_argument('--eval_metrics', action='store_true')
    parser.add_argument('--external_ood_test', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = args.random_state
    epoch = args.epoch
    task = args.task
    dataset = args.dataset
    training = args.training
    masked = args.masked
    clahe = args.clahe
    crop_masked = args.crop_masked
    eval_metrics = args.eval_metrics
    is_groupby = args.is_groupby
    external_ood_test = args.external_ood_test

    # Select base folder of dataset
    base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    name = f"model_baseline_{dataset}_{random_state}"
    category_order = ['Asian', 'Black', 'White', 'Hispanic']

    if clahe:
        name = f"clahe_preprocessing_{name}"

    if masked:
        name = f"mask_preprocessing_{name}"

    if crop_masked:
        name = f"crop_masked_preprocessing_{name}"

    if dataset == 'mimic':
        train_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/train_clean_dataset.csv"
        test_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
        valid_output_path ="/deep_learning/output/Sutariya/main/mimic/dataset/validation_clean_dataset.csv"
        meta_file_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
        demographic_data_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/admissions.csv.gz"
        all_dataset_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"
    else:
        raise NotImplementedError

    if external_ood_test:
        external_ood_test  = True
        demographic_data_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/demographics_CXP.csv"
        train_dataset_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train.csv"
        valid_dataset_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/valid.csv"
        external_total_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/chexpert_total_dataset.csv"
    else:
        external_ood_test = False
    
    if task == 'diagnostic':
        multi_label = True
    elif task == 'race':
        multi_label = False
        diag_trained_model_path = f"diagnostic/{name}.pth"
    else:
        raise NotImplementedError

    if not os.path.exists("/deep_learning/output/Sutariya/main/mimic/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/main/mimic/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath(
        "/deep_learning/output/Sutariya/main/mimic/wandb"
    )

    wandb.init(
        project=f"mimic_preprocessing_{task}",
        name=f"{name}",
        dir="/deep_learning/output/Sutariya/main/mimic/wandb",
        config={
            "learning_rate": 0.0001,
            "Task": task,
            "save_model_file_name": name,
            "Uncertain Labels": "-1 = 0, NAN = 0",
            "epochs": epoch,
            "Augmentation": "Yes",
            "optimiser": "AdamW",
            "architecture": "DenseNet121",
            "dataset": dataset,
            "Standardization": "Yes",
        },
    )

    if not (os.path.exists(train_output_path) and os.path.exists(test_output_path)):
        # Merge and clean the data
        total_data_merge = add_demographic_data(all_dataset_path, demographic_data_path)
        total_data_merge = add_metadata(total_data_merge, meta_file_path)
        total_data_clean = cleaning_datasets(total_data_merge, False)
        sampling_total_dataset = sampling_datasets(total_data_clean)
        
        total_data_path_merge = merge_file_path_and_add_dicom_id(
                "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES.txt",
                sampling_total_dataset,
        )
        total_mask_path_merge = add_lung_mask_mimic_dataset(total_data_path_merge)
        if not (os.path.exists(train_output_path)) and not (
            os.path.exists(test_output_path)
            ):
            split_train_test_data(
            total_mask_path_merge, 35, train_output_path, test_output_path, "race"
            )
        else:
            print("Data is already sampled spit into train and test")
        
        train_df = pd.read_csv(train_output_path)

        split_and_save_datasets(
            train_df,
            train_path=train_output_path,
            val_path=valid_output_path,
            val_size=0.05,
            random_seed=random_state,
        )
        print("Splitting Completed...")
    else:
        print(
            f"Files {train_output_path} && {test_output_path} already exists. Skipping save..."
        )

    if task == "diagnostic":
        # Define disease labels
        labels = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion"]

        model_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/{name}.pth"
        if training:
            # If model weights don't exist, prepare data and train model
            if not os.path.exists(model_path):
                train_df = pd.read_csv(train_output_path)
                val_df = pd.read_csv(valid_output_path)

                # Filter by mask quality if masked training is specified
                if masked:
                    train_df = train_df[train_df['Dice RCA (Mean)'] > 0.7] 
                    val_df = val_df[val_df['Dice RCA (Mean)'] > 0.7]
        
                train_loader = prepare_dataloaders(
                    images_path= train_df["Path"],
                    labels= train_df[labels].values,
                    dataframe= train_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                    external_ood_test = False
                )
                val_loader = prepare_dataloaders(
                    images_path= val_df["Path"],
                    labels= val_df[labels].values,
                    dataframe= val_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                    external_ood_test = False
                )

                # Use label smoothing loss to improve generalization
                criterion = LabelSmoothingLoss(smoothing=0.1, mode='multilabel')

                # Initialize DenseNet121 model with ImageNet weights and 11 output features
                model = DenseNet_Model(
                    weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                    out_feature=11
                )

                # Train the model
                model_training(
                model= model,
                train_loader= train_loader,
                val_loader= val_loader,
                loss_function= criterion,
                tasks= task,
                actual_labels= labels,
                num_epochs= epoch,
                device=device,
                multi_label=multi_label
                )

                model_dir = os.path.dirname(model_path)
                # Make sure model checkpoint directory exists
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), model_path)

            else:
                print("File already exists skip training...")

        else:   
            # Load pretrained model weights if not training
            weights = torch.load(
                                model_path,
                                map_location=device,
                                weights_only=True)
            
            model = DenseNet_Model(weights=None, out_feature=11)
            model.load_state_dict(weights)

        testing_df = pd.read_csv(test_output_path)

        test_loader = prepare_dataloaders(
                    images_path= testing_df["Path"],
                    labels= testing_df[labels].values,
                    dataframe= testing_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test = False
                )
        
        # Evaluate model on test data
        model_testing(test_loader=test_loader,
                    model= model, 
                    dataframe= testing_df, 
                    original_labels= labels,
                    masked= masked, 
                    clahe= clahe, 
                    task= task, 
                    crop_masked=crop_masked,
                    name= name, 
                    base_dir=base_dir, 
                    device= device, 
                    multi_label=multi_label, 
                    is_groupby=is_groupby,
                    external_ood_test= False)

        # Run external out-of-distribution (Chexpert) testing if specified
        if external_ood_test:
            base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
            ex_total_df = pd.read_csv(external_total_path)


            test_loader = prepare_dataloaders(
                    images_path= ex_total_df["Path"],
                    labels= ex_total_df[labels].values,
                    dataframe= ex_total_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test = external_ood_test
                )
        
            # Evaluate on external OOD test data
            model_testing(test_loader=test_loader,
                        model= model, 
                        dataframe= ex_total_df, 
                        original_labels= labels, 
                        masked= masked, 
                        clahe= clahe, 
                        task= task, 
                        crop_masked= crop_masked,
                        name= name, 
                        base_dir=base_dir, 
                        device= device, 
                        multi_label=multi_label, 
                        is_groupby=is_groupby,
                        external_ood_test =external_ood_test)

    elif task == "race":
        model_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/{name}.pth"
        encoder = OneHotEncoder(categories=[category_order], handle_unknown='ignore', sparse_output=False)
        if training:
            # Train model if weights are not already saved
            if not os.path.exists(model_path):
                train_df = pd.read_csv(train_output_path)
                val_df = pd.read_csv(valid_output_path) 

                # Filter datasets by mask quality if masked training requested
                if masked:
                    train_df = train_df[train_df['Dice RCA (Mean)'] > 0.7]
                    val_df = val_df[val_df['Dice RCA (Mean)'] > 0.7]
                    
                # Used OneHotEncoding to treat all race label seperately and as multi-class problem    
                train_encoded_array = encoder.fit_transform(train_df[['race']])
                new_column_names = encoder.get_feature_names_out(['race'])
                train_df_encoded = pd.DataFrame(train_encoded_array, columns=new_column_names, index=train_df.index)
                train_df = pd.concat([train_df, train_df_encoded], axis=1)
                train_df.dropna(axis=0, inplace=True)
                val_encoded_array = encoder.transform(val_df[['race']])
                val_df_encoded = pd.DataFrame(val_encoded_array, columns=new_column_names, index=val_df.index)
                val_df = pd.concat([val_df, val_df_encoded], axis=1)
                val_df.dropna(axis=0, inplace=True)


                train_loader = prepare_dataloaders(
                    images_path= train_df["Path"],
                    labels= train_df["race_encoded"].values,
                    dataframe= train_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                    external_ood_test =False
                )
                val_loader = prepare_dataloaders(
                    images_path= val_df["Path"],
                    labels= val_df['race_encoded'].values,
                    dataframe= val_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform= None,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                    external_ood_test = False
                )

                # Use label smoothing loss for multiclass classification
                criterion = LabelSmoothingLoss(smoothing=0.1, mode='multiclass')

                # Initialize base DenseNet model for 4-class race classification
                base_model = DenseNet_Model(
                    weights=None,
                    out_feature=4
                )

                # Transfer learning from diagnostic model if available
                model = model_transfer_learning(
                    f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{diag_trained_model_path}",
                    base_model,
                    device,
                    False
                )

                model_training(
                    model= model,
                    train_loader= train_loader,
                    val_loader= val_loader,
                    loss_function= criterion,
                    tasks= task,
                    actual_labels= labels,
                    num_epochs= epoch,
                    device=device,
                    multi_label=multi_label
                    )
                
                model_dir = os.path.dirname(model_path)
                # Ensure model checkpoint directory exists and save weights
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), model_path)
            else:
                print("File already exists skip training...")

        else:
            # Load pretrained model weights for evaluation
            weights = torch.load(
                            model_path,
                            map_location=device,
                            weights_only=True,
                        )
            model = DenseNet_Model(weights=None, out_feature=4)
            model.load_state_dict(weights)

        test_df = pd.read_csv(test_output_path)
        test_df = test_df[test_df['race'].isin(['Asian', 'Black', 'White', 'Hispanic'])]
        test_encoded_array = encoder.fit_transform(test_df[['race']])
        new_column_names = encoder.get_feature_names_out(['race'])
        test_df_encoded = pd.DataFrame(test_encoded_array, columns=new_column_names)
        test_df = pd.concat([test_df, test_df_encoded], axis=1)
        test_df.dropna(axis=0, inplace=True)

        test_loader = prepare_dataloaders(
                    images_path= test_df["Path"],
                    labels= test_df[new_column_names].values,
                    dataframe= test_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test = False
                )
        
        # Perform model testing and evaluation
        model_testing(test_loader=test_loader,
                      model= model, 
                      dataframe= test_df, 
                      original_labels= new_column_names, 
                      masked= masked, 
                      clahe= clahe, 
                      task= task, 
                      crop_masked= crop_masked,
                      name= name, 
                      base_dir=base_dir, 
                      device= device, 
                      multi_label=multi_label, 
                      is_groupby=is_groupby,
                      external_ood_test = False)

        # External OOD evaluation if enabled
        if external_ood_test:
            base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
            ex_total_df = pd.read_csv(external_total_path)
            ex_total_df = ex_total_df[ex_total_df['race'].isin(['Asian', 'Black', 'White', 'Hispanic'])]
            ex_total_encoded_array = encoder.fit_transform(ex_total_df[['race']])
            new_column_names = encoder.get_feature_names_out(['race'])
            ex_total_df_encoded = pd.DataFrame(ex_total_encoded_array, columns=new_column_names)
            ex_total_df = pd.concat([ex_total_df, ex_total_df_encoded], axis=1)
            ex_total_df.dropna(axis=0, inplace=True)
    
            test_loader = prepare_dataloaders(
                    images_path= ex_total_df["Path"],
                    labels= ex_total_df[new_column_names].values,
                    dataframe= ex_total_df,
                    masked= masked,
                    clahe= clahe,
                    crop_masked= crop_masked,
                    transform = None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                    external_ood_test =external_ood_test
                )

            model_testing(test_loader=test_loader,
                        model= model, 
                        dataframe= ex_total_df, 
                        original_labels= new_column_names, 
                        masked= masked, 
                        clahe= clahe, 
                        task= task, 
                        crop_masked= crop_masked,
                        name= name, 
                        base_dir=base_dir, 
                        device= device, 
                        multi_label=multi_label, 
                        is_groupby=is_groupby,
                        external_ood_test= external_ood_test)