import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import wandb
from sklearn.preprocessing import LabelEncoder
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from datasets.data import MyDataset
from data_preprocessing.process_dataset import (
    add_demographic_data,
    add_lung_mask_dataset,
    add_metadata,
    cleaning_datasets,
    get_group_by_data,
    merge_file_path_and_add_dicom_id,
    sampling_datasets,
)
from datasets.dataloader import prepare_dataloaders
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from evaluation.model_testing import model_calibration, model_testing, model_testing_metrics_eval
from models.build_model import DenseNet_Model, model_transfer_learning
from train.model_training import model_training
from helper.losses import LabelSmoothingLoss
from PIL import Image
import os
import cv2
from tqdm import tqdm
import multiprocessing as mp
from helper.generate_plot import generate_plot
from helper.generate_result_tables import generate_tabel
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and Testing Arguments")

    parser.add_argument('--random_state', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--task', type=str, choices=["diagnostic", "race"], default="diagnostic")
    parser.add_argument('--dataset', type=str, default="mimic")
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--masked', action='store_true')
    parser.add_argument('--clahe', action='store_true')
    parser.add_argument('--reweight', action='store_true')
    parser.add_argument('--is_groupby', action='store_true')
    parser.add_argument('--external_ood_test', action='store_true')
    
    args = parser.parse_args()
    print(vars(args))
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = args.random_state
    epoch = args.epoch
    task = args.task
    dataset = args.dataset
    training = args.training
    masked = args.masked
    clahe = args.clahe
    reweight = args.reweight
    is_groupby = args.is_groupby
    external_ood_test = args.external_ood_test

    base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    name = f"model_baseline_{dataset}_{random_state}"

    if clahe:
        name = f"clahe_preprocessing_{name}"

    if masked:
        name = f"mask_preprocessing_{name}"

    if reweight:
        name = f"reweight_preprocessing_{name}"

    if dataset == 'mimic':
        train_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/train_clean_dataset.csv"
        test_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
        valid_output_path ="/deep_learning/output/Sutariya/main/mimic/dataset/validation_clean_dataset.csv"
        meta_file_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
        demographic_data_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/admissions.csv.gz"
        all_dataset_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"
    else:
        raise NotImplementedError
    
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
        if not (os.path.exists(train_output_path)) and not (
            os.path.exists(test_output_path)
            ):
            split_train_test_data(
            total_data_path_merge, 35, train_output_path, test_output_path, "race"
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

    # test_data = pd.read_csv(test_output_path)
    # val_df = pd.read_csv(valid_output_path)
    # top_races = val_df["race"].value_counts().index[:5]
    # val_df = val_df[val_df["race"].isin(top_races)].copy()
    # race_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_race_101.pth"
    # race_lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessing_mimic_race_101.pth"
    # race_clahe_weights= "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_race_101.pth"
    # weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101.pth"
    # lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessingmimic_diagnostic_101.pth"
    # clahe_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_diagnostic_101.pth"
    
    # generate_plot(weights, lung_weights, clahe_weights, device, val_df, multi_label, base_dir)
    # generate_tabel(race_weights, race_lung_weights, race_clahe_weights, weights, lung_weights, clahe_weights, device, test_data, base_dir)
    # labels = [
    #     "No Finding",
    #     "Enlarged Cardiomediastinum",
    #     "Cardiomegaly",
    #     "Lung Opacity",
    #     "Lung Lesion",
    #     "Edema",
    #     "Consolidation",
    #     "Pneumonia",
    #     "Atelectasis",
    #     "Pneumothorax",
    #     "Pleural Effusion"]

    # test_loader = prepare_dataloaders(
    #             test_data["Path"],
    #             test_data[labels].values,
    #             test_data,
    #             masked,
    #             clahe,
    #             base_dir,
    #             shuffle=False,
    #             is_multilabel=multi_label,
    #         )
    # val_loader = prepare_dataloaders(
    #             val_df["Path"],
    #             val_df[labels].values,
    #             val_df,
    #             masked,
    #             clahe,
    #             base_dir,
    #             shuffle=False,
    #             is_multilabel=multi_label,
    #         )
     
    # model_calibration(clahe_weights, device, test_loader, val_loader, labels)
    if task == "diagnostic":
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
            if not os.path.exists(model_path):
                train_df = pd.read_csv(train_output_path)
                val_df = pd.read_csv(valid_output_path)
                if reweight:
                    top_races = train_df["race"].value_counts().index[:5]
                    train_df = train_df[train_df["race"].isin(top_races)].copy()
                    val_df = val_df[val_df["race"].isin(top_races)].copy()
                train_loader = prepare_dataloaders(
                    images_path= train_df["Path"],
                    labels= train_df[labels].values,
                    dataframe= train_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                )
                val_loader = prepare_dataloaders(
                    images_path= val_df["Path"],
                    labels= val_df[labels].values,
                    dataframe= val_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                )
                criterion = LabelSmoothingLoss(smoothing=0.1, mode='multilabel')
                # criterion = nn.BCEWithLogitsLoss()
                model = DenseNet_Model(
                    weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                    out_feature=11
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

                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), model_path)

            else:
                print("File already exists skip training...")

        else:   
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
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
        model_testing(test_loader=test_loader,
                      model= model, 
                      dataset= testing_df, 
                      original_labels= labels, 
                      masked= masked, 
                      clahe= clahe, 
                      task= task, 
                      name= name, 
                      base_dir=base_dir, 
                      device= device, 
                      multi_label=multi_label, 
                      is_groupby=is_groupby)
        
        if external_ood_test:
            testing_df = pd.read_csv(external_test_path)
            test_loader = prepare_dataloaders(
                    images_path= testing_df["Path"],
                    labels= testing_df[labels].values,
                    dataframe= testing_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
            model_testing(test_loader=test_loader,
                        model= model, 
                        dataset= testing_df, 
                        original_labels= labels, 
                        masked= masked, 
                        clahe= clahe, 
                        task= task, 
                        name= name, 
                        base_dir=base_dir, 
                        device= device, 
                        multi_label=multi_label, 
                        is_groupby=is_groupby)

    elif task == "race":
        label_encoder = LabelEncoder()
        model_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/{name}.pth"
        if training:
            if not os.path.exists(model_path):
                train_df = pd.read_csv(train_output_path)
                val_df = pd.read_csv(valid_output_path)
                top_races = train_df["race"].value_counts().index[:4]
                train_df = train_df[train_df["race"].isin(top_races)].copy()
                val_df = val_df[val_df["race"].isin(top_races)].copy()
                train_df["race_encoded"] = label_encoder.fit_transform(train_df["race"])
                val_df["race_encoded"] = label_encoder.transform(val_df["race"])
                labels = label_encoder.classes_
                train_loader = prepare_dataloaders(
                    images_path= train_df["Path"],
                    labels= train_df["race_encoded"].values,
                    dataframe= train_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                )
                val_loader = prepare_dataloaders(
                    images_path= val_df["Path"],
                    labels= val_df['race_encoded'].values,
                    dataframe= val_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                )
                criterion = LabelSmoothingLoss(smoothing=0.1, mode='multiclass')
                base_model = DenseNet_Model(
                    weights=None,
                    out_feature=4
                )
                model = model_transfer_learning(
                    f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{diag_trained_model_path}",
                    base_model,
                    device
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

                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                torch.save(model.state_dict(), model_path)
            else:
                print("File already exists skip training...")

        else:
            weights = torch.load(
                            model_path,
                            map_location=device,
                            weights_only=True,
                        )
            model = DenseNet_Model(weights=None, out_feature=4)
            model.load_state_dict(weights)

        testing_df = pd.read_csv(test_output_path)
        testing_df["race_encoded"] = label_encoder.fit_transform(testing_df["race"])
        labels = label_encoder.classes_

        test_loader = prepare_dataloaders(
                    images_path= testing_df["Path"],
                    labels= testing_df["race_encoded"].values,
                    dataframe= testing_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
        model_testing(test_loader=test_loader,
                      model= model, 
                      dataset= testing_df, 
                      original_labels= labels, 
                      masked= masked, 
                      clahe= clahe, 
                      task= task, 
                      name= name, 
                      base_dir=base_dir, 
                      device= device, 
                      multi_label=multi_label, 
                      is_groupby=is_groupby)

        if external_ood_test:
            testing_df = pd.read_csv(external_test_path)
            testing_df["race_encoded"] = label_encoder.fit_transform(testing_df["race"])
            labels = top_races.values
    
            test_loader = prepare_dataloaders(
                    images_path= testing_df["Path"],
                    labels= testing_df["race_encoded"].values,
                    dataframe= testing_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
        model_testing(test_loader=test_loader,
                      model= model, 
                      dataset= testing_df, 
                      original_labels= labels, 
                      masked= masked, 
                      clahe= clahe, 
                      task= task, 
                      name= name, 
                      base_dir=base_dir, 
                      device= device, 
                      multi_label=multi_label, 
                      is_groupby=is_groupby)

    # weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101.pth"
    # lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessingmimic_diagnostic_101.pth"
    # clahe_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_diagnostic_101.pth"
    # generate_tabel(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_df, base_dir)
    # generate_strip_plot(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_df, multi_label)
                
            