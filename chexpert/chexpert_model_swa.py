import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wandb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from data_preprocessing.process_dataset import (
    add_demographic_data,
    cleaning_datasets,
    add_lung_mask_chexpert_dataset,
    get_group_by_data,
    merge_dataframe,
    sampling_datasets,
)
from datasets.dataloader import prepare_dataloaders
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from evaluation.model_testing import model_testing
from models.build_model import DenseNet_Model, model_transfer_learning
from train.model_training import model_training
from helper.losses import LabelSmoothingLoss
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training and Testing Arguments")

    parser.add_argument('--random_state', type=int, default=100)
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--task', type=str, choices=["diagnostic", "race"], default="diagnostic")
    parser.add_argument('--dataset', type=str, default="chexpert")
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

    base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
    name = f"model_baseline_{dataset}_{random_state}"

    if clahe:
        name = f"clahe_preprocessing_{name}"

    if masked:
        name = f"mask_preprocessing_{name}"

    if reweight:
        name = f"reweight_preprocessing_{name}"

    if dataset == 'chexpert':
        demographic_data_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/demographics_CXP.csv"
        train_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train_clean_dataset.csv"
        valid_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/validation_clean_dataset.csv"
        test_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/test_clean_dataset.csv"
        all_dataset_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train.csv"
    else:
        raise NotImplementedError

    if task == 'diagnostic':
        multi_label = True
    elif task == 'race':
        multi_label = False
        diag_trained_model_path = f"diagnostic/{name}.pth"
    else:
        raise NotImplementedError

    if not os.path.exists("/deep_learning/output/Sutariya/main/chexpert/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/main/chexpert/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath(
        "/deep_learning/output/Sutariya/main/chexpert/wandb"
    )

    wandb.init(
        project=f"chexpert_preprocessing_{task}",
        name=f"{name}",
        dir="/deep_learning/output/Sutariya/main/chexpert/wandb",
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

    if not (os.path.exists(train_output_path) and os.path.exists(valid_output_path)):
        all_data = pd.read_csv(all_dataset_path)
        demographic_data = pd.read_csv(demographic_data_path)
        all_data_merge = merge_dataframe(all_data, demographic_data)
        all_data_clean = cleaning_datasets(all_data_merge)
        all_dataset = sampling_datasets(all_data_clean)
        total_mask_path_merge = add_lung_mask_chexpert_dataset(all_dataset)
        if not (os.path.exists(train_output_path)) and not (
            os.path.exists(test_output_path)
        ):
            split_train_test_data(
                total_mask_path_merge, 35, train_output_path, test_output_path, "race"
            )
        else:
            print("Data is already sampled spit into train and test")
        train_data = pd.read_csv(train_output_path)
        split_and_save_datasets(
            train_data,
            train_output_path,
            valid_output_path,
            val_size=0.05,
            random_seed=random_state,
        )
    else:
        print(
            f"Files {train_output_path} && {valid_output_path} already exists. Skipping save."
        )

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

        model_path = f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{task}/{name}.pth"
        if training:
            if not os.path.exists(model_path):
                train_df = pd.read_csv(train_output_path)
                val_df = pd.read_csv(valid_output_path)
                if masked:
                    train_df = train_df[train_df['Dice RCA (Mean)'] > 0.7] 
                    val_df = val_df[val_df['Dice RCA (Mean)'] > 0.7]
                if reweight:
                    top_races = train_df["race"].value_counts().index[:4]
                    train_df = train_df[train_df["race"].isin(top_races)].copy()
                    val_df = val_df[val_df["race"].isin(top_races)].copy()
                train_loader = prepare_dataloaders(
                    images_path= train_df["Path"],
                    labels= train_df[labels].values,
                    dataframe= train_df,
                    masked= masked,
                    clahe= clahe,
                    reweight= reweight,
                    transform= None,
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
                    transform= None,
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
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
        model_testing(test_loader=test_loader,
                      model= model, 
                      dataframe= testing_df, 
                      original_labels= labels, 
                      masked= masked, 
                      clahe= clahe, 
                      task= task, 
                      reweight= reweight,
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
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
            model_testing(test_loader=test_loader,
                        model= model, 
                        dataframe= testing_df, 
                        original_labels= labels, 
                        masked= masked, 
                        clahe= clahe, 
                        task= task, 
                        reweight= reweight,
                        name= name, 
                        base_dir=base_dir, 
                        device= device, 
                        multi_label=multi_label, 
                        is_groupby=is_groupby)

    elif task == "race":
        label_encoder = LabelEncoder()
        model_path = f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{task}/{name}.pth"
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
                    transform=  None,
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
                    transform= None,
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
                    f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{diag_trained_model_path}",
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
                    transform= None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
        model_testing(test_loader=test_loader,
                      model= model, 
                      dataframe= testing_df, 
                      original_labels= labels, 
                      masked= masked, 
                      clahe= clahe, 
                      task= task, 
                      reweight= reweight,
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
                    transform = None,
                    base_dir= base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
        
        model_testing(test_loader=test_loader,
                      model= model, 
                      dataframe= testing_df, 
                      original_labels= labels, 
                      masked= masked, 
                      clahe= clahe, 
                      task= task, 
                      reweight= reweight,
                      name= name, 
                      base_dir=base_dir, 
                      device= device, 
                      multi_label=multi_label, 
                      is_groupby=is_groupby)
    # weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101.pth"
    # lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessingmimic_diagnostic_101.pth"
    # clahe_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_diagnostic_101.pth"
    # generate_tabel(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_data, base_dir)
    # generate_strip_plot(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_data, multi_label)
                
            