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

if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 100
    epoch = 30
    task = "diagnostic"
    dataset = "mimic"
    training = True
    masked = False
    clahe= False
    reweight = True
    is_groupby = True
    multi_label = True
    external_ood_test = True
    train_path = None
    #Change the path acrroding to model usage
    trained_model_path = 'traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101' if task == 'race' else None

    base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    name = (
        f"traininig_with_data_balance_{dataset}_{task}_{random_state}"
        if training
        else f"validation_plot_results_{dataset}_{task}_{random_state}"
    )

    external_test_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/test_clean_dataset.csv"
    train_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/train_clean_dataset.csv"
    test_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
    valid_output_path ="/deep_learning/output/Sutariya/main/mimic/dataset/validation_clean_dataset.csv"
    meta_file_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
    demographic_data_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/admissions.csv.gz"
    all_dataset_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"
    
    label_encoder = LabelEncoder()

    if not os.path.exists("/deep_learning/output/Sutariya/main/mimic/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/main/mimic/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath(
        "/deep_learning/output/Sutariya/main/mimic/wandb"
    )

    wandb.init(
        project=f"mimic_preprocessing_groupby_{task}"
        if is_groupby
        else f"mimic_preprocessing_{task}",
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
            "PreprocessingMethod": "CHALE"
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
        
        train_data = pd.read_csv(train_output_path)

        split_and_save_datasets(
            train_data,
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
    # val_data = pd.read_csv(valid_output_path)
    # top_races = val_data["race"].value_counts().index[:5]
    # val_data = val_data[val_data["race"].isin(top_races)].copy()
    # race_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_race_101.pth"
    # race_lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessing_mimic_race_101.pth"
    # race_clahe_weights= "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_race_101.pth"
    # weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101.pth"
    # lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessingmimic_diagnostic_101.pth"
    # clahe_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_diagnostic_101.pth"
    
    # generate_plot(weights, lung_weights, clahe_weights, device, val_data, multi_label, base_dir)
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
    #             val_data["Path"],
    #             val_data[labels].values,
    #             val_data,
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

        if training:
            if not os.path.exists(f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{name}.pth"):
                train_data = pd.read_csv(train_output_path)
                val_data = pd.read_csv(valid_output_path)
                if reweight:
                    top_races = train_data["race"].value_counts().index[:5]
                    train_data = train_data[train_data["race"].isin(top_races)].copy()
                    val_data = val_data[val_data["race"].isin(top_races)].copy()
                train_loader = prepare_dataloaders(
                    train_data["Path"],
                    train_data[labels].values,
                    train_data,
                    masked,
                    clahe,
                    reweight,
                    base_dir,
                    shuffle=True,
                    is_multilabel=multi_label,
                )
                val_loader = prepare_dataloaders(
                    val_data["Path"],
                    val_data[labels].values,
                    val_data,
                    masked,
                    clahe,
                    reweight,
                    base_dir,
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
                model,
                train_loader,
                val_loader,
                criterion,
                task,
                labels,
                epoch,
                device=device,
                multi_label=multi_label,
                is_swa=True,
                )
                torch.save(
                    model.state_dict(),
                    f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{name}.pth",
                )
            else:
                print("File already exists skip training...")
                testing_data = pd.read_csv(test_output_path)
                test_loader = prepare_dataloaders(
                            testing_data["Path"],
                            testing_data[labels].values,
                            testing_data,
                            masked,
                            clahe,
                            reweight=False,
                            base_dir=base_dir,
                            shuffle=False,
                            is_multilabel=multi_label)
                weights = torch.load(
                                f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{name}.pth",
                                map_location=device,
                                weights_only=True,
                            )
                test_model = DenseNet_Model(weights=None, out_feature=11)
                test_model.load_state_dict(weights)
                model_testing(
                        test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)
            if external_ood_test:
                testing_data = pd.read_csv(external_test_path)
                test_loader = prepare_dataloaders(
                            testing_data["Path"],
                            testing_data[labels].values,
                            testing_data,
                            masked,
                            clahe,
                            reweight=False,
                            base_dir=base_dir,
                            shuffle=False,
                            is_multilabel=multi_label)
                weights = torch.load(
                                f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{name}.pth",
                                map_location=device,
                                weights_only=True,
                            )
                test_model = DenseNet_Model(weights=None, out_feature=11)
                test_model.load_state_dict(weights)
                model_testing(
                        test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)  

    elif task == "race":
        if training:
            top_races = train_data["race"].value_counts().index[:5]
            train_data = train_data[train_data["race"].isin(top_races)].copy()
            val_data = val_data[val_data["race"].isin(top_races)].copy()
            train_data["race_encoded"] = label_encoder.fit_transform(train_data["race"])
            val_data["race_encoded"] = label_encoder.transform(val_data["race"])
            labels = label_encoder.classes_
            train_loader = prepare_dataloaders(
                train_data["Path"],
                train_data["race_encoded"].values,
                train_data,
                masked,
                clahe,
                reweight=reweight,
                base_dir=base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_dataloaders(
                val_data["Path"],
                val_data["race_encoded"].values,
                val_data,
                masked,
                clahe,
                reweight=reweight,
                base_dir=base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            criterion = LabelSmoothingLoss(smoothing=0.1, mode='multiclass')
            # model = DenseNet_Model(
            #     weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
            #     out_feature=5,
            # )
            # model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            # If you want to use transfer learning and want to diagnostic latent represetation preserve then run below lines of code
            base_model = DenseNet_Model(
                weights=None,
                out_feature=5,
            )
            model = model_transfer_learning(
                f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{trained_model_path}.pth",
                base_model,
                device,
            )

            model_training(
            model,
            train_loader,
            val_loader,
            criterion,
            task,
            labels,
            epoch,
            device=device,
            multi_label=multi_label,
            is_swa=True,
        )
            torch.save(
            model.state_dict(),
            f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{name}.pth",
        )

            testing_data = pd.read_csv(test_output_path)
            testing_data["race_encoded"] = label_encoder.fit_transform(
                        testing_data["race"]
                        )
            
            labels = label_encoder.classes_
            test_loader = prepare_dataloaders(
                            testing_data["Path"],
                            testing_data["race_encoded"].values,
                            testing_data,
                            masked,
                            clahe,
                            reweight=False,
                            base_dir=base_dir,
                            shuffle=False,
                            is_multilabel=multi_label,
                        )
            weights = torch.load(
                            f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{name}.pth",
                            map_location=device,
                            weights_only=True,
                        )
            test_model = DenseNet_Model(weights=None, out_feature=5)
            test_model.load_state_dict(weights)
            model_testing(
                    test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)

        if external_ood_test:
            testing_data = pd.read_csv(external_test_path)
            testing_data["race_encoded"] = label_encoder.fit_transform(
                        testing_data["race"]
                        )
            
            labels = top_races.values
            test_loader = prepare_dataloaders(
                            testing_data["Path"],
                            testing_data["race_encoded"].values,
                            testing_data,
                            masked,
                            clahe,
                            reweight=False,
                            base_dir=base_dir,
                            shuffle=False,
                            is_multilabel=multi_label,
                        )
            weights = torch.load(
                            f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{name}.pth",
                            map_location=device,
                            weights_only=True,
                        )
            test_model = DenseNet_Model(weights=None, out_feature=5)
            test_model.load_state_dict(weights)
            model_testing(
                    test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)

    # weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101.pth"
    # lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessingmimic_diagnostic_101.pth"
    # clahe_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_diagnostic_101.pth"
    # generate_tabel(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_data, base_dir)
    # generate_strip_plot(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_data, multi_label)
                
            