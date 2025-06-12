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

from data_preprocessing.process_dataset import (
    add_demographic_data,
    add_lung_mask_dataset,
    add_metadata,
    cleaning_datasets,
    get_group_by_data,
    merge_file_path_and_add_dicom_id,
    sampling_datasets,
)
from datasets.dataloader import prepare_mimic_dataloaders, prepare_chexpert_dataloaders
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from evaluation.model_testing import model_testing, model_testing_metrics_eval
from evaluation.groupby_eval import groupby_testing
from models.build_model import DenseNet_Model, model_transfer_learning, Resnet_Model, Efficientnet_Model, ConvNeXt_Model
from train.model_training import model_training
from helper.losses import LabelSmoothingLoss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 102
    epoch = 30
    training = False
    task = "diagnostic"
    is_groupby = False
    dataset = "mimic"
    masked = False
    multi_label = True
    external_ood_test = False
    normal_testing = True

    base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    name = (
        f"traininig_cosine_warmrestart_label_smoothing_{dataset}_{task}_{random_state}"
        if training
        else f"testing_earlystop_valauc_and_cosine_lr_{dataset}_{task}_{random_state}"
    )
    label_encoder = LabelEncoder()

    external_val_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/validation_clean_dataset.csv"
    external_train_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train_clean_dataset.csv"
    
    train_output_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/train_mask_clean_dataset.csv"
        if masked
        else "/deep_learning/output/Sutariya/main/mimic/dataset/train_clean_dataset.csv"
    )
    valid_output_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/test_mask_clean_dataset.csv"
        if masked
        else "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
    )
    meta_file_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-metadata.csv.gz"
    train_mask_file_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/train_dataset.csv"
    )
    test_mask_file_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/test_dataset.csv"
    )
    train_file_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/train_dataset.csv"
    )
    test_file_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/test_dataset.csv"
    )
    demographic_data_path = (
        "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/admissions.csv.gz"
    )
    all_dataset_path = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/mimic-cxr-2.0.0-chexpert.csv.gz"

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
        "Pleural Effusion",
    ]

    # if not os.path.exists("/deep_learning/output/Sutariya/main/mimic/wandb"):
    #     os.mkdir("/deep_learning/output/Sutariya/main/mimic/wandb")
    # os.environ["WANDB_DIR"] = os.path.abspath(
    #     "/deep_learning/output/Sutariya/main/mimic/wandb"
    # )

    # wandb.init(
    #     project=f"mimic_preprocessing_groupby_{task}"
    #     if is_groupby
    #     else f"mimic_preprocessing_{task}",
    #     name=f"{name}",
    #     dir="/deep_learning/output/Sutariya/main/mimic/wandb",
    #     config={
    #         "learning_rate": 0.0001,
    #         "Task": task,
    #         "save_model_file_name": name,
    #         "Uncertain Labels": "-1 = 0, NAN = 0",
    #         "epochs": epoch,
    #         "Augmentation": "Yes",
    #         "optimiser": "AdamW",
    #         "architecture": "DenseNet121",
    #         "dataset": dataset,
    #         "Standardization": "Yes",
    #     },
    # )




    if not (os.path.exists(train_output_path) and os.path.exists(valid_output_path)):
        # Merge and clean the data
        total_data_merge = add_demographic_data(all_dataset_path, demographic_data_path)
        total_data_merge = add_metadata(total_data_merge, meta_file_path)
        total_data_clean = cleaning_datasets(total_data_merge, False)
        sampling_total_dataset = sampling_datasets(total_data_clean)
        if masked:
            total_data_path_merge = merge_file_path_and_add_dicom_id(
                "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/IMAGE_FILENAMES.txt",
                sampling_total_dataset,
            )
            sampling_total_dataset = add_lung_mask_dataset(total_data_path_merge)
            if not (os.path.exists(train_mask_file_path)) and not (
                os.path.exists(test_mask_file_path)
            ):
                split_train_test_data(
                    sampling_total_dataset, 40, train_file_path, test_file_path, "race"
                )
            else:
                print("Data is already sampled spit into train and test")
        else:
            if not (os.path.exists(train_file_path)) and not (
                os.path.exists(test_file_path)
            ):
                split_train_test_data(
                    sampling_total_dataset, 40, train_file_path, test_file_path, "race"
                )
            else:
                print("Data is already sampled spit into train and test")

        train_data = pd.read_csv(train_file_path)
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
            f"Files {train_output_path} && {valid_output_path} already exists. Skipping save..."
        )

    if training:
        train_data = pd.read_csv(train_output_path)
        val_data = pd.read_csv(valid_output_path)

        if task == "diagnostic":
            train_loader = prepare_mimic_dataloaders(
                train_data["Path"],
                train_data[labels].values,
                train_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_mimic_dataloaders(
                val_data["Path"],
                val_data[labels].values,
                val_data,
                masked,
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
        elif task == "race":
            top_races = train_data["race"].value_counts().index[:5]
            train_data = train_data[train_data["race"].isin(top_races)].copy()
            val_data = val_data[val_data["race"].isin(top_races)].copy()
            labels = top_races.values
            train_data["race_encoded"] = label_encoder.fit_transform(train_data["race"])
            val_data["race_encoded"] = label_encoder.transform(val_data["race"])
            train_loader = prepare_mimic_dataloaders(
                train_data["Path"],
                train_data["race_encoded"].values,
                train_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_mimic_dataloaders(
                val_data["Path"],
                val_data["race_encoded"].values,
                val_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            criterion = nn.CrossEntropyLoss()
            # model = DenseNet_Model(
            #     weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
            #     out_feature=5,
            # )
            # model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            # If you want to use transfer learning and want to diagnostic latent represetation preserve then uncoomment below lines of code
            base_model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=5,
            )
            model = model_transfer_learning(
                "/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_model_swa_diagnostic_101.pth",
                base_model,
                device,
            )
        elif task == "ethnicity":
            train_data["ethnicity_encoded"] = label_encoder.fit_transform(
                train_data["ethnicity"]
            )
            val_data["ethnicity_encoded"] = label_encoder.transform(
                val_data["ethnicity"]
            )
            train_loader = prepare_mimic_dataloaders(
                train_data["Path"],
                train_data["ethnicity_encoded"].values,
                train_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_mimic_dataloaders(
                val_data["Path"],
                val_data["ethnicity_encoded"].values,
                val_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=4,
            )
            model = model_transfer_learning(
                "/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_model_swa_diagnostic_101.pth",
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

    else:
        if external_ood_test:
            testing_data = pd.read_csv(external_val_path)
            # external_train_data = pd.read_csv(external_train_path)
            # testing_data = pd.concat([external_val_data, external_train_data])
            if is_groupby:
                groupby_testing(all_dataset_path, demographic_data_path, None, valid_output_path, model_path= "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_earlystop_valauc_and_cosine_lr_mimic_diagnostic_101.pth", 
                                task=task, name=name, device=device, masked=masked, is_multilabel=multi_label, validate_data= False, base_dir=base_dir)
            else:
                testing_data = pd.read_csv(valid_output_path)
                if normal_testing:
                    if task == "diagnostic":
                        test_loader = prepare_mimic_dataloaders(
                        testing_data["Path"],
                        testing_data[labels].values,
                        testing_data,
                        masked,
                        base_dir,
                        shuffle=False,
                        is_multilabel=multi_label)
                        weights = torch.load(
                            "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_cosine_label_smoothin_mimic_diagnostic_101.pth",
                            map_location=device,
                            weights_only=True,
                        )
                        test_model = DenseNet_Model(weights=None, out_feature=11)

                    elif task == "race":
                        top_races = testing_data["race"].value_counts().index[:5]
                        testing_data = testing_data[testing_data["race"].isin(top_races)].copy()
                        labels = top_races.values
                        testing_data["race_encoded"] = label_encoder.fit_transform(
                            testing_data["race"]
                        )
                        test_loader = prepare_mimic_dataloaders(
                            testing_data["Path"],
                            testing_data["race_encoded"].values,
                            testing_data,
                            masked,
                            base_dir,
                            shuffle=False,
                            is_multilabel=multi_label,
                        )
                        weights = torch.load(
                            "/deep_learning/output/Sutariya/main/mimic/checkpoints/model_traininig_diagnostic_102.pth",
                            map_location=device,
                            weights_only=True,
                        )
                        test_model = DenseNet_Model(weights=None, out_feature=5)
                    elif task == "ethnicity":
                        testing_data["ethnicity_encoded"] = label_encoder.fit_transform(
                            testing_data["ethnicity"]
                        )
                        test_loader = prepare_mimic_dataloaders(
                            testing_data["Path"],
                            testing_data["ethnicity_encoded"].values,
                            testing_data,
                            masked,
                            base_dir,
                            shuffle=False,
                            is_multilabel=multi_label,
                        )
                        weights = torch.load(
                            "ethnicity_model_50.pth", map_location=device, weights_only=True
                        )
                        test_model = DenseNet_Model(weights=None, out_feature=3)
                    test_model.load_state_dict(weights)
                    model_testing(
                    test_loader, test_model, labels, task, name, device, multi_label=multi_label
                )
                
                else:
                    if task == "diagnostic":
                        model_testing_metrics_eval(
                        testing_data, test_model, labels, task, name, masked=masked, device=device, multi_label=multi_label, group_name=None, threshold_finding=False,
                        metrics_saving=True, threshold_file_path="deep_learning/output/Sutariya/main/mimic/evaluation_files/testing_earlystop_valauc_and_cosine_lr_mimic_diagnostic_102_threshold.csv", is_groupby_testing=True
                    )

                
        else:
            if is_groupby:
                groupby_testing(all_dataset_path, demographic_data_path, None, valid_output_path, model_path= "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_earlystop_valauc_and_cosine_lr_mimic_diagnostic_101.pth", validate_data= False, base_dir=base_dir)
            else:
                testing_data = pd.read_csv(valid_output_path)
                if normal_testing:
                    if task == "diagnostic":
                        test_loader = prepare_mimic_dataloaders(
                        testing_data["Path"],
                        testing_data[labels].values,
                        testing_data,
                        masked,
                        base_dir,
                        shuffle=False,
                        is_multilabel=multi_label)
                        weights = torch.load(
                            "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_cosine_label_smoothin_mimic_diagnostic_101.pth",
                            map_location=device,
                            weights_only=True,
                        )
                        test_model = DenseNet_Model(weights=None, out_feature=11)

                    elif task == "race":
                        top_races = testing_data["race"].value_counts().index[:5]
                        testing_data = testing_data[testing_data["race"].isin(top_races)].copy()
                        labels = top_races.values
                        testing_data["race_encoded"] = label_encoder.fit_transform(
                            testing_data["race"]
                        )
                        test_loader = prepare_mimic_dataloaders(
                            testing_data["Path"],
                            testing_data["race_encoded"].values,
                            testing_data,
                            masked,
                            base_dir,
                            shuffle=False,
                            is_multilabel=multi_label,
                        )
                        weights = torch.load(
                            "/deep_learning/output/Sutariya/main/mimic/checkpoints/model_traininig_diagnostic_102.pth",
                            map_location=device,
                            weights_only=True,
                        )
                        test_model = DenseNet_Model(weights=None, out_feature=5)
                    elif task == "ethnicity":
                        testing_data["ethnicity_encoded"] = label_encoder.fit_transform(
                            testing_data["ethnicity"]
                        )
                        test_loader = prepare_mimic_dataloaders(
                            testing_data["Path"],
                            testing_data["ethnicity_encoded"].values,
                            testing_data,
                            masked,
                            base_dir,
                            shuffle=False,
                            is_multilabel=multi_label,
                        )
                        weights = torch.load(
                            "ethnicity_model_50.pth", map_location=device, weights_only=True
                        )
                        test_model = DenseNet_Model(weights=None, out_feature=3)
                    test_model.load_state_dict(weights)
                    model_testing(
                    test_loader, test_model, labels, task, name, device, multi_label=multi_label
                )
                
                else:
                    if task == "diagnostic":
                        test_model = DenseNet_Model(weights=None, out_feature=11)
                        model_testing_metrics_eval(
                        testing_data, test_model, labels, task, name, masked=masked, device=device, multi_label=multi_label, group_name=None, threshold_finding=False,
                        metrics_saving=True, threshold_file_path="deep_learning/output/Sutariya/main/mimic/evaluation_files/testing_earlystop_valauc_and_cosine_lr_mimic_diagnostic_102_threshold.csv", is_groupby_testing=True
                    )

                


