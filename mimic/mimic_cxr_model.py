import os
import sys

import pandas as pd
import torch
import torch.nn as nn
import torchvision
import wandb
from sklearn.preprocessing import LabelEncoder

from data_preprocessing.process_dataset import (
    add_demographic_data,
    add_lung_mask_dataset,
    add_metadata,
    cleaning_datasets,
    get_group_by_data,
    merge_file_path_and_add_dicom_id,
    sampling_datasets,
)
from datasets.dataloader import prepare_mimic_dataloaders
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from evaluation.model_testing import model_testing
from models.build_model import DenseNet_Model, model_transfer_learning
from train.model_training import model_training

torch.cuda.empty_cache()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 101
    epoch = 40
    training = True
    task = "diagnostic"
    is_groupby = False
    dataset = "mimic"
    masked = False
    multi_label = True
    base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
    name = (
        f"traininig_without_earlystop_and_cosine_restart_lr_{task}_{random_state}"
        if training
        else f"testing_without_earlystop_cosine_restatrt_lr_{task}_{random_state}"
    )
    label_encoder = LabelEncoder()

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
        },
    )

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
                train_data["file_path"],
                train_data[labels].values,
                train_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_mimic_dataloaders(
                val_data["file_path"],
                val_data[labels].values,
                val_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            criterion = nn.BCEWithLogitsLoss()
            model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=11,
            )
        elif task == "race":
            top_races = train_data["race"].value_counts().index[:5]
            train_data = train_data[train_data["race"].isin(top_races)].copy()
            val_data = val_data[val_data["race"].isin(top_races)].copy()
            labels = top_races.values
            train_data["race_encoded"] = label_encoder.fit_transform(train_data["race"])
            val_data["race_encoded"] = label_encoder.transform(val_data["race"])
            train_loader = prepare_mimic_dataloaders(
                train_data["file_path"],
                train_data["race_encoded"].values,
                train_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_mimic_dataloaders(
                val_data["file_path"],
                val_data["race_encoded"].values,
                val_data,
                masked,
                base_dir,
                shuffle=True,
                is_multilabel=multi_label,
            )
            criterion = nn.CrossEntropyLoss()
            model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=5,
            )
            # model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            # If you want to use transfer learning and want to diagnostic latent represetation preserve then uncoomment below lines of code
            base_model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=5,
            )
            model = model_transfer_learning(
                "/deep_learning/output/Sutariya/main/mimic/checkpoints/mask_model_traininig_diagnostic_101.pth",
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
                "/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_daignostic_model_10.pth",
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
        if is_groupby:
            all_dataset = add_demographic_data(all_dataset_path, demographic_data_path)
            all_dataset.loc[all_dataset["race"].str.startswith("WHITE"), "race"] = (
                "WHITE"
            )
            all_dataset.loc[all_dataset["race"].str.startswith("BLACK"), "race"] = (
                "BLACK"
            )
            all_dataset.loc[all_dataset["race"].str.startswith("ASIAN"), "race"] = (
                "ASIAN"
            )
            test_dataset = pd.read_hdf(test_file_path, key="mask_test")
            race_groupby_dataset = get_group_by_data(test_dataset, "race")
            subject_to_race = dict(zip(all_dataset["subject_id"], all_dataset["race"]))

            assert not test_dataset.duplicated("subject_id").any(), (
                "Duplicate subject_ids found in test_dataset"
            )
            assert not test_dataset.duplicated("file_path").any(), (
                "Duplicate image paths found in test_dataset"
            )

            for idx, row in test_dataset.iterrows():
                sid = row["subject_id"]
                test_race = row["race"]

                assert sid in subject_to_race, (
                    f"subject_id {sid} not found in all_dataset"
                )
                assert subject_to_race[sid] == test_race, (
                    f"Race mismatch for subject_id {sid}: test_dataset has '{test_race}', all_dataset has '{subject_to_race[sid]}'"
                )

            for group in race_groupby_dataset.keys():
                assert not race_groupby_dataset[group].duplicated("subject_id").any(), (
                    f"Duplicate subject_ids in group {group}"
                )
                assert not race_groupby_dataset[group].duplicated("file_path").any(), (
                    f"Duplicate image paths in group {group}"
                )
                test_loader = prepare_mimic_dataloaders(
                    race_groupby_dataset[group]["file_path"],
                    race_groupby_dataset[group][labels].values,
                    race_groupby_dataset[group],
                    masked,
                    base_dir,
                    shuffle=False,
                    is_multilabel=multi_label,
                )
                weights = torch.load(
                    "/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_model_swa_diagnostic_101.pth",
                    map_location=device,
                    weights_only=True,
                )
                test_model = DenseNet_Model(weights=None, out_feature=11)
                test_model.load_state_dict(weights)
                model_testing(
                    test_loader,
                    test_model,
                    labels,
                    task,
                    device,
                    multi_label=multi_label,
                    group_name=group,
                )

        testing_data = pd.read_hdf(test_file_path, key="mask_test")

        if task == "diagnostic":
            test_loader = prepare_mimic_dataloaders(
                testing_data["file_path"],
                testing_data[labels].values,
                testing_data,
                masked,
                base_dir,
                shuffle=False,
                is_multilabel=multi_label,
            )
            weights = torch.load(
                "/deep_learning/output/Sutariya/main/mimic/checkpoints/mimic_mask_diagnostic_model_20.pth",
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
                testing_data["file_path"],
                testing_data["race_encoded"].values,
                testing_data,
                masked,
                base_dir,
                shuffle=False,
                is_multilabel=multi_label,
            )
            weights = torch.load(
                "/deep_learning/output/Sutariya/main/mimic/checkpoints/model_traininig_race_101.pth",
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
            test_loader, test_model, labels, task, device, multi_label=multi_label
        )
