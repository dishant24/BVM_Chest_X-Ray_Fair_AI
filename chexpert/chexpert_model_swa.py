
import os
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wandb
from sklearn.preprocessing import LabelEncoder
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from data_preprocessing.process_dataset import (
    add_demographic_data,
    cleaning_datasets,
    get_group_by_data,
    merge_dataframe,
    sampling_datasets,
)
from datasets.dataloader import prepare_chexpert_dataloaders, prepare_mimic_dataloaders
from datasets.split_store_dataset import split_and_save_datasets, split_train_test_data
from evaluation.model_testing import model_testing
from models.build_model import DenseNet_Model, model_transfer_learning
from train.model_training import model_training
from helper.losses import BCEWithLogitsLossWithLabelSmoothing


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random_state = 101
    epoch = 40
    training = False
    task = "diagnostic"
    dataset = "mimic"
    is_groupby = False
    multi_label = True    
    external_ood_test = True
    masked = False
    name = (
        f"traininig_on_{dataset}_{task}_{random_state}"
        if training
        else f"testing_on_{dataset}_{task}_{random_state}"
    )
    label_encoder = LabelEncoder()

    if not os.path.exists("/deep_learning/output/Sutariya/main/chexpert/wandb"):
        os.mkdir("/deep_learning/output/Sutariya/main/chexpert/wandb")
    os.environ["WANDB_DIR"] = os.path.abspath(
        "/deep_learning/output/Sutariya/main/chexpert/wandb"
    )

    wandb.init(
        project=f"cxr_preprocessing_{task}"
        if not is_groupby
        else f"cxr_preprocessing_groupby_{task}",
        dir="/deep_learning/output/Sutariya/main/chexpert/wandb",
        name=name,
        config={
            "learning_rate": 0.0001,
            "Task": task,
            "save_model_file_name": f"{task}_model_swa_{random_state}",
            "Uncertain Labels": "-1 = 0, NAN = 0",
            "epochs": epoch,
            "Augmentation": "Yes",
            "optimiser": "AdamW",
            "SWA": "Yes",
            "architecture": "DenseNet121",
            "dataset": dataset,
            "Standardization": "Yes",
        },
    )

    external_data_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_dataset.csv"
    training_file_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train_dataset.csv"
    test_file_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/test_dataset.csv"
    demographic_data_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/demographics_CXP.csv"
    train_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train_clean_dataset.csv"
    val_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/validation_clean_dataset.csv"
    all_dataset_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train.csv"

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

    if not (os.path.exists(train_output_path) and os.path.exists(val_output_path)):
        all_data = pd.read_csv(all_dataset_path)
        demographic_data = pd.read_csv(demographic_data_path)
        all_data_merge = merge_dataframe(all_data, demographic_data)
        all_data_clean = cleaning_datasets(all_data_merge)
        all_dataset = sampling_datasets(all_data_clean)
        if not (os.path.exists(training_file_path)) and not (
            os.path.exists(test_file_path)
        ):
            split_train_test_data(
                all_dataset, 20, training_file_path, test_file_path, "race"
            )
        else:
            print("Data is already sampled spit into train and test")
        train_data = pd.read_csv(training_file_path)
        split_and_save_datasets(
            train_data,
            train_output_path,
            val_output_path,
            val_size=0.05,
            random_seed=random_state,
        )
    else:
        print(
            f"Files {train_output_path} && {val_output_path} already exists. Skipping save."
        )


    def groupby_testing(all_dataset_path: str, demographic_data_path: str, test_file_path: str, model_path: str,validate_data: bool= True):

        training_data = pd.read_csv(all_dataset_path)
        demographic_data = pd.read_csv(demographic_data_path)
        all_dataset = merge_dataframe(training_data, demographic_data)
        test_dataset = pd.read_csv(test_file_path)
        race_groupby_dataset = get_group_by_data(test_dataset, "race")

        if validate_data:
            all_dataset = add_demographic_data(all_dataset_path, demographic_data_path)
            subject_to_race = dict(zip(all_dataset["subject_id"], all_dataset["race"]))
            all_dataset.loc[all_dataset["race"].str.startswith("WHITE"), "race"] = (
                "WHITE"
            )
            all_dataset.loc[all_dataset["race"].str.startswith("BLACK"), "race"] = (
                "BLACK"
            )
            all_dataset.loc[all_dataset["race"].str.startswith("ASIAN"), "race"] = (
                "ASIAN"
            )
        
            assert not test_dataset.duplicated("subject_id").any(), (
                "Duplicate subject_ids found in test_dataset"
            )
            assert not test_dataset.duplicated("file").any(), (
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
            assert not race_groupby_dataset[group].duplicated("file").any(), (
                f"Duplicate image paths in group {group}"
            )
            test_loader = prepare_mimic_dataloaders(
                race_groupby_dataset[group]["Path"],
                race_groupby_dataset[group][labels].values,
                race_groupby_dataset[group],
                masked,
                base_dir="//gaia/imageData/public/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/",
                shuffle=False,
                is_multilabel=multi_label,
            )
            weights = torch.load(
                model_path,
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
                f'/deep_learning/output/Sutariya/main/chexpert/checkpoint/{name}.csv',
                device,
                multi_label=multi_label,
                group_name=group,
            )


    if training:
        training_dataset = pd.read_csv(train_output_path)
        validation_dataset = pd.read_csv(val_output_path)

        if task == "diagnostic":
            # if you want to train single diagnostic label uncomment below  2 line and comment above 4 line
            # labels = ['Pneumonia']
            criterion = BCEWithLogitsLossWithLabelSmoothing(0.1)
            train_loader = prepare_chexpert_dataloaders(
                training_dataset["Path"],
                training_dataset[labels].values,
                training_dataset,
                masked,
                base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_chexpert_dataloaders(
                validation_dataset["Path"],
                validation_dataset[labels].values,
                validation_dataset,
                masked,
                base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                shuffle=True,
                is_multilabel=multi_label,
            )
            model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=11,
            )
        elif task == "race":
            top_races = training_dataset["race"].value_counts().index[:5]
            labels = top_races.values
            training_dataset = training_dataset[
                training_dataset["race"].isin(top_races)
            ].copy()
            validation_dataset = validation_dataset[
                validation_dataset["race"].isin(top_races)
            ].copy()
            training_dataset["race_encoded"] = label_encoder.fit_transform(
                training_dataset["race"]
            )
            validation_dataset["race_encoded"] = label_encoder.transform(
                validation_dataset["race"]
            )
            train_loader = prepare_chexpert_dataloaders(
                training_dataset["Path"],
                training_dataset["race_encoded"].values,
                training_dataset,
                masked,
                base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_chexpert_dataloaders(
                validation_dataset["Path"],
                validation_dataset["race_encoded"].values,
                validation_dataset,
                masked,
                base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                shuffle=True,
                is_multilabel=multi_label,
            )
            criterion = nn.CrossEntropyLoss()
            # model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
            # Comment below if you want to train race model
            base_model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=5,
            )
            model = model_transfer_learning(
                "/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/diagnostic_model_training_101.pth",
                base_model,
                device,
            )
        elif task == "ethnicity":
            training_dataset["ethnicity_encoded"] = label_encoder.fit_transform(
                training_dataset["ethnicity"]
            )
            validation_dataset["ethnicity_encoded"] = label_encoder.transform(
                validation_dataset["ethnicity"]
            )
            train_loader = prepare_chexpert_dataloaders(
                training_dataset["Path"],
                training_dataset["ethnicity_encoded"].values,
                training_dataset,
                masked,
                base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                shuffle=True,
                is_multilabel=multi_label,
            )
            val_loader = prepare_chexpert_dataloaders(
                validation_dataset["Path"],
                validation_dataset["ethnicity_encoded"].values,
                validation_dataset,
                masked,
                base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                shuffle=True,
                is_multilabel=multi_label,
            )
            criterion = nn.CrossEntropyLoss()
            base_model = DenseNet_Model(
                weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
                out_feature=4,
            )
            model = model_transfer_learning(
                "/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/model_swa_10.pth",
                base_model,
                device,
            )
        else:
            print("Task value should be diagnostic or race or ethnicity...")

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
            f"/deep_learning/output/Sutariya/main/chexpert/checkpoint/{task}/{name}.pth",
        )

    else:
        if external_ood_test:
            if is_groupby:
                groupby_testing(all_dataset_path, demographic_data_path, test_file_path, model_path= "/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/diagnostic_model_training_101.pth", validate_data= False)
            else:  
                testing_data = pd.read_csv(external_data_path)
                if task == "diagnostic":
                    test_loader = prepare_mimic_dataloaders(
                        testing_data["Path"],
                        testing_data[labels].values,
                        testing_data,
                        masked,
                        base_dir='//gaia/imageData/public/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/',
                        shuffle=False,
                        is_multilabel=multi_label,
                    )
                    weights = torch.load(
                        "/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/diagnostic_model_training_101.pth",
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
                        base_dir='//gaia/imageData/public/MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/',
                        shuffle=False,
                        is_multilabel=multi_label,
                    )
                    weights = torch.load(
                        "/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/diagnostic_model_training_101.pth",
                        map_location=device,
                        weights_only=True,
                    )
                    test_model = DenseNet_Model(weights=None, out_feature=5)
                test_model.load_state_dict(weights)
                model_testing(
                    test_loader, test_model, labels, task, f'/deep_learning/output/Sutariya/main/chexpert/checkpoint/{name}.csv',device, multi_label=multi_label
                )
        else:
            if is_groupby:
                groupby_testing(all_dataset_path, demographic_data_path, test_file_path, model_path= "/deep_learning/output/Sutariya/main/mimic/checkpoint/diagnostic/diagnostic_model_training_101.pth", validate_data= False)
            else:
                testing_dataset = pd.read_csv(
                    "/deep_learning/output/Sutariya/main/chexpert/dataset/test_dataset.csv"
                )
                if task == "diagnostic":
                    test_loader = prepare_chexpert_dataloaders(
                        testing_dataset["Path"],
                        testing_dataset[labels].values,
                        testing_dataset,
                        masked,
                        base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                        shuffle=False,
                        is_multilabel=multi_label,
                    )
                    weights = torch.load(
                        "/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/diagnostic_model_training_101.pth",
                        map_location=device,
                        weights_only=True,
                    )
                    test_model = DenseNet_Model(weights=None, out_feature=11)
                elif task == "race":
                    top_races = testing_dataset["race"].value_counts().index[:5]
                    testing_dataset = testing_dataset[
                        testing_dataset["race"].isin(top_races)
                    ].copy()
                    labels = top_races.values
                    testing_dataset["race_encoded"] = label_encoder.fit_transform(
                        testing_dataset["race"]
                    )
                    test_loader = prepare_chexpert_dataloaders(
                        testing_dataset["Path"],
                        testing_dataset["race_encoded"].values,
                        testing_dataset,
                        masked,
                        base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                        shuffle=False,
                        is_multilabel=multi_label,
                    )
                    weights = torch.load(
                        "/deep_learning/output/Sutariya/main/chexpert/checkpoint/diagnostic/diagnostic_model_training_101.pth",
                        map_location=device,
                        weights_only=True,
                    )
                    test_model = DenseNet_Model(weights=None, out_feature=5)
                elif task == "ethnicity":
                    testing_dataset["ethnicity_encoded"] = label_encoder.fit_transform(
                        testing_dataset["ethnicity"]
                    )
                    test_loader = prepare_chexpert_dataloaders(
                        testing_dataset["Path"],
                        testing_dataset["ethnicity_encoded"].values,
                        testing_dataset,
                        masked,
                        base_dir="/deep_learning/output/Sutariya/main/chexpert/dataset",
                        shuffle=False,
                        is_multilabel=multi_label,
                    )
                    weights = torch.load(
                        "race_model_swa_1.pth", map_location=device, weights_only=True
                    )
                    test_model = DenseNet_Model(weights=None, out_feature=4)
                else:
                    print("Task value should be in diagnostic or race or ethnicity...")

                test_model.load_state_dict(weights)
                model_testing(
                    test_loader, test_model, labels, task, f'/deep_learning/output/Sutariya/main/chexpert/checkpoint/{name}.csv',device, multi_label=multi_label
                )
