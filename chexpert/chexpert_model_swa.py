import os
import sys
import cv2
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import wandb
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
import torch.nn as nn
import torchvision



from datasets.data import ApplyLungMask 
from data_preprocessing.process_dataset import (
    add_demographic_data,
    cleaning_datasets,
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
from concurrent.futures import ProcessPoolExecutor
from functools import partial


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
    is_groupby = True
    multi_label = True
    external_ood_test = True
    #Change the path acrroding to model usage
    trained_model_path = 'traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101' if task == 'race' else None
    base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'

    name = (
        f"traininig_auroc_stopping_cosine_label_smoothing{dataset}_{task}_{random_state}"
        if training
        else f"testing_groupby_with_cosine_label_smoothing_{dataset}_{task}_{random_state}"
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

    external_test_path = "/deep_learning/output/Sutariya/main/mimic/dataset/validation_clean_dataset.csv"
    demographic_data_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/demographics_CXP.csv"
    train_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/train_clean_dataset.csv"
    valid_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/validation_clean_dataset.csv"
    test_output_path = "/deep_learning/output/Sutariya/main/chexpert/dataset/test_clean_dataset.csv"
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

    if not (os.path.exists(train_output_path) and os.path.exists(valid_output_path)):
        all_data = pd.read_csv(all_dataset_path)
        demographic_data = pd.read_csv(demographic_data_path)
        all_data_merge = merge_dataframe(all_data, demographic_data)
        all_data_clean = cleaning_datasets(all_data_merge)
        all_dataset = sampling_datasets(all_data_clean)
        if not (os.path.exists(train_output_path)) and not (
            os.path.exists(test_output_path)
        ):
            split_train_test_data(
                all_dataset, 35, train_output_path, test_output_path, "race"
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

    
    masked_path = '/deep_learning/input/data/chexmask/chexmask-database-a-large-scale-dataset-of-anatomical-segmentation-masks-for-chest-x-ray-images-1.0.0/Preprocessed/CheXpert.csv'
    masked_data = pd.read_csv(masked_path)
    train_data = pd.read_csv(train_output_path)
    val_data = pd.read_csv(valid_output_path)
    test_data = pd.read_csv(test_output_path)
    all_data = pd.concat([train_data, val_data, test_data])

    def process_row(row, base_dir, save_dir, masker_params):
        try:
            image_path = os.path.join(base_dir, row["Path"])
            save_path = os.path.join(save_dir, row["Path"])

            # Load and resize image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"Warning: could not load image {image_path}")
                return

            image = cv2.resize(image, masker_params["image_shape"], interpolation=cv2.INTER_AREA)

            # Reconstruct masker object inside each process
            masker = ApplyLungMask(**masker_params)
            mask = masker.compute_combined_mask(row["Left Lung"], row["Right Lung"], row["Heart"])
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Save
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, masked_image)
        except Exception as e:
            print(f"Error processing {row['Path']}: {e}")
    
    def apply_and_save_masked_images_parallel(df, base_dir, save_dir, masker):
        masker_params = {
            "margin_radius": masker.margin_radius,
            "original_shape": masker.original_shape,
            "image_shape": masker.image_shape,
        }

        with ProcessPoolExecutor() as executor:
            tasks = []
            for _, row in df.iterrows():
                tasks.append(executor.submit(process_row, row, base_dir, save_dir, masker_params))

            for future in tqdm(tasks):
                future.result()

    masker = ApplyLungMask(
    margin_radius=60,
    original_shape=(1024, 1024),
    image_shape=(224, 224)
    )

    base_image_dir = "/deep_learning/output/Sutariya/main/chexpert/dataset/CheXpert-v1.0-small/"
    train_save_dir = "/deep_learning/output/Sutariya/main/chexpert/dataset/Chexpert-Mask/CheXpert-v1.0-small/"

    apply_and_save_masked_images_parallel(masked_data, base_image_dir, train_save_dir, masker)

    # if task == "diagnostic":
    #     labels = [
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


    #     if training:
    #         train_data = pd.read_csv(train_output_path)
    #         val_data = pd.read_csv(valid_output_path)
    #         train_loader = prepare_dataloaders(
    #             train_data["Path"],
    #             train_data[labels].values,
    #             train_data,
    #             masked,
    #             clahe,
    #             base_dir,
    #             shuffle=True,
    #             is_multilabel=multi_label,
    #         )
    #         val_loader = prepare_dataloaders(
    #             val_data["Path"],
    #             val_data[labels].values,
    #             val_data,
    #             masked,
    #             clahe,
    #             base_dir,
    #             shuffle=True,
    #             is_multilabel=multi_label,
    #         )
    #         criterion = LabelSmoothingLoss(smoothing=0.1, mode='multilabel')
    #         # criterion = nn.BCEWithLogitsLoss()
    #         model = DenseNet_Model(
    #             weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
    #             out_feature=11
    #         )

    #         model_training(
    #         model,
    #         train_loader,
    #         val_loader,
    #         criterion,
    #         task,
    #         labels,
    #         epoch,
    #         device=device,
    #         multi_label=multi_label,
    #         is_swa=True,
    #         )
    #         torch.save(
    #             model.state_dict(),
    #             f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{name}.pth",
    #         )

    #         testing_data = pd.read_csv(test_output_path)
    #         test_loader = prepare_dataloaders(
    #                     testing_data["Path"],
    #                     testing_data[labels].values,
    #                     testing_data,
    #                     masked,
    #                     base_dir,
    #                     shuffle=False,
    #                     is_multilabel=multi_label)
    #         weights = torch.load(
    #                         f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{name}.pth",
    #                         map_location=device,
    #                         weights_only=True,
    #                     )
    #         test_model = DenseNet_Model(weights=None, out_feature=11)
    #         test_model.load_state_dict(weights)
    #         model_testing(
    #                 test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)
    #     if external_ood_test:
    #         testing_data = pd.read_csv(external_test_path)
    #         test_loader = prepare_dataloaders(
    #                     testing_data["Path"],
    #                     testing_data[labels].values,
    #                     testing_data,
    #                     masked,
    #                     base_dir,
    #                     shuffle=False,
    #                     is_multilabel=multi_label)
    #         weights = torch.load(
    #                         f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{name}.pth",
    #                         map_location=device,
    #                         weights_only=True,
    #                     )
    #         test_model = DenseNet_Model(weights=None, out_feature=11)
    #         test_model.load_state_dict(weights)
    #         model_testing(
    #                 test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)  

    # elif task == "race":
    #     if training:
    #         top_races = train_data["race"].value_counts().index[:5]
    #         train_data = train_data[train_data["race"].isin(top_races)].copy()
    #         val_data = val_data[val_data["race"].isin(top_races)].copy()
    #         train_data["race_encoded"] = label_encoder.fit_transform(train_data["race"])
    #         val_data["race_encoded"] = label_encoder.transform(val_data["race"])
    #         labels = label_encoder.classes_
    #         train_loader = prepare_dataloaders(
    #             train_data["Path"],
    #             train_data["race_encoded"].values,
    #             train_data,
    #             masked,
    #             clahe,
    #             base_dir,
    #             shuffle=True,
    #             is_multilabel=multi_label,
    #         )
    #         val_loader = prepare_dataloaders(
    #             val_data["Path"],
    #             val_data["race_encoded"].values,
    #             val_data,
    #             masked,
    #             clahe,
    #             base_dir,
    #             shuffle=True,
    #             is_multilabel=multi_label,
    #         )
    #         criterion = LabelSmoothingLoss(smoothing=0.1, mode='multiclass')
    #         # model = DenseNet_Model(
    #         #     weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1,
    #         #     out_feature=5,
    #         # )
    #         # model = DenseNet_Model(weights=torchvision.models.DenseNet121_Weights.IMAGENET1K_V1, out_feature=5)
    #         # If you want to use transfer learning and want to diagnostic latent represetation preserve then run below lines of code
    #         base_model = DenseNet_Model(
    #             weights=None,
    #             out_feature=5,
    #         )
    #         model = model_transfer_learning(
    #             f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{trained_model_path}.pth",
    #             base_model,
    #             device,
    #         )

    #         model_training(
    #         model,
    #         train_loader,
    #         val_loader,
    #         criterion,
    #         task,
    #         labels,
    #         epoch,
    #         device=device,
    #         multi_label=multi_label,
    #         is_swa=True,
    #     )
    #         torch.save(
    #         model.state_dict(),
    #         f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{name}.pth",
    #     )

    #         testing_data = pd.read_csv(test_output_path)
    #         testing_data["race_encoded"] = label_encoder.fit_transform(
    #                     testing_data["race"]
    #                     )
            
    #         labels = label_encoder.classes_
    #         test_loader = prepare_dataloaders(
    #                         testing_data["Path"],
    #                         testing_data["race_encoded"].values,
    #                         testing_data,
    #                         masked,
    #                         base_dir,
    #                         shuffle=False,
    #                         is_multilabel=multi_label,
    #                     )
    #         weights = torch.load(
    #                         f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{name}.pth",
    #                         map_location=device,
    #                         weights_only=True,
    #                     )
    #         test_model = DenseNet_Model(weights=None, out_feature=5)
    #         test_model.load_state_dict(weights)
    #         model_testing(
    #                 test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)

    #     if external_ood_test:
    #         testing_data = pd.read_csv(external_test_path)
    #         testing_data["race_encoded"] = label_encoder.fit_transform(
    #                     testing_data["race"]
    #                     )
            
    #         labels = top_races.values
    #         test_loader = prepare_dataloaders(
    #                         testing_data["Path"],
    #                         testing_data["race_encoded"].values,
    #                         testing_data,
    #                         masked,
    #                         base_dir,
    #                         shuffle=False,
    #                         is_multilabel=multi_label,
    #                     )
    #         weights = torch.load(
    #                         f"/deep_learning/output/Sutariya/main/chexpert/checkpoints/{name}.pth",
    #                         map_location=device,
    #                         weights_only=True,
    #                     )
    #         test_model = DenseNet_Model(weights=None, out_feature=5)
    #         test_model.load_state_dict(weights)
    #         model_testing(
    #                 test_model, testing_data, labels,  masked, clahe, task, name, base_dir, device, multi_label=multi_label, is_groupby=is_groupby)

    # # weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_auroc_stopping_cosine_label_smoothing_mimic_diagnostic_101.pth"
    # # lung_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_lung_masking_preprocessingmimic_diagnostic_101.pth"
    # # clahe_weights = "/deep_learning/output/Sutariya/main/mimic/checkpoints/traininig_with_clahe_preprocessing_mimic_diagnostic_101.pth"
    # # generate_tabel(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_data, base_dir)
    # # generate_strip_plot(test_loader, lung_test_loader, clahe_test_loader, weights, lung_weights, clahe_weights, device, test_model, labels, testing_data, multi_label)
                
            