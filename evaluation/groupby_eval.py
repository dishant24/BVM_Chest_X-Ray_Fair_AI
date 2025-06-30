from typing import Optional
import pandas as pd

from data_preprocessing.process_dataset import add_demographic_data, get_group_by_data
from datasets.dataloader import prepare_mimic_dataloaders
import torch

from evaluation.model_testing import model_testing
from models.build_model import DenseNet_Model


def groupby_testing(test_file_path: str, model_path: str, task: str, name: str, device:Optional[torch.device], masked: bool=False, clahe: bool=False,  is_multilabel:bool = True, base_dir=None):

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
     
     test_dataset = pd.read_csv(test_file_path)
     top_races = test_dataset["race"].value_counts().index[:5]
     test_dataset = test_dataset[test_dataset["race"].isin(top_races)].copy()
     race_groupby_dataset = get_group_by_data(test_dataset, "race")

     for group in race_groupby_dataset.keys():
         assert not race_groupby_dataset[group].duplicated("subject_id").any(), (
             f"Duplicate subject_ids in group {group}"
         )
         assert not race_groupby_dataset[group].duplicated("Path").any(), (
             f"Duplicate image paths in group {group}"
         )
         test_loader = prepare_mimic_dataloaders(
             race_groupby_dataset[group]["Path"],
             race_groupby_dataset[group][labels].values,
             race_groupby_dataset[group],
             masked,
             clahe,
             base_dir = base_dir,
             shuffle=False,
             is_multilabel=is_multilabel
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
             test_dataset,
             labels,
             task,
             name,
             device,
             multi_label=is_multilabel,
             group_name=group,
         )