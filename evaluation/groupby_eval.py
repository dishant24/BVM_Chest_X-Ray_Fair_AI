from typing import Optional
import pandas as pd

from data_preprocessing.process_dataset import add_demographic_data, get_group_by_data
from datasets.dataloader import prepare_mimic_dataloaders
import torch

from evaluation.model_testing import model_testing
from models.build_model import DenseNet_Model


def groupby_testing(test_file_path: str, model_path: str, task: str, name: str, device:Optional[torch.device], masked: bool=False, clahe: bool=False,  is_multilabel:bool = True, base_dir=None):

    """
    Performs model testing on subgroups of a test dataset stratified by race.
    Loads the test dataset, selects the top 5 races by frequency, groups by race,
    prepares DataLoaders for each group, loads a pre-trained model, and runs evaluation.

    Parameters
    ----------
    test_file_path : str
        Path to the CSV file containing test dataset metadata and labels.
    model_path : str
        Path to the saved model weights file.
    task : str
        Specific task identifier for evaluation.
    name : str
        Name identifier for the evaluation run.
    device : Optional[torch.device]
        Device (CPU/GPU) for model evaluation.
    masked : bool, optional
        Whether to apply lung mask preprocessing (default False).
    clahe : bool, optional
        Whether to apply CLAHE contrast enhancement (default False).
    is_multilabel : bool, optional
        Whether the task is multilabel classification (default True).
    base_dir : optional
        Base directory to prepend image paths.

    Returns
    -------
    None
        Runs evaluation separately on each subgroup, no return value.
    """
    
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