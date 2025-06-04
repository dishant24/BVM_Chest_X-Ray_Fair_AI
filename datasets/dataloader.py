from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.data import MyDataset
from typing import Optional, Union

import pandas as pd

def prepare_mimic_dataloaders(
    images_path: Union[pd.Series, list, str],
    labels: Union[pd.Series, list, None],
    dataframe: pd.DataFrame,
    masked: bool,
    base_dir: Optional[str] = None,
    shuffle: bool = False,
    is_multilabel: bool = True,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
            transforms.Normalize(mean=[0.5062] * 3, std=[0.2873] * 3),
            transforms.RandomResizedCrop(
                (200, 200),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(degrees=10),
        ]
    )
    dataset = MyDataset(
        images_path,
        labels,
        dataframe,
        masked,
        transform,
        base_dir,
        is_multilabel=is_multilabel,
    )
    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=shuffle, num_workers=8, pin_memory=True
    )

    return data_loader


def prepare_chexpert_dataloaders(
    images_path: Union[pd.Series, list, str],
    labels: Union[pd.Series, list, None],
    dataframe: pd.DataFrame,
    masked: bool,
    base_dir: Optional[str] = None,
    shuffle: bool = False,
    is_multilabel: bool = True,
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
            transforms.Normalize(mean=[0.5062] * 3, std=[0.2873] * 3),
            transforms.RandomResizedCrop(
                (200, 200),
                scale=(0.9, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomRotation(degrees=10),
        ]
    )

    dataset = MyDataset(
        images_path,
        labels,
        dataframe,
        masked,
        transform,
        base_dir,
        is_multilabel=is_multilabel,
    )
    data_loader = DataLoader(
        dataset, batch_size=4, shuffle=shuffle, num_workers=8, pin_memory=True
    )

    return data_loader
