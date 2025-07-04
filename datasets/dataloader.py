from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from datasets.data import MyDataset
from typing import Optional, Union

import pandas as pd

def prepare_dataloaders(
    images_path: Union[pd.Series, list, str],
    labels: Union[pd.Series, list, None],
    dataframe: pd.DataFrame,
    masked: bool=False,
    clahe: bool=False,
    reweight : bool=False,
    base_dir: Optional[str] = None,
    shuffle: bool = False,
    is_multilabel: bool = True,
) -> DataLoader:
    transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Resize(
                (250, 250), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
            transforms.Normalize(mean=[0.5062] * 3, std=[0.2873] * 3),
            transforms.RandomResizedCrop(
                (224, 224),
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
        clahe,
        transform,
        base_dir,
        is_multilabel=is_multilabel,
    )

    if reweight:
        total_race = sum(dataframe['race'].value_counts())
        race_weights = {r: total_race / c for r, c in dataframe['race'].value_counts().items()}
        dataframe['sample_weight'] = dataframe['race'].map(race_weights)
        sample_weights = dataframe['sample_weight'].values
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)

        data_loader = DataLoader(
            dataset, batch_size=8, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler
        )
    else:
        data_loader = DataLoader(
            dataset, batch_size=8, shuffle=shuffle, num_workers=8, pin_memory=True, drop_last=True
        )
    return data_loader
