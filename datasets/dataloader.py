from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import torchvision
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
    transform : Union[None, torchvision.transforms, transforms.Compose] = None,
    base_dir: Optional[str] = None,
    shuffle: bool = False,
    is_multilabel: bool = True,
    external_ood_test:bool = False,
) -> DataLoader:

    if transform is None:
        transform = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(0.3),
                transforms.RandomRotation(degrees=10),
            ]
        )
    dataset = MyDataset(
        image_paths= images_path,
        labels= labels,
        dataframe= dataframe,
        masked= masked,
        clahe= clahe,
        transform= transform,
        base_dir= base_dir,
        is_multilabel=is_multilabel,
        external_ood_test =external_ood_test
    )

    if reweight:
        total_race = sum(dataframe['race'].value_counts())
        race_weights = {r: total_race / c for r, c in dataframe['race'].value_counts().items()}
        dataframe['sample_weight'] = dataframe['race'].map(race_weights)
        sample_weights = dataframe['sample_weight'].values
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(dataframe), replacement=True)

        data_loader = DataLoader(
            dataset, batch_size=8, num_workers=8, pin_memory=True, prefetch_factor= 2, drop_last=True, sampler=sampler
        )
    else:
        data_loader = DataLoader(
            dataset, batch_size=8, shuffle=shuffle, num_workers=8, prefetch_factor=2, pin_memory=True, drop_last=True
        )
    return data_loader
