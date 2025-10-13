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
    crop_masked : bool=False,
    transform : Union[None, torchvision.transforms, transforms.Compose] = None,
    base_dir: Optional[str] = None,
    shuffle: bool = False,
    is_multilabel: bool = True,
    external_ood_test:bool = False,
) -> DataLoader:
    
    """
    Prepares a PyTorch DataLoader for the given images and labels with optional
    masking, cropping, CLAHE, and data augmentation transforms.

    Parameters
    ----------
    images_path : Union[pd.Series, list, str]
        List or series of image file paths.
    labels : Union[pd.Series, list, None]
        Corresponding labels for the images.
    dataframe : pd.DataFrame
        Dataframe containing metadata such as lung masks.
    masked : bool, optional
        Whether to apply lung masking on the images (default False).
    clahe : bool, optional
        Whether to apply CLAHE contrast enhancement (default False).
    crop_masked : bool, optional
        Whether to crop images based on lung mask (default False).
    transform : torchvision.transforms or None, optional
        Custom transforms to apply; if None, a default Compose pipeline is used (default None).
    base_dir : Optional[str], optional
        Base directory for image file paths (default None).
    shuffle : bool, optional
        Whether to shuffle the dataset (default False).
    is_multilabel : bool, optional
        Whether the task is multilabel classification (default True).
    external_ood_test : bool, optional
        Whether to use an external out-of-distribution test dataset path (default False).

    Returns
    -------
    DataLoader
        A PyTorch DataLoader object for the prepared dataset.
    """

    if transform is None:
        transform = transforms.Compose(
            [   
                transforms.ToTensor(),
                transforms.Resize(
                    (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
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
        crop_masked = crop_masked,
        clahe= clahe,
        transform= transform,
        base_dir= base_dir,
        is_multilabel=is_multilabel,
        external_ood_test =external_ood_test
    )

    data_loader = DataLoader(
            dataset, batch_size=8, shuffle=shuffle, num_workers=8, prefetch_factor=2, pin_memory=True, drop_last=True
    )
    return data_loader
