import multiprocessing
import os
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import cv2
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision

def decode_rle_numpy(rle_str: Optional[str], shape: Tuple[int, int]) -> np.ndarray:
    if pd.isna(rle_str) or not isinstance(rle_str, str):
        return np.zeros(shape, dtype=np.uint8)

    rle = np.fromiter(map(int, rle_str.strip().split()), dtype=np.int32)
    starts = rle[0::2] - 1
    lengths = rle[1::2]

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for start, length in zip(starts, lengths):
        mask[start:start + length] = 1
    return mask.reshape(shape)

class ApplyLungMask:
    def __init__(
        self,
        original_shape: Tuple[int, int] = (1024, 1024),
    ):
        self.original_shape = original_shape

    def compute_combined_mask(
        self, left_rle: Optional[str], right_rle: Optional[str], heart_rle: Optional[str]
    ) -> np.ndarray:
        left = decode_rle_numpy(left_rle, self.original_shape)
        right = decode_rle_numpy(right_rle, self.original_shape)
        heart = decode_rle_numpy(heart_rle, self.original_shape)
        combined = np.clip(left + right + heart, 0, 1)
        return combined


def crop_image_to_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:

    image = np.array(image)
    assert image.shape[:2] == mask.shape[:2], "Image and mask must have same height and width"

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        print("Warning: Mask is empty. Returning original image.")
        return image

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped_np = image[rmin:rmax+1, cmin:cmax+1]
    cropped_img = Image.fromarray(cropped_np)

    return cropped_img


class MyDataset(Dataset):
    def __init__(
        self,
        image_paths: Union[list, pd.Series],
        labels: Union[list, np.ndarray, pd.Series],
        dataframe: pd.DataFrame,
        masked: bool = False,
        clahe: bool = False,
        transform: Union[torchvision.transforms.Compose, None] = None,
        base_dir: Optional[str] = None,
        is_multilabel: bool = True,
        external_ood_test:bool = False
    ):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.masked = masked
        self.clahe = clahe
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
        self.is_multilabel = is_multilabel
        self.external_ood_test = external_ood_test

        self.create_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.masker  = ApplyLungMask()
        # if self.masked:
        #     if self.external_ood_test:
        #         self.base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
        #     else:
        #         self.base_dir = '/deep_learning/output/Sutariya/MIMIC-CXR-MASK/'

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.image_paths[idx]
        full_path = os.path.join(self.base_dir, file_path)

        image = Image.open(full_path)
        row = self.df.iloc[idx]

        if self.masked:
            image = image.resize((1024, 1024))
            mask = self.masker.compute_combined_mask(row["Left Lung"], row["Right Lung"], row["Heart"])
            image = crop_image_to_mask(image, mask)

        image = image.resize((224, 224))

        if self.clahe:
            img_np = np.array(image)
            img_np = self.create_clahe.apply(img_np)
            image =  Image.fromarray(img_np)

        if self.transform:
            image = self.transform(image)

        if self.is_multilabel:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)

        return image, label, self.image_paths[idx]