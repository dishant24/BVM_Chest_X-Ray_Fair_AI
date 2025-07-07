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
        margin_radius: int = 20,
        original_shape: Tuple[int, int] = (1024, 1024),
        image_shape: Tuple[int, int] = (224, 224),
    ):
        self.margin_radius = margin_radius
        self.original_shape = original_shape
        self.image_shape = image_shape

    def dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        selem = disk(self.margin_radius)
        return binary_dilation(mask, structure=selem).astype(np.uint8)

    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        resized = cv2.resize(mask, (self.image_shape[1], self.image_shape[0]), interpolation=cv2.INTER_NEAREST)
        return resized.astype(np.uint8)

    def compute_combined_mask(
        self, left_rle: Optional[str], right_rle: Optional[str], heart_rle: Optional[str]
    ) -> np.ndarray:
        left = self.dilate_mask(decode_rle_numpy(left_rle, self.original_shape))
        right = self.dilate_mask(decode_rle_numpy(right_rle, self.original_shape))
        heart = self.dilate_mask(decode_rle_numpy(heart_rle, self.original_shape))
        combined = np.clip(left + right + heart, 0, 1)
        return self.resize_mask(combined)

def compute_mask_entry(row: pd.Series, masker: ApplyLungMask) -> Tuple[str, np.ndarray]:
    key = row["Path"]
    mask = masker.compute_combined_mask(row["Left Lung"], row["Right Lung"], row["Heart"])
    return key, mask.astype(np.uint8)

class MyDataset(Dataset):
    def __init__(
        self,
        image_paths: Union[list, pd.Series],
        labels: Union[list, np.ndarray, pd.Series],
        dataframe: pd.DataFrame,
        masked: bool = False,
        clahe: bool = False,
        transform: torchvision.transforms.Compose = None,
        base_dir: Optional[str] = None,
        is_multilabel: bool = True,
    ):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.masked = masked
        self.clahe = clahe
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
        self.is_multilabel = is_multilabel

        self.create_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        if self.masked:
            self.base_dir = '/deep_learning/output/Sutariya/MIMIC-CXR-MASK/'

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.image_paths[idx]
        full_path = os.path.join(self.base_dir, file_path)

        image = Image.open(full_path).convert("L").resize((224, 224))
        
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
