import multiprocessing
import os
from typing import Optional, Tuple, Union, Dict

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from numba import njit
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from skimage.transform import resize
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision


@njit
def decode_rle_numba(
    starts: np.ndarray, lengths: np.ndarray, shape0: int, shape1: int
) -> np.ndarray:
    flat_mask = np.zeros(shape0 * shape1, dtype=np.uint8)
    for i in range(len(starts)):
        flat_mask[starts[i] : starts[i] + lengths[i]] = 1
    return flat_mask.reshape((shape0, shape1))


def decode_rle(rle_str: Optional[str]) -> np.ndarray:
    shape = (1024, 1024)
    if pd.isna(rle_str) or not isinstance(rle_str, str):
        return np.zeros(shape, dtype=np.uint8)

    rle = np.fromiter(map(int, rle_str.strip().split()), dtype=np.int32)
    starts = rle[0::2]
    lengths = rle[1::2]
    return decode_rle_numba(starts, lengths, shape[0], shape[1])


class ApplyLungMask:
    def __init__(
        self,
        margin_radius: int = 20,
        original_shape: Tuple[int, int] = (1024, 1024),
        image_shape: Tuple[int, int] = (224, 224),
    ):
        self.margin_radius = margin_radius
        self.image_shape = image_shape
        self.original_shape = original_shape

    def dilate_mask(self, mask: np.ndarray) -> np.ndarray:
        selem = disk(self.margin_radius)
        return binary_dilation(mask, structure=selem).astype(np.uint8)

    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        return resize(
            mask, self.image_shape, order=0, preserve_range=True, anti_aliasing=False
        ).astype(np.uint8)

    def compute_combined_mask(
        self, left_rle: Optional[str], right_rle: Optional[str], heart_rle: Optional[str]
    ) -> np.ndarray:
        left = self.dilate_mask(decode_rle(left_rle))
        right = self.dilate_mask(decode_rle(right_rle))
        heart = self.dilate_mask(decode_rle(heart_rle))
        mask = np.clip(left + right + heart, 0, 1)
        return self.resize_mask(mask)


def compute_mask_entry(row: pd.Series, masker: ApplyLungMask) -> Tuple[str, np.ndarray]:
    key = row["file_path"]
    mask = masker.compute_combined_mask(
        row["Left Lung"], row["Right Lung"], row["Heart"]
    )
    return key, mask.astype(np.uint8)


class MyDataset(Dataset):
    def __init__(
        self,
        image_paths: Union[list, pd.Series],
        labels: Union[list, np.ndarray, pd.Series],
        dataframe: pd.DataFrame,
        masked: bool = False,
        transform: torchvision.transforms.Compose = None,
        base_dir: Optional[str] = None,
        is_multilabel: bool = True,
    ):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.masked = masked
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
        self.is_multilabel = is_multilabel
        self.masker = ApplyLungMask(margin_radius=60)
        self.mask_cache: Dict[str, np.ndarray] = {}

        if self.masked:
            results = Parallel(n_jobs=multiprocessing.cpu_count())(
                delayed(compute_mask_entry)(row, self.masker)
                for _, row in tqdm(self.df.iterrows(), total=len(self.df))
            )
            self.mask_cache = dict(results)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.image_paths[idx]
        full_path = os.path.join(self.base_dir, file_path) if self.base_dir else file_path
        image = Image.open(full_path).convert("L").resize((224, 224))

        if self.masked:
            image_np = np.array(image)
            mask = self.mask_cache[file_path]
            masked_image_np = image_np * mask
            image = Image.fromarray(masked_image_np.astype(np.uint8))

        if self.transform:
            image = self.transform(image)

        if self.is_multilabel:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label