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
    """
    Decodes a run-length encoded (RLE) string into a binary mask of a specified shape using NumPy.

    RLE is a compact way to represent binary masks, encoding the start positions and lengths of runs 
    of 1s (mask) in a flattened array. This function converts the RLE string back into the original 2D mask.

    Parameters
    ----------
    rle_str : Optional[str]
        Run-length encoded string with space-separated start positions and lengths.
        If None or NaN, returns an empty mask.
    shape : Tuple[int, int]
        The shape (height, width) of the output mask.

    Returns
    -------
    np.ndarray
        A 2D NumPy array of dtype uint8 representing the decoded mask,
        with 1s indicating the mask and 0s elsewhere.
    """
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
    """
    A utility class to apply lung and heart masks based on run-length encoded (RLE) mask strings,
    decoding them to binary masks of a given original shape and combining them into a single mask.

    Attributes
    ----------
    original_shape : Tuple[int, int]
        The height and width of the mask images (default is (1024, 1024)).
    """
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
    """
    Crops an image tightly around the non-zero regions of a corresponding binary mask with optional horizontal padding.

    The cropping is based on finding the bounding box of the mask's foreground pixels. If the mask is empty,
    it returns the original image and warns the user.

    Parameters
    ----------
    image : np.ndarray
        The input image array (height x width x channels or grayscale).
    mask : np.ndarray
        A binary mask array of the same height and width as the image.

    Returns
    -------
    np.ndarray or PIL.Image.Image
        The cropped image focused on the mask region, returned as a PIL Image.
        Returns original image if the mask is empty.
    """

    image = np.array(image)
    assert image.shape[:2] == mask.shape[:2], "Image and mask must have same height and width"

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        print("Warning: Mask is empty. Returning original image.")
        return image

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add 50 pixel margin to 1024 resolution image to capture peripheral view
    h, w = mask.shape
    cmin_pad = max(cmin - 50, 0)
    cmax_pad = min(cmax + 50, w)

    cropped_np = image[rmin:rmax+1, cmin_pad:cmax_pad]
    cropped_img = Image.fromarray(cropped_np)

    return cropped_img



def compute_mask_entry(row: pd.Series, masker: ApplyLungMask) -> Tuple[str, np.ndarray]:
    key = row["Path"]
    mask = masker.compute_combined_mask(row["Left Lung"], row["Right Lung"], row["Heart"])
    return key, mask.astype(np.uint8)


class MyDataset(Dataset):
    """
    Custom PyTorch dataset class for loading dataset.

    Attributes
    ----------
    image_paths : list
        List of image file paths relative to base_dir.
    labels : list or numpy.ndarray or pd.Series
        Corresponding labels for each image.
    df : pd.DataFrame
        Metadata dataframe containing mask RLEs and other info.
    masked : bool
        Whether to apply lung masks on loaded images.
    crop_masked : bool
        Whether to crop image tightly around lung mask region.
    clahe : bool
        Whether to apply CLAHE histogram equalization for contrast enhancement.
    transform : torchvision.transforms.Compose or None
        Optional transforms to apply on images.
    base_dir : str or None
        Base directory for image paths.
    is_multilabel : bool
        Flag to indicate if labels are multilabel (float tensor) or single-label (long tensor).
    external_ood_test : bool
        Whether to use external dataset directory for masking.

    Methods
    -------
    __len__()
        Returns number of samples.
    __getitem__(idx)
        Loads and processes image and returns image tensor, label tensor, and relative path.
    """
    def __init__(
        self,
        image_paths: Union[list, pd.Series],
        labels: Union[list, np.ndarray, pd.Series],
        dataframe: pd.DataFrame,
        masked: bool = False,
        crop_masked: bool = False,
        clahe: bool = False,
        transform: Union[torchvision.transforms.Compose, None] = None,
        base_dir: Optional[str] = None,
        is_multilabel: bool = True,
        external_ood_test:bool = False
    ):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.masked = masked
        self.crop_masked = crop_masked
        self.clahe = clahe
        self.df = dataframe.reset_index(drop=True)
        self.base_dir = base_dir
        self.transform = transform
        self.is_multilabel = is_multilabel
        self.external_ood_test = external_ood_test

        self.create_clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        self.masker  = ApplyLungMask()

        # We already compute mask image of both dataset and used this computed mask image here
        if self.masked:
            if self.external_ood_test:
                self.base_dir = '/deep_learning/output/Sutariya/main/chexpert/dataset'
            else:
                self.base_dir = '/deep_learning/output/Sutariya/MIMIC-CXR-MASK/'

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_path = self.image_paths[idx]
        full_path = os.path.join(self.base_dir, file_path)

        image = Image.open(full_path)
        row = self.df.iloc[idx]

        
        if self.crop_masked:
            image = image.resize((1024, 1024))
            mask = self.masker.compute_combined_mask(row["Left Lung"], row["Right Lung"], row["Heart"])
            image = crop_image_to_mask(image, mask) 

        if self.clahe:
            image = image.resize((224, 224))
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