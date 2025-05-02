
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torchvision import transforms
import torch
from PIL import Image
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
import pandas as pd


class ApplyLungMask:
    def __init__(self, left_rle, right_rle, margin_radius=20, original_shape=(1024, 1024), image_shape=(512, 512)):
        self.left_rle = left_rle
        self.right_rle = right_rle
        self.margin_radius = margin_radius
        self.original_shape = original_shape
        self.image_shape = image_shape

    def decode_rle(self, rle_str):
        if isinstance(rle_str, pd.Series):
            rle_str = rle_str.iloc[0]  
        if pd.isna(rle_str):
            return np.zeros(self.original_shape, dtype=np.uint8)
        
        s = list(map(int, rle_str.strip().split()))
        starts, lengths = s[0::2], s[1::2]
        flat_mask = np.zeros(self.original_shape[0] * self.original_shape[1], dtype=np.uint8)
        for start, length in zip(starts, lengths):
            flat_mask[start:start + length] = 1
        return flat_mask.reshape(self.original_shape)


    def dilate_mask(self, mask):
        selem = disk(self.margin_radius)
        return binary_dilation(mask, structure=selem).astype(np.uint8)

    def resize_mask(self, mask):
        mask_img = Image.fromarray(mask.astype(np.uint8) * 255)
        mask_resized = mask_img.resize((self.image_shape[1], self.image_shape[0]), resample=Image.NEAREST)
        return np.array(mask_resized) // 255

    def __call__(self, image):
        if isinstance(image, torch.Tensor):
            image = F.to_pil_image(image)

        image_resized = image.resize(self.image_shape[::-1], Image.BILINEAR)
        image_np = np.array(image_resized)

        left_mask = self.decode_rle(self.left_rle)
        right_mask = self.decode_rle(self.right_rle)

        left_mask = self.dilate_mask(left_mask)
        right_mask = self.dilate_mask(right_mask)

        combined_mask = np.clip(left_mask + right_mask, 0, 1)
        combined_mask = self.resize_mask(combined_mask)

        masked_image = image_np * combined_mask

        return Image.fromarray(masked_image.astype(np.uint8))

        
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, dataframe,  transform=None, base_dir=None, is_multilabel=True):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.transform = transform
        self.is_multilabel = is_multilabel
        self.base_dir = base_dir
        self.dataframe = dataframe

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.base_dir, self.image_paths[idx])
        image = Image.open(path).convert('L')

        # row = self.dataframe.iloc[idx]
        # left_rle = row['Left Lung']
        # right_rle = row['Right Lung']

        # # Apply lung mask
        # masker = ApplyLungMask(left_rle, right_rle)
        # image = masker(image)

        if self.transform:
            image = self.transform(image)

        if self.is_multilabel:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.int32)
        return image, label