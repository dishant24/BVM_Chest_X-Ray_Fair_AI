
from torch.utils.data import Dataset, DataLoader, random_split
import os
from torchvision import transforms
import torch
from PIL import Image
        
class MyDataset(Dataset):
    def __init__(self, image_paths, labels, task=None, transform=None, base_dir=None, dataset=None):
        self.image_paths = list(image_paths)
        self.labels = labels
        self.transform = transform
        self.task = task
        self.base_dir = 'MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/' if dataset == 'mimic' else '/deep_learning/output/Sutariya/main/chexpert/dataset'

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = os.path.join(self.base_dir, self.image_paths[idx])
        image = Image.open(path).convert('L')

        if self.transform:
            image = self.transform(image)

        if self.task == 'diagnostic':
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label