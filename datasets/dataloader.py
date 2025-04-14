from torchvision import transforms
from datasets.data import MyDataset
from torch.utils.data import DataLoader
import torch

def prepare_dataloaders(data_images, labels, sampler=None, shuffle=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
        transforms.Lambda(lambda i: i/255),
        transforms.Normalize(mean=[0.5062, 0.5062, 0.5062], std=[0.2873, 0.2873, 0.2873]), # Adapt to own standard deviation and mean to Chexpert
        transforms.Lambda(lambda i: i.to(torch.float32)),
        transforms.RandomResizedCrop((224,224), scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(contrast=(0.7, 1.2)) # Randomly change the brightness, contrast, saturation and hue of an image
    ])

    dataset = MyDataset(data_images,labels,transform)
    data_loader = DataLoader(dataset, batch_size=16, shuffle=shuffle, sampler=sampler)

    return data_loader