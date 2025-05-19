from torchvision import transforms
from datasets.data import MyDataset
from torch.utils.data import DataLoader
import torch



def prepare_mimic_dataloaders(images_path, labels, dataframe, masked, base_dir=None, shuffle=False, is_multilabel=True):
     
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), 
    transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
    transforms.Normalize(mean=[0.5062]*3, std=[0.2873]*3),
    transforms.RandomResizedCrop(
        (200, 200),
        scale=(0.9, 1.0),               
        ratio=(0.9, 1.1),              
        interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation(degrees=10)
    ])
    dataset = MyDataset(images_path, labels, dataframe, masked, transform, base_dir, is_multilabel=is_multilabel)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=shuffle, num_workers=4, pin_memory=True, prefetch_factor=2,
                            persistent_workers=True)

    return data_loader

def prepare_chexpert_dataloaders(images_path, labels, dataframe, masked, base_dir=None, shuffle=False, is_multilabel=True):
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC), 
    transforms.Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
    transforms.Normalize(mean=[0.5062]*3, std=[0.2873]*3),
    transforms.RandomResizedCrop(
        (200, 200),
        scale=(0.9, 1.0),               
        ratio=(0.9, 1.1),              
        interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomVerticalFlip(0.3),
    transforms.RandomRotation(degrees=10)
    ])


    dataset = MyDataset(images_path, labels, dataframe, masked,transform, base_dir, is_multilabel=is_multilabel)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=shuffle, num_workers=12, 
                                pin_memory=True)
    return data_loader