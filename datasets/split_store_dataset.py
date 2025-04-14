
from sklearn.model_selection import train_test_split
import os
from torchvision.io import read_image
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import torch
import pandas as pd

def split_and_save_datasets(dataset, train_path='train.csv', val_path='val.csv', val_size=0.1, random_seed=42):

    train_data, val_data = train_test_split(dataset, test_size=val_size, random_state=random_seed)
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)

def load_and_store_ethnic_images_labels(dataset, path, device):
    """
    Loads images and processes ethnic labels.

    Args:
    - training_data_merge (DataFrame): Contains image paths and diagnostic labels.

    Returns:
    - data_images (List[Tensor]): List of image tensors.
    - data_labels (Tensor): Tensor of diagnostic class labels.
    """

    label_encoder = LabelEncoder()
    if not os.path.exists(path):
        data_images = []
        paths = tqdm(dataset['Path'], desc="Loading images")
        for p in paths:
            full_path = '/deep_learning/output/Sutariya/chexpert' + '/' + str(p)
            img = read_image(full_path)
            data_images.append(img)
            paths.set_postfix({'Loaded': len(data_images)})
        torch.save(data_images, path)
    else:
        data_images = torch.load(path, map_location=device, weights_only=True)

    dataset['ethnicity_encoded'] = label_encoder.fit_transform(dataset['ethnicity'])
    data_labels = torch.tensor(dataset['ethnicity_encoded'].values, dtype=torch.long)
    
    return data_images, data_labels

def sample_test(df_group, disease, n_samples=N):
    positives = df_group[df_group[disease] == 1]

    sampled_pos = positives.sample(n=n_samples, random_state=42)

    return sampled_pos


def split_train_test_data(dataset, N):
    test_dfs = []
    diseases = [
        "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
        "Enlarged Cardiomediastinum", "Fracture", "Lung Lesion",
        "Lung Opacity", "No Finding", "Pleural Effusion", "Pleural Other",
        "Pneumonia", "Pneumothorax", 'Support Devices'
    ]
    ethnic_groups = ["Non-Hispanic/Non-Latino", "Hispanic/Latino"]

    df = dataset
    train_ethnic_group = df.groupby('ethnicity')
    group_by_data = {}
    for group, data in train_ethnic_group:
        group_by_data[group] = data
    for ethnicity in ethnic_groups:
        df_ethnicity = group_by_data[ethnicity]
        for disease in diseases:
            sampled_test = sample_test(df_ethnicity, disease, N)
            test_dfs.append(sampled_test)

    # Combine test samples
    final_test_df = pd.concat(test_dfs).drop_duplicates().reset_index(drop=True)
    train_df = df[~df['subject_id'].isin(final_test_df['subject_id'].unique())].reset_index(drop=True) 

    # Save
    final_test_df.to_csv('test_dataset.csv', index=False)
    train_df.to_csv('train_dataset.csv', index=False)

    print("✅ Done! Test shape:", final_test_df.shape, train_df.shape)

def load_and_store_diagnostic_images_labels(dataset, path, device):
    """
    Loads images and processes diagnostic labels.

    Args:
    - training_data_merge (DataFrame): Contains image paths and diagnostic labels.

    Returns:
    - data_images (List[Tensor]): List of image tensors.
    - data_labels (Tensor): Tensor of diagnostic class labels.
    """

    
    if not os.path.exists(path):
        data_images = []
        paths = tqdm(dataset['Path'], desc="Loading images")
        for path in paths:
            full_path = '/deep_learning/output/Sutariya/chexpert' + '/' + str(path)
            img = read_image(full_path)
            data_images.append(img)
            paths.set_postfix({'Loaded': len(data_images)})
        torch.save(data_images, path)
    else:
        data_images = torch.load(path, map_location=device, weights_only=True)

    data_labels = dataset[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices']].values
    data_labels = torch.tensor(data_labels, dtype=torch.long)
    
    return data_images, data_labels


def load_and_store_race_images_labels(dataset, save_path, device):
    """
    Loads images and processes race labels for multi-class classification.

    Args:
    - training_data_merge (DataFrame): Contains image paths and race labels.

    Returns:
    - data_images (List[Tensor]): List of image tensors.
    - data_labels (Tensor): Tensor of race class labels.
    """
    label_encoder = LabelEncoder()

    # Keep only top 3 race categories
    top_races = dataset['race'].value_counts().index[:3]
    dataset = dataset[dataset['race'].isin(top_races)].copy()

    if not os.path.exists(save_path):
        data_images = []
        paths = tqdm(dataset['Path'], desc="Loading images")
        for path in paths:
            full_path = '/deep_learning/output/Sutariya/chexpert' + '/' + str(path)
            img = read_image(full_path)
            data_images.append(img)
            paths.set_postfix({'Loaded': len(data_images)})
        torch.save(data_images, save_path)

    else:
        data_images = torch.load(save_path, map_location=device, weights_only=True)

    dataset['race_encoded'] = label_encoder.fit_transform(dataset['race'])
    data_labels = torch.tensor(dataset['race_encoded'].values, dtype=torch.long)
    
    return data_images, data_labels