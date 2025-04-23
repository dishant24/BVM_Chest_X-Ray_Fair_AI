
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


def sample_test(df_group, disease, n_samples=0):
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