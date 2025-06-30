import pandas as pd
from sklearn.model_selection import train_test_split
import os
from typing import Union

def split_and_save_datasets(
    dataset: pd.DataFrame, train_path : Union[list, str, os.path] ="train.csv", val_path: Union[list, str, os.path]="val.csv", val_size: int=0.1, random_seed: int=42
)-> None:
    train_data, val_data = train_test_split(
        dataset, test_size=val_size, random_state=random_seed
    )
    train_data.to_csv(train_path, index=False)
    val_data.to_csv(val_path, index=False)


def sample_test(df_group: pd.DataFrame, disease: Union[list, str], n_samples: int=0)-> pd.DataFrame:
    positives = df_group[df_group[disease] == 1]
    print(f"Disease:{disease} have {len(positives)} values.")
    sampled_pos = positives.sample(n=n_samples, random_state=42)

    return sampled_pos


def split_train_test_data(dataset: pd.DataFrame, N: int, train_path: Union[list, str, os.path], test_path: Union[list, str, os.path], split_by: str="race")-> None:
    test_dfs = []
    diseases = [
        "No Finding",
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Opacity",
        "Lung Lesion",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
    ]
    # Select only top 5 values of the column
    groups = dataset[split_by].value_counts().index[:5].values
    print(groups)
    df = dataset.copy()
    train_group = df.groupby(split_by)
    group_by_data = {}
    for group, data in train_group:
        group_by_data[group] = data
    for group in groups:
        df_group = group_by_data[group]
        for disease in diseases:
            print(f"Computed for {disease}...")
            if len(df_group[disease]) < N:
                sampled_test = sample_test(df_group, disease, len(df_group[disease]))
            else:
                sampled_test = sample_test(df_group, disease, N)
            test_dfs.append(sampled_test)

    # Combine test samples
    final_test_df = pd.concat(test_dfs).drop_duplicates().reset_index(drop=True)
    train_df = df[
        ~df["subject_id"].isin(final_test_df["subject_id"].unique())
    ].reset_index(drop=True)

    # Save
    final_test_df.to_csv(test_path, index=False)
    train_df.to_csv(train_path, index=False)

    print("✅ Done! Test shape:", final_test_df.shape, train_df.shape)
