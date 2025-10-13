import pandas as pd
import os
from typing import Union


def sample_test(df_group: pd.DataFrame, disease: Union[list, str], n_samples: int=0)-> pd.DataFrame:
    positives = df_group[df_group[disease] == 1]
    print(f"Disease:{disease} have {len(positives)} values.")
    sampled_pos = positives.sample(n=n_samples, random_state=42)

    return sampled_pos


def split_train_test_data(dataset: pd.DataFrame, N: int, train_path: Union[list, str, os.path], test_path: Union[list, str, os.path], split_by: str="race")-> None:
    """
    Splits the dataset into train and test sets based on stratification by a specified column (default "race").
    Samples up to N test samples per disease within each group specified by 'split_by' column.
    Saves the train and test splits as CSV files to the specified paths.

    Parameters
    ----------
    dataset : pd.DataFrame
        The full dataset to split.
    N : int
        Number of test samples to draw per disease within each group.
    train_path : Union[list, str, os.PathLike]
        File path to save the training set CSV.
    test_path : Union[list, str, os.PathLike]
        File path to save the testing set CSV.
    split_by : str, optional
        Column name by which to group and stratify splits (default "race").

    Returns
    -------
    None
        Outputs train and test CSV files at the specified locations.
    """
    
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
    groups = dataset[split_by].values
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
