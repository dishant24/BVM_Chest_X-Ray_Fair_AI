import pandas as pd

from typing import Union
import os

def select_most_positive_sample(group: pd.DataFrame) -> pd.Series:
    disease_columns = [
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

    group["positive_count"] = group[disease_columns].sum(axis=1)

    positive_cases = group[group["positive_count"] > 0]

    if not positive_cases.empty:
        selected_sample = positive_cases.loc[positive_cases["positive_count"].idxmax()]
    else:
        selected_sample = group.sample(n=1).iloc[0]

    return selected_sample


def get_group_by_data(data: pd.DataFrame, group_column_name: str) -> pd.DataFrame:
    groups_data = {}
    for name, group_data in data.groupby(group_column_name):
        groups_data[name] = group_data
    return groups_data


# Merge the data with demographic data
def add_demographic_data(training_data: pd.DataFrame, demographic_data: pd.DataFrame)-> pd.DataFrame:
    # Load the CSV files
    df_chexpert = pd.read_csv(training_data, compression="gzip")
    df_patients = pd.read_csv(demographic_data, compression="gzip")

    # Check for duplicate subject_id in patients dataset
    df_patients_unique = df_patients[["subject_id", "race"]].drop_duplicates(
        subset=["subject_id"]
    )

    # Verify uniqueness
    assert df_patients_unique.duplicated(subset=["subject_id"]).sum() == 0, (
        "Duplicate subject_id found in patients dataset"
    )

    # Merge using 'subject_id'
    df_merged = df_chexpert.merge(df_patients_unique, on="subject_id", how="left")
    df_cleaned = df_merged.dropna(subset=["race"])

    return df_cleaned


def merge_file_path_and_add_dicom_id(file_path: Union[list, str, os.path], dataframe: pd.DataFrame)-> pd.DataFrame:
    paths = []
    data = []
    patient_id, study_id = dataframe["subject_id"], dataframe["study_id"]
    with open(file_path, "r") as f:
        paths = f.readlines()
    for path in paths:
        p = path[:-1]
        path = p.split("/")
        patient_id = path[2][1:]
        study_id = path[3][1:]
        dicom_id = path[4].rstrip(".jpg")
        data.append((patient_id, study_id, dicom_id, p))

    df_paths = pd.DataFrame(
        data, columns=["subject_id", "study_id", "dicom_id", "Path"]
    )
    df_paths["subject_id"] = df_paths["subject_id"].astype("int32")
    df_paths["study_id"] = df_paths["study_id"].astype("int32")
    dataframe = dataframe.reset_index(drop=True)
    print(len(dataframe))
    dataframe["subject_id"] = dataframe["subject_id"].astype("int32")
    dataframe["study_id"] = dataframe["study_id"].astype("int32")
    merge_file_path_dataset = dataframe.merge(
        df_paths, on=["subject_id", "study_id"], how="inner"
    )
    print(len(merge_file_path_dataset))
    merge_file_path_dataset.drop_duplicates(
        subset=["subject_id", "study_id"], inplace=True
    )
    print(len(merge_file_path_dataset))

    return merge_file_path_dataset


# Select the single subject_id per patient which has most positive disease
def sampling_datasets(training_dataset: pd.DataFrame)-> pd.DataFrame:
    training_dataset = training_dataset.groupby("subject_id", group_keys=False).apply(
        select_most_positive_sample
    )
    training_dataset.drop(columns=["positive_count"], inplace=True, errors="ignore")
    training_dataset.reset_index(drop=True)

    return training_dataset


def add_lung_mask_dataset(dataset: pd.DataFrame)-> pd.DataFrame:
    file_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/MASK-MIMIC-CXR-JPG.csv"
    )
    mask_df = pd.read_csv(file_path)
    mask_df = mask_df[["dicom_id", "Left Lung", "Right Lung", "Heart"]]
    merge_mask_dataset = pd.merge(dataset, mask_df, how="inner", on="dicom_id")
    print(len(merge_mask_dataset))
    merge_mask_dataset.drop_duplicates(subset=["subject_id"], inplace=True)

    return merge_mask_dataset


# Merge the data with demographic data
def merge_dataframe(training_data: pd.DataFrame, demographic_data: pd.DataFrame)-> pd.DataFrame:
    path = training_data["Path"]
    patientid = []
    for i in path:
        id = i.split(sep="/")[2]
        id = id.replace("patient", "")
        patientid.append(float(id))

    temp_patient = pd.DataFrame(patientid, columns=["patient_id"])
    training_data = training_data.reset_index(drop=True)
    training_data["subject_id"] = temp_patient["patient_id"]
    training_data_merge = training_data.merge(demographic_data, on="subject_id")
    return training_data_merge


def add_metadata(dataset: pd.DataFrame, metadata_path: Union[list, str, os.path])-> pd.DataFrame:
    meta_data = pd.read_csv(metadata_path)
    meta_data = meta_data[["subject_id", "study_id", "ViewPosition"]]
    meta_data = meta_data.reset_index(drop=True)
    dataset = dataset.reset_index(drop=True)
    sampling_total_dataset = pd.merge(
        dataset, meta_data, how="inner", on=["subject_id", "study_id"]
    )
    sampling_total_dataset = sampling_total_dataset.drop_duplicates(
        ["subject_id", "study_id"]
    )

    return sampling_total_dataset


def cleaning_datasets(traning_dataset: pd.DataFrame, is_chexpert: bool =True)-> pd.DataFrame:
    traning_dataset.drop(
        ["Pleural Other", "Fracture", "Support Devices"], axis=1, inplace=True
    )
    traning_dataset[
        [
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
    ] = (
        traning_dataset[
            [
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
        ].fillna(0.0)
        == 1.0
    ).astype(
        int
    )  # In The limits of fair medical imaging paper they treat uncertain label as negative and fill NA with 0.

    traning_dataset.loc[traning_dataset["race"].str.startswith("WHITE"), "race"] = "WHITE"
    traning_dataset.loc[traning_dataset["race"].str.startswith("BLACK"), "race"] = "BLACK"
    traning_dataset.loc[traning_dataset["race"].str.startswith("ASIAN"), "race"] = "ASIAN"
    
    # Select only Frontal View
    if is_chexpert:
        traning_dataset = traning_dataset[
            traning_dataset["Frontal/Lateral"] == "Frontal"
        ]
    else:
        traning_dataset = traning_dataset[
            traning_dataset.ViewPosition.isin(["AP", "PA"])
        ]

    return traning_dataset
