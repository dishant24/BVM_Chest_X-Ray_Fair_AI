import pandas as pd

from typing import Union
import os



def select_most_positive_sample(group: pd.DataFrame) -> pd.Series:
    """
    Selects the sample (row) within a given patient or study group that has the highest number 
    of positive disease findings across multiple pathology columns.

    This function calculates the total number of positive indications (values of 1) across 
    several predefined disease-related columns for each record in the provided DataFrame group. 
    It then returns the row with the maximum count of positive diseases. If no positive findings 
    are present in the group, a random sample (row) from the group is returned instead.

    Parameters
    ----------
    group : pd.DataFrame
        A subset of the dataset representing a single patient or study group.
        It must contain the specified disease columns with binary indicators (0 or 1).

    Returns
    -------
    pd.Series
        The row corresponding to the sample with the highest number of positive disease findings.
        If no positive findings are found, a random row from the group is returned.

    """

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
    """
    Splits a DataFrame into groups based on unique values of a specified column and
    returns a dictionary where each key is a group name and the corresponding value 
    is the subset DataFrame for that group.

    Parameters
    ----------
    data : pd.DataFrame
        The input DataFrame to be grouped.
    group_column_name : str
        The name of the column on whose unique values the grouping is based.

    Returns
    -------
    dict
        A dictionary with keys as unique group names and values as DataFrames corresponding to each group.
    """
    groups_data = {}
    for name, group_data in data.groupby(group_column_name):
        groups_data[name] = group_data
    return groups_data


def add_demographic_data(training_data: pd.DataFrame, demographic_data: pd.DataFrame)-> pd.DataFrame:
    """
    Loads training and demographic data from compressed CSV files, then merges demographic 
    race information into the training dataset based on unique 'subject_id'. Drops rows 
    where race information is missing after the merge.

    Parameters
    ----------
    training_data : str
        File path to the training data CSV file (gzip compressed).
    demographic_data : str
        File path to the demographic data CSV file (gzip compressed).

    Returns
    -------
    pd.DataFrame
        The merged dataset including demographic race information, with missing race rows removed.
    """
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
    """
    Reads file paths from a given file or list, extracts 'subject_id', 'study_id', and 'dicom_id' 
    from the file path structure, then merges this information into the provided dataframe 
    based on matching 'subject_id' and 'study_id'.

    Parameters
    ----------
    file_path : Union[list, str, os.PathLike]
        File containing paths or a list of paths with format expected to contain subject and study IDs.
    dataframe : pd.DataFrame
        DataFrame containing 'subject_id' and 'study_id' to merge with extracted path info.

    Returns
    -------
    pd.DataFrame
        DataFrame enriched with 'dicom_id' and the full file path.
    """
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
    """
    Samples the training dataset by grouping on 'subject_id' and selecting the sample with the 
    highest number of positive disease findings per group using the function `select_most_positive_sample`.
    Drops the helper column 'positive_count' if present and resets the index before returning.

    Parameters
    ----------
    training_dataset : pd.DataFrame
        The input training dataset containing multiple samples per subject.

    Returns
    -------
    pd.DataFrame
        A sampled dataset with one representative record per subject having the most positive findings.
    """
    training_dataset = training_dataset.groupby("subject_id", group_keys=False).apply(
        select_most_positive_sample
    )
    training_dataset.drop(columns=["positive_count"], inplace=True, errors="ignore")
    training_dataset.reset_index(drop=True)

    return training_dataset


def add_lung_mask_mimic_dataset(dataset: pd.DataFrame)-> pd.DataFrame:
    """
    Adds lung mask segmentation data to the MIMIC dataset by merging with a CSV file containing mask metrics.
    The merge is done on 'dicom_id'. Duplicate entries based on 'subject_id' are dropped after merging.

    Parameters
    ----------
    dataset : pd.DataFrame
        The original MIMIC dataset to which lung mask data will be added.

    Returns
    -------
    pd.DataFrame
        The merged dataset enriched with lung mask measurements, with duplicates dropped on 'subject_id'.
    """
    file_path = (
        "/deep_learning/output/Sutariya/main/mimic/dataset/MASK-MIMIC-CXR-JPG.csv"
    )
    mask_df = pd.read_csv(file_path)
    mask_df = mask_df[["dicom_id", 'Dice RCA (Mean)', 'Left Lung', 'Right Lung', 'Heart']]
    merge_mask_dataset = pd.merge(dataset, mask_df, how="inner", on="dicom_id")
    print(len(merge_mask_dataset))
    merge_mask_dataset.drop_duplicates(subset=["subject_id"], inplace=True)

    return merge_mask_dataset

def add_lung_mask_chexpert_dataset(dataset: pd.DataFrame)-> pd.DataFrame:
    """
    Adds lung mask segmentation data to the CheXpert dataset by merging with a CSV file containing mask metrics.
    Path information is prefixed with the dataset folder before merging on 'Path'.

    Parameters
    ----------
    dataset : pd.DataFrame
        The original CheXpert dataset to which lung mask data will be added.

    Returns
    -------
    pd.DataFrame
        The merged dataset enriched with lung mask measurements.
    """
    file_path = (
        "/deep_learning/output/Sutariya/main/chexpert/dataset/CheXpert.csv"
    )
    mask_df = pd.read_csv(file_path)
    mask_df = mask_df[["Path", 'Dice RCA (Mean)', 'Left Lung', 'Right Lung', 'Heart']]
    mask_df['Path'] = 'CheXpert-v1.0-small/' + mask_df['Path']
    merge_mask_dataset = pd.merge(dataset, mask_df, how="inner", on="Path")
    print(len(merge_mask_dataset))

    return merge_mask_dataset


def merge_dataframe(training_data: pd.DataFrame, demographic_data: pd.DataFrame)-> pd.DataFrame:
    """
    Extracts patient IDs from the 'Path' column in the training dataset, converts them to floats,
    assigns these IDs as a new 'subject_id' column, and then merges the training dataset with
    demographic data on 'subject_id'.

    Parameters
    ----------
    training_data : pd.DataFrame
        The training dataset containing a 'Path' column with patient ID embedded in the path string.
    demographic_data : pd.DataFrame
        The demographic dataset containing patient information, including 'subject_id'.

    Returns
    -------
    pd.DataFrame
        The merged dataset combining training data and demographic information based on patient ID.
    """
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
    """
    Reads metadata from a CSV file, extracts relevant columns ('subject_id', 'study_id', 'ViewPosition'),
    and merges this metadata with the input dataset. Removes duplicate entries based on 'subject_id' and 'study_id'.

    Parameters
    ----------
    dataset : pd.DataFrame
        The primary dataset to be enriched with metadata.
    metadata_path : Union[list, str, os.PathLike]
        File path (or list) to the metadata CSV file containing 'subject_id', 'study_id', and 'ViewPosition'.

    Returns
    -------
    pd.DataFrame
        Dataset merged with metadata.
    """
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
    """
    Cleans the training dataset by dropping irrelevant columns, converting uncertain labels to negative
    by filling NA with 0, filtering for frontal views if CheXpert dataset, and harmonizing race/ethnicity labels.
    Eliminates entries with unknown or multiple races and restricts to AP or PA view positions.

    Parameters
    ----------
    training_dataset : pd.DataFrame
        The dataset to be cleaned and filtered.
    is_chexpert : bool, optional
        Flag indicating whether the dataset is from CheXpert (default is True).

    Returns
    -------
    pd.DataFrame
        Cleaned and filtered dataset with consistent labeling and race/ethnicity categories.
    """
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

    # Select only Frontal View
    if is_chexpert:
        traning_dataset = traning_dataset[
            traning_dataset["Frontal/Lateral"] == "Frontal"
        ]
        hispanic_data_df = traning_dataset[
                            (traning_dataset['ethnicity'] == 'Hispanic/Latino')
                            & ~(traning_dataset['race'].isin(['White', 'Black', 'Asian']))
                        ]
        hispanic_data_df['race'] = 'Hispanic'
        asian_white_black_data_df = traning_dataset[(traning_dataset['ethnicity'] == 'Non-Hispanic/Non-Latino') &
                                    (traning_dataset['race'].isin(['White', 'Black', 'Asian']))]
        traning_dataset = pd.concat([hispanic_data_df, asian_white_black_data_df], axis=0)
    else:
        traning_dataset.loc[traning_dataset["race"].str.startswith("WHITE"), "race"] = "White"
        traning_dataset.loc[traning_dataset["race"].str.startswith("BLACK"), "race"] = "Black"
        traning_dataset.loc[traning_dataset["race"].str.startswith("ASIAN"), "race"] = "Asian"

        traning_dataset.loc[traning_dataset.race.isin([
                "HISPANIC OR LATINO",
                "HISPANIC/LATINO - PUERTO RICAN",
                "HISPANIC/LATINO - GUATEMALAN",
                "HISPANIC/LATINO - HONDURAN",
                "HISPANIC/LATINO - COLUMBIAN",
                "HISPANIC/LATINO - DOMINICAN",
                "HISPANIC/LATINO - SALVADORAN",
                "HISPANIC/LATINO - CENTRAL AMERICAN",
                "HISPANIC/LATINO - CUBAN",
                "HISPANIC/LATINO - MEXICAN",
                "PORTUGUESE",
                "SOUTH AMERICAN"]),
                "race"] = "Hispanic"

        traning_dataset = traning_dataset[~traning_dataset['race'].isin([
                        "UNKNOWN",
                        "OTHER",
                        "UNABLE TO OBTAIN",
                        "PATIENT DECLINED TO ANSWER",
                        "MULTIPLE RACE/ETHNICITY"
                    ])]

        traning_dataset = traning_dataset[
            traning_dataset.ViewPosition.isin(["AP", "PA"])
        ]

    return traning_dataset
