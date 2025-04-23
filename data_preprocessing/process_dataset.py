import pandas as pd

def select_most_positive_sample(group):

    disease_columns = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture'
    ]
    
    group['positive_count'] = group[disease_columns].sum(axis=1)
    
    positive_cases = group[group['positive_count'] > 0]
    
    if not positive_cases.empty:

        selected_sample = positive_cases.loc[positive_cases['positive_count'].idxmax()]
    else:
        selected_sample = group.sample(n=1).iloc[0]
    
    return selected_sample

def get_group_by_data(data, group_column_name):
    
    groups_data = {}
    for name, group_data in data.groupby(group_column_name):
        groups_data[name] = group_data
    return groups_data

# Merge the data with demographic data
def add_demographic_data(training_data, demographic_data):
     
     # Load the CSV files
     df_chexpert = pd.read_csv(training_data, compression='gzip')
     df_patients = pd.read_csv(demographic_data, compression='gzip')

     # Check for duplicate subject_id in patients dataset
     df_patients_unique = df_patients[['subject_id', 'race']].drop_duplicates(subset=['subject_id'])

     # Verify uniqueness
     assert df_patients_unique.duplicated(subset=['subject_id']).sum() == 0, "Duplicate subject_id found in patients dataset"

     # Merge using 'subject_id' 
     df_merged = df_chexpert.merge(df_patients_unique, on="subject_id", how="left")
     df_cleaned = df_merged.dropna(subset=['race'])

     return df_cleaned

def merge_file_path(file_path, dataframe):
     paths = []
     data = []
     patient_id, study_id = dataframe['subject_id'], dataframe['study_id']
     with open(file_path, 'r') as f:
          paths = f.readlines()
     for path in paths:
          path = path[:-1]
          patient_id = path[11:19]
          study_id = path[21:29]
          data.append((patient_id, study_id, path))

     df_paths = pd.DataFrame(data, columns=['subject_id', 'study_id', 'file_path'])
     df_paths['subject_id'] = df_paths['subject_id'].astype(str)
     df_paths['study_id'] = df_paths['study_id'].astype(str)
     dataframe['subject_id'] = dataframe['subject_id'].astype(str)
     dataframe['study_id'] = dataframe['study_id'].astype(str)

     merge_file_path_dataset = dataframe.merge(df_paths, on=['subject_id', 'study_id'], how='left')

     merge_file_path_dataset.drop_duplicates(subset=['subject_id', 'study_id'], inplace=True)

     return merge_file_path_dataset

# Select the single subject_id per patient which has most positive disease 
def sampling_datasets(training_dataset):

    training_dataset = training_dataset.groupby('subject_id', group_keys=False).apply(select_most_positive_sample)
    training_dataset.drop(columns=['positive_count'], inplace=True, errors='ignore')
    
    return training_dataset


# Merge the data with demographic data
def merge_dataframe(training_data, demographic_data):
    path = training_data['Path']
    patientid = []
    for i in path:
        id = i.split(sep='/')[2]
        id = id.replace("patient", "")
        patientid.append(float(id))

    temp_patient = pd.DataFrame(patientid,columns=['patient_id'])
    training_data = training_data.reset_index(drop=True)
    training_data['subject_id'] = temp_patient['patient_id']
    training_data_merge = training_data.merge(demographic_data, on='subject_id')
    return training_data_merge


def cleaning_datasets(traning_dataset):

    traning_dataset[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']] = (traning_dataset[['No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
    'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
    'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture',
    'Support Devices']].fillna(0.0) == 1.0).astype(int)  # In The limits of fair medical imaging paper they treat uncertain label as negative and fill NA with 0.

    #Select only Frontal View 
    traning_dataset = traning_dataset[traning_dataset['Frontal/Lateral'] == 'Frontal']

    return traning_dataset

