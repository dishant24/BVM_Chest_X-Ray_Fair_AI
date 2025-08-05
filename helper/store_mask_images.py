
import os
import cv2
from tqdm import tqdm    
from concurrent.futures import ThreadPoolExecutor 
from functools import partial
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_preprocessing.process_dataset import (
    add_demographic_data,
    add_lung_mask_dataset,
    add_metadata,
    cleaning_datasets,
    get_group_by_data,
    merge_file_path_and_add_dicom_id,
    sampling_datasets,
)

from datasets.data import ApplyLungMask 
import wandb

def process_row(row, base_dir, save_dir, masker_params):
     try:
         image_path = os.path.join(base_dir, row["Path"])
         save_path = os.path.join(save_dir, row["Path"])
         if os.path.exists(save_path):
            return
         # Load and resize image
         image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
         if image is None:
             print(f"Warning: could not load image {image_path}")
             return

         image = cv2.resize(image, masker_params["image_shape"], interpolation=cv2.INTER_AREA)

         # Reconstruct masker object inside each process
         masker = ApplyLungMask(**masker_params)
         mask = masker.compute_combined_mask(row["Left Lung"], row["Right Lung"], row["Heart"])
         masked_image = cv2.bitwise_and(image, image, mask=mask)

         # Save
         print(f"New image save at: {save_path}")
         os.makedirs(os.path.dirname(save_path), exist_ok=True)
         cv2.imwrite(save_path, masked_image)
     except Exception as e:
         print(f"Error processing {row['Path']}: {e}")

def apply_and_save_masked_images_parallel(df, base_dir, save_dir, masker):
    masker_params = {
        "margin_radius": masker.margin_radius,
        "original_shape": masker.original_shape,
        "image_shape": masker.image_shape,
    }
    with ThreadPoolExecutor() as executor:  # ← change here
        tasks = []
        for _, row in df.iterrows():
            tasks.append(executor.submit(process_row, row, base_dir, save_dir, masker_params))
        for future in tqdm(tasks):
            future.result()


if __name__ == "__main__":

     base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
     save_dir = '/deep_learning/output/Sutariya/MIMIC-CXR-MASK/'

     wandb.init(
        project="store_mask_images"
    )

     train_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/train_clean_dataset.csv"
     test_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
     valid_output_path ="/deep_learning/output/Sutariya/main/mimic/dataset/validation_clean_dataset.csv"

     train_df, test_df, valid_df = pd.read_csv(train_output_path), pd.read_csv(test_output_path), pd.read_csv(valid_output_path)

     merge_df = pd.concat([train_df, test_df, valid_df])

     merge_df = add_lung_mask_dataset(merge_df)

     masker = ApplyLungMask(
     margin_radius=60,
     original_shape=(1024, 1024),
     image_shape=(224, 224)
     )
     apply_and_save_masked_images_parallel(merge_df, base_dir, save_dir, masker)