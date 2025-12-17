# Improving Generalization and Robustness of ChestX-ray AI Models: Preprocessing method to mitigate Racial/Ethnic Bias

This project focuses on improving the **fairness** and **robustness** of AI models used in **medical chest X-ray (CXR) diagnosis**. The primary objective is mitigate **racial demographic biases** that can arise in deep learning models trained on medical imaging CXR datasets.

---

## 🚀 Objective

Build processing method for an AI model that is:
- **Accurate**: High performance on core classification tasks (e.g., disease detection)
- **Fair**: Ensures consistent performance across different racial, gender, and age groups
- **Robust**: Maintains reliability under various distribution shifts and real-world scenarios

---

## 📦 Installation
```
git clone https://github.com/dishant24/BVM_Chest_X-Ray_Fair_AI.git
cd BVM_Chest_X_Ray_Fair_AI
pip install -r requirements.txt
```

## ⚙️ Methods and Approach

1. **Model Training**:
   - Reproduce initial experiments of reference papers and build DenseNet model
   - Train model on diagnosis labels, use same embedding to train model on race (transfer learning for race classification)
   - Evaluate model performance across demographic slices

3. **Bias Mitigation**:
   - Use lung segmentation so model focuses on relevant clinical features 
   - Use CLAHE histogram equalization method to reduce noise and improve contrast


## 🗂️ Dataset(s)

This project uses publicly available chest X-ray datasets such as:
- **MIMIC-CXR-JPG**
- **CheXpert**

These datasets include metadata for demographic attributes.

---

## ⚙️ How to Run Experiments

This section explains how to use this codebase to run all experiments exactly as described, including training, testing, external dataset evaluation, group-wise analysis, plotting, and result table generation.

### 1. Prepare Dataset Paths

Configure dataset paths before running experiments.

For **MIMIC** dataset:
You can dowanload this dataset from here: https://physionet.org/content/mimic-cxr-jpg/2.0.0/

meta_file_path = mimic-cxr-2.0.0-metadata.csv.gz
demographic_data_path = admissions.csv.gz
all_dataset_path = mimic-cxr-2.0.0-chexpert.csv.gz

- If you running `mimic_cxr_model.py` file it take original data and code apply cleaning and splitting datasets automatically and save files.


For **external out-of-distribution (OOD) tests** using CheXpert:
You can dowanload this dataset from here: https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2

demographic_data_path = demographics_CXP.csv
train_dataset_path = chexpert/train.csv


---

### 2.Training and Testing

Use the provided `mimic_cxr_model.py` script and relevant flags to run training and testing:

#### a) Train Diagnostic Model on MIMIC
```
python mimic_cxr_model.py --task diagnostic --random_state 100 --epoch 30 --dataset mimic --training
```

#### b) Test Diagnostic Model on MIMIC
```
python mimic_cxr_model.py --task diagnostic --random_state 100 --epoch 30 --dataset mimic
```

#### c) Train Race Classification Model on MIMIC
- You just need to change the task to race
```
python mimic_cxr_model.py --task race --random_state 100 --epoch 30 --dataset mimic --training
```

#### d) Test Race Classification Model on MIMIC
```
python mimic_cxr_model.py --task race --random_state 100 --epoch 30 --dataset mimic
```
---

### 3. External Dataset Evaluation

Evaluate models on the external CheXpert dataset:

#### a) Diagnostic External Dataset Evaluation
- Add external_ood_test flag
```
python mimic_cxr_model.py --task diagnostic --random_state 100 --epoch 30 --dataset mimic --external_ood_test
```

#### b) Race External Dataset Evaluation
```
python mimic_cxr_model.py --task race --random_state 100 --epoch 30 --dataset mimic --external_ood_test
```

---

### 5. Generate AUROC Plots

Create comparative AUROC plots for all models and preprocessing methods. To include external data, add `--external_ood_test` flag.
```
python helper/generate_plot.py --task diagnostic --random_state 100 --dataset mimic
```

---

### 6. Generate Result Tables

Produce result summary tables comparing models and preprocessing effects. Add `--external_ood_test` for including external dataset results.
```
python helper/generate_result_tables.py --random_state 100 --dataset mimic
```

---

## Notes

- Make sure all dataset path variables are correctly set before running.
- Training flags are mandatory to initiate model training; omit to run inference only.
- The `multi_label` flag is set internally based on the task.
- Adjust `random_state` and epochs as needed for reproducibility and training length.
- Scripts assume running in an environment with required dependencies and access to GPU for best performance.

This guide enables exact replication of the experiments for fair, robust, and accurate chest X-ray AI model development.


