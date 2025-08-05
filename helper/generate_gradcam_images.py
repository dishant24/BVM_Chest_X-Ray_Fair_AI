import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, Lambda
from PIL import Image
import wandb
from torchcam.methods import GradCAMpp
import argparse
import pandas as pd
import cv2
from sklearn.preprocessing import LabelEncoder
import math
import torch.nn as nn
import random
import matplotlib
from tqdm import tqdm
from datasets.dataloader import prepare_dataloaders
from models.build_model import DenseNet_Model, model_transfer_learning

def overlay_heatmap_on_image(orig_img, activation_map, alpha=0.4):
     # Normalize activation map
     activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
     
     # Resize CAM to match image size
     heatmap = Image.fromarray(np.uint8(activation_map * 255)).resize(orig_img.size, Image.BILINEAR)
     heatmap = np.array(heatmap) / 255.0
     
     # Apply colormap using matplotlib
     cmap = matplotlib.colormaps['jet']
     heatmap_colored = cmap(heatmap)[:, :, :3]
     
     # Convert original image to float
     img_arr = np.array(orig_img) / 255.0
     img_arr = np.stack([img_arr] * 3, axis=-1)
     
     # Blend image and heatmap
     blended = (1 - alpha) * img_arr + alpha * heatmap_colored
     blended = np.clip(blended, 0, 1)
     
     return blended

def generate_cam_images(dataframe, weight_path, class_names, task, clahe, masked):
     device = 'cuda' if torch.cuda.is_available() else 'cpu'
     create_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

     print(weight_path)
     if task == 'diagnostic':
          model = DenseNet_Model(weights=None, out_feature=11)
          weights = torch.load(weight_path, map_location=device, weights_only=True)
          model.load_state_dict(weights)
     else:
          base_model = DenseNet_Model(weights=None, out_feature=4)
          model = model_transfer_learning(weight_path, base_model, device, True)

     model.eval().to(device)
     cam_extractor = GradCAMpp(model)

     transform = Compose([
          Resize((224, 224)),
          ToTensor(),
          Lambda(lambda i: i.repeat(3, 1, 1) if i.shape[0] == 1 else i),
          Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])

     loader = prepare_dataloaders(
                    images_path= dataframe["Path"],
                    labels= dataframe[class_names].values,
                    dataframe= dataframe,
                    masked= masked,
                    clahe= clahe,
                    reweight= False,
                    transform = transform,
                    base_dir= "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/",
                    shuffle=True,
                    is_multilabel=True,
                    external_ood_test = False
                )

     num_classes = len(class_names)
     heatmaps = {i: [] for i in range(num_classes)}
     # images = {i: [] for i in range(num_classes)}
     for inputs, labels, _ in tqdm(loader, total=len(loader)):
          for img_tensor, label in zip(inputs, labels):
               img_tensor = img_tensor.unsqueeze(0).to(device)
               label = label.unsqueeze(0).to(device)
               with torch.set_grad_enabled(True):
                    output = model(img_tensor)
               if task == 'diagnostic':
                    probs = torch.sigmoid(output)[0]
               else:
                    probs = torch.softmax(output, dim=1)[0]
               for cls in range(num_classes):
                    activation_map = cam_extractor(class_idx=cls, scores=output)[0].cpu().squeeze().numpy()
                    heatmaps[cls].append(activation_map)

     # Average maps and overlays per class
     avg_heatmap_per_class = {}

     for class_idx in range(num_classes):
          maps = heatmaps[class_idx]
          if maps:
                    maps_stack = np.stack(maps, axis=0)
                    avg_saliency_maps = np.mean(maps_stack, axis=0)
                    avg_heatmap_per_class[class_idx] = avg_saliency_maps

     # Visualization
     rows = num_classes
     cols = 1
     fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows))

     for cls in range(num_classes):
          label_name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
          axes[cls].imshow(avg_heatmap_per_class.get(cls).astype(np.float32), cmap='jet')
          axes[cls].set_title(f"Avg Activation Map: {label_name}")
          axes[cls].axis('off')

     plt.tight_layout()
     wandb.log({"Average Grad-CAMs": wandb.Image(fig)})
     plt.close(fig)
     
     # fig, ax = plt.subplots(10, 2, figsize=(40, 30))
     # plot_idx = 0

     # for inputs, labels, _ in tqdm(loader, total=len(loader)):
     #      for img_tensor, label in zip(inputs, labels):
     #           img_tensor = img_tensor.unsqueeze(0).to(device)
     #           label = label.unsqueeze(0).to(device)

     #           with torch.set_grad_enabled(True):
     #                output = model(img_tensor)

     #           if task == 'diagnostic':
     #                probs = torch.sigmoid(output)[0]
     #           else:
     #                probs = torch.softmax(output, dim=1)[0]

     #           for cls in range(num_classes):
     #                activation_map = cam_extractor(class_idx=cls, scores=output)[0].cpu().squeeze().numpy()

     #                if plot_idx < 10:
     #                     original_img = img_tensor[0].cpu().numpy()
     #                     ax[plot_idx, 0].imshow(original_img, cmap='gray')
     #                     ax[plot_idx, 0].set_title("Original Image")
     #                     ax[plot_idx, 0].axis("off")

     #                     ax[plot_idx, 1].imshow(original_img, cmap='gray')
     #                     ax[plot_idx, 1].imshow(activation_map, cmap='jet', alpha=0.5)
     #                     ax[plot_idx, 1].set_title(f"Activation Map - Class {class_names[cls]}")
     #                     ax[plot_idx, 1].axis("off")

     #           plot_idx += 1
     #           if plot_idx >= 10:
     #                break
     #      if plot_idx >= 10:
     #           break


     # plt.tight_layout()
     # wandb.log({"Images of Clahe": wandb.Image(fig)})
     # plt.close(fig)


if __name__ == "__main__":
     parser = argparse.ArgumentParser(description="Testing Arguments")
     parser.add_argument('--random_state', type=int, default=100)
     parser.add_argument('--task', type=str, choices=["diagnostic", "race"], default="diagnostic")
     parser.add_argument('--dataset', type=str, default="mimic")
     parser.add_argument('--masked', action='store_true')
     parser.add_argument('--clahe', action='store_true')
     args = parser.parse_args()
     print(vars(args))
     torch.cuda.empty_cache()
     random_state = args.random_state
     task = args.task
     dataset = args.dataset
     masked = args.masked
     clahe = args.clahe
     name = f"model_baseline_{dataset}_{random_state}"
     wandb.init(
        project=f"{dataset}_preprocessing_{task}",
        name="Avarage_grade_cam")

     label_encoder = LabelEncoder()
     
     create_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

     if masked:
          weight_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/mask_preprocessing_{name}.pth"
          base_dir = "/deep_learning/output/Sutariya/MIMIC-CXR-MASK/"
     elif clahe:
          base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
          weight_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/clahe_preprocessing_{name}.pth"
     else:
          base_dir = "MIMIC-CXR/physionet.org/files/mimic-cxr-jpg/2.1.0/"
          weight_path = f"/deep_learning/output/Sutariya/main/mimic/checkpoints/{task}/{name}.pth"
          
     test_output_path = "/deep_learning/output/Sutariya/main/mimic/dataset/test_clean_dataset.csv"
     test_df = pd.read_csv(test_output_path)

     if task == 'race':
          test_df['race_encode'] = label_encoder.fit_transform(test_df['race'])
          labels = label_encoder.classes_
     else:
          labels = [
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
          "Pleural Effusion"]     

     generate_cam_images(test_df, weight_path, labels, task, clahe, masked)


