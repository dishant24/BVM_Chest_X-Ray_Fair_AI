import torchvision
import torch.nn as nn
import torch
import numpy as np

class DenseNet_Model(nn.Module):
     """
    DenseNet121-based neural network adapted for medical imaging classification tasks.

    Parameters
    ----------
    weights : torchvision.models.DenseNet121_Weights or None
        Pretrained weights to initialize the DenseNet121 encoder; None for random initialization.
    out_feature : int
        Number of output features/classes for the classification head.

    Attributes
    ----------
    encoder : nn.Module
        DenseNet121 model used as a feature extractor.
    relu : nn.ReLU
        ReLU activation applied after the encoder output.
    clf : nn.Linear
        Linear layer mapping from encoder features to classification outputs.
    """
     def __init__(self, weights, out_feature):
          super().__init__()
          self.weight = weights
          self.out_feature = out_feature
          self.encoder = torchvision.models.densenet121(weights=weights) # Adapt the architecture to initial paper: The limits of fair medical imaging and almost all other papers
          self.relu = nn.ReLU()
          self.clf = nn.Linear(1000, out_feature)

     def encode(self, x):
          return self.encoder(x)

     def forward(self, x):
          z = self.encode(x)
          z = self.relu(z)
          return self.clf(z)

     def predict_proba(self, x):
          return torch.sigmoid(self(x))



# Help to build model for race prediction training
def model_transfer_learning(path:str, model:torch.nn.Module, device:torch.device, gradcam:bool):
     """
    Loads pretrained weights into a model for transfer learning, optionally freezing 
    the encoder parameters for feature extraction, typically for Grad-CAM visualization.

    Parameters
    ----------
    path : str
        Path to the pretrained model weights file.
    model : torch.nn.Module
        The model into which weights will be loaded.
    device : torch.device
        Device on which to load the weights (CPU/GPU).
    gradcam : bool
        If False, freezes the encoder layers by disabling gradient updates.

    Returns
    -------
    torch.nn.Module
        The model with loaded weights and optionally frozen encoder.
    """

     state_dict = torch.load(path, map_location=device, weights_only=True)
     state_dict.pop("clf.weight", None)
     state_dict.pop("clf.bias", None)
     
     model.load_state_dict(state_dict, strict=False)

     if not gradcam:
          for params in model.encoder.parameters():
               params.requires_grad = False

     return model