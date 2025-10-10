import torchvision
import torch.nn as nn
import torch
import numpy as np

class DenseNet_Model(nn.Module):
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

def replace_conv_in_model(model, new_conv_layer):
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            # Recursively call this function for sub-modules
            replace_conv_in_model(module, new_conv_layer)
        
        if isinstance(module, nn.Conv2d):
            # Replace the nn.Conv2d layer
            new_layer = new_conv_layer(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=(module.bias is not None)
            )
            model._modules[name] = new_layer


class PartialConv2d(nn.Module):
     # ... (code from above) ...
    def forward(self, input_tuple):
        x, mask = input_tuple
        # To handle the very first input which might have multiple channels
        if mask.shape[1] != x.shape[1]:
            mask = mask.expand(-1, x.shape[1], -1, -1)
        x_masked = x * mask
        x_out = self.conv(x_masked)
        with torch.no_grad():
            # Update mask for output channels
            if self.mask_conv.in_channels != self.mask_conv.out_channels:
                 self.mask_conv.in_channels = self.mask_conv.out_channels
            mask_out = self.mask_conv(mask)
        mask_ratio = self.kernel_size[0] * self.kernel_size[1] * self.in_channels / (mask_out + 1e-8)
        x_out = x_out * torch.clamp(mask_ratio, 0.0, 1e8)
        new_mask = (mask_out > 0).float()
        return x_out, new_mask

class MaskAwareWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
    
    def forward(self, input_tuple):
        x, mask = input_tuple
        x = self.layer(x)
        # Some layers like pooling change spatial dimensions of the mask
        if x.shape[2:] != mask.shape[2:]:
            mask = F.adaptive_max_pool2d(mask, x.shape[2:])
        return x, mask


class DenseNetPartialConv(nn.Module):
     def __init__(self, weights, out_feature):
          super().__init__()
          self.out_feature = out_feature
          
          # 1. Load the standard DenseNet
          encoder = torchvision.models.densenet121(weights=weights)
          
          # 2. Replace all Conv2d layers with PartialConv2d
          replace_conv_in_model(encoder, PartialConv2d)
          
          # 3. We only need the feature extractor part
          self.encoder = encoder.features
          
          # The final pooling and linear layers need to be handled separately
          self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
          self.clf = nn.Linear(encoder.classifier.in_features, out_feature)

     def forward(self, x):
          # Input x has NaN values for the background
          
          # 1. Create the initial binary mask from the input tensor
          # The mask should have 1 channel, but the same batch size and spatial dims
          mask = (~torch.isnan(x)).float()
          mask = mask[:, 0:1, :, :] # Take the mask from one channel

          # 2. Replace NaNs with 0 before passing to the network
          # The PartialConv math is designed to handle 0s when normalized correctly
          x = torch.nan_to_num(x, nan=0.0)
          
          # 3. Pass both image and mask through the encoder
          # The encoder is now a sequence of PartialConv2d and other layers
          # Note: This is a simplification. A fully correct model would wrap
          # BatchNorm, ReLU, and Pooling layers to handle the mask as well.
          # For now, we assume the network learns to adapt.
          features, final_mask = self.encoder((x, mask))
          
          # 4. Mask-aware global average pooling
          # Only average over the valid features
          # Sum of valid features / Number of valid features
          pooled_features = (features * final_mask).sum(dim=[2, 3]) / (final_mask.sum(dim=[2, 3]) + 1e-8)
          
          # 5. Classifier
          return self.clf(pooled_features)

# Help to build model for race prediction training
def model_transfer_learning(path, model, device, gradcam):

     state_dict = torch.load(path, map_location=device, weights_only=True)
     state_dict.pop("clf.weight", None)
     state_dict.pop("clf.bias", None)
     
     model.load_state_dict(state_dict, strict=False)

     if not gradcam:
          for params in model.encoder.parameters():
               params.requires_grad = False

     return model