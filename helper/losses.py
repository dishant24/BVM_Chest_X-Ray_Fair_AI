import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0, mode='multiclass'):
        
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        assert mode in ['multiclass', 'multilabel'], "Mode must be 'multiclass' or 'multilabel'"
        self.smoothing = smoothing
        self.mode = mode

    def forward(self, x, target):
        if self.mode == 'multiclass':
            loss = nn.CrossEntropyLoss(label_smoothing=self.smoothing)
            return loss(x, target)

        elif self.mode == 'multilabel':
            with torch.no_grad():
                smooth_target = target * (1 - self.smoothing) + 0.5 * self.smoothing
            return F.binary_cross_entropy_with_logits(x, smooth_target)

