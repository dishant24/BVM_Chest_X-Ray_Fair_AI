import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Implements label smoothing loss for both multiclass and multilabel classification tasks.

    Label smoothing helps regularize the model by preventing it from becoming too confident about a single class,
    which can improve generalization.

    Parameters
    ----------
    smoothing : float, optional
        Smoothing factor between [0, 1) to control the amount of label smoothing (default 0.0).
    mode : str, optional
        Either 'multiclass' or 'multilabel', specifying the classification task type.

    Methods
    -------
    forward(x, target)
        Computes the smoothed loss given model outputs `x` and ground truth `target`.
    """
     
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

