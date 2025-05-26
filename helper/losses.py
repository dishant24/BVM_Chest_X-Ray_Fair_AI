import torch
import torch.nn as nn

class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    """
    Binary Cross Entropy loss with label smoothing.

    Args:
        smoothing (float): Label smoothing factor (default: 0.0).
    """

    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        assert 0.0 <= smoothing < 1.0, "Smoothing must be in [0, 1)"
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Smooth the labels: 1 -> 1 - ε, 0 -> ε
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return self.bce(logits, targets)