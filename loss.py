'''import torch
import torch.nn as nn

# Adversarial loss for GANs
class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with Logits

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)

# Multi-scale L1 Loss (if needed)
class MultiScaleL1Loss(nn.Module):
    def __init__(self):
        super(MultiScaleL1Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, predictions, targets):
        return self.criterion(predictions, targets)

# Instantiate loss functions
adversarial_loss = AdversarialLoss()
multi_scale_l1_loss = MultiScaleL1Loss()'''

#cnn 
import torch
import torch.nn as nn

# ── Binary Cross-Entropy Loss ───────────────────────────────
# Use this for pixel-wise binary segmentation
bce_loss = nn.BCELoss()

# ── Dice Loss ────────────────────────────────────────────────
# Complementary to BCE, focuses on overlap
class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions and targets are both (N, 1, H, W) with values in [0,1]
        intersection = (predictions * targets).sum(dim=(1,2,3))
        total = predictions.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
        dice_score = (2 * intersection + self.smooth) / (total + self.smooth)
        # Return 1 - mean Dice coefficient as a loss
        return 1 - dice_score.mean()

dice_loss = DiceLoss()

# ── Combined Loss (Optional) ─────────────────────────────────
# A common practice is to combine BCE + Dice for more stable training
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCELoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.bce_weight * self.bce(predictions, targets) + \
               self.dice_weight * self.dice(predictions, targets)

bce_dice_loss = BCEDiceLoss()
