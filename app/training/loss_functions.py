import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    label = 1 -> same class
    label = 0 -> different class
    """
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, label: torch.Tensor):
        dist = F.pairwise_distance(emb1, emb2)

        positive_loss = label * torch.pow(dist, 2)
        negative_loss = (1 - label) * torch.pow(torch.clamp(self.margin - dist, min=0.0), 2)

        loss = positive_loss + negative_loss

        return loss.mean()   # ✅ only one mean