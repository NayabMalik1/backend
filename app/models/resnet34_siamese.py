import torch
import torch.nn as nn
from torchvision import models


class ResNet34Embedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        backbone = models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None
        )

        old_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        if pretrained:
            with torch.no_grad():
                backbone.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.embedding_head(x)
        return x


class SiameseResNet34(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        pretrained: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.encoder = ResNet34Embedding(
            embed_dim=embed_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return e1, e2