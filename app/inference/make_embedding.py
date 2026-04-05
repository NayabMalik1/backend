import os
from typing import Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from app.models.resnet34_siamese import SiameseResNet34
from app.settings import IMAGE_SIZE, EMBED_DIM, DEVICE, MODEL_PATH

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

def load_model(model_path: Optional[str] = None) -> SiameseResNet34:
    model_path = model_path or MODEL_PATH

    model = SiameseResNet34(
        embed_dim=EMBED_DIM,
        pretrained=False,
        freeze_backbone=False,
    ).to(DEVICE)

    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model

def image_to_tensor(image_path: str) -> torch.Tensor:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("L")
    tensor = _transform(img).unsqueeze(0)  # [1,1,H,W]
    return tensor

@torch.no_grad()
def make_embedding(image_path: str, model: SiameseResNet34) -> torch.Tensor:
    x = image_to_tensor(image_path).to(DEVICE)
    emb = model.forward_once(x)
    emb = F.normalize(emb, p=2, dim=1)
    return emb.squeeze(0).cpu()
