import os
import random
from typing import Dict, List

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class PairDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        families: List[str],
        image_size: int = 224,
        pairs_per_epoch: int = 1000,
    ):
        self.root_dir = root_dir
        self.families = families
        self.pairs_per_epoch = pairs_per_epoch

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        self.family_to_images: Dict[str, List[str]] = {}

        for family in families:
            family_dir = os.path.join(root_dir, family)
            if not os.path.isdir(family_dir):
                continue

            files = []
            for f in os.listdir(family_dir):
                if f.lower().endswith(".png"):
                    full_path = os.path.join(family_dir, f)
                    try:
                        with Image.open(full_path) as img:
                            img.verify()
                        files.append(full_path)
                    except Exception:
                        continue

            if len(files) >= 2:
                self.family_to_images[family] = files

        self.valid_families = list(self.family_to_images.keys())

        if len(self.valid_families) < 2:
            raise ValueError("At least 2 valid families with >=2 images each are required.")

    def __len__(self):
        return self.pairs_per_epoch

    def _load_image(self, path: str) -> torch.Tensor:
        img = Image.open(path).convert("L")
        return self.transform(img)

    def __getitem__(self, index: int):
        same_class = random.randint(0, 1)

        if same_class == 1:
            family = random.choice(self.valid_families)
            img_paths = random.sample(self.family_to_images[family], 2)
            label = 1.0
        else:
            fam1, fam2 = random.sample(self.valid_families, 2)
            img_paths = [
                random.choice(self.family_to_images[fam1]),
                random.choice(self.family_to_images[fam2]),
            ]
            label = 0.0

        img1 = self._load_image(img_paths[0])
        img2 = self._load_image(img_paths[1])

        return img1, img2, torch.tensor(label, dtype=torch.float32)