import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from app.models.resnet34_siamese import SiameseResNet34
from app.training.pair_dataset import PairDataset
from app.training.loss_functions import ContrastiveLoss


TRAIN_DIR = "data/train_images"
MODEL_DIR = "trained_models"
MODEL_PATH = os.path.join(MODEL_DIR, "resnet34_best.pth")
HISTORY_PATH = os.path.join(MODEL_DIR, "train_history.json")

TRAIN_FAMILIES = ["benign", "banking", "smsware"]

IMAGE_SIZE = 224
PAIRS_PER_EPOCH = 1000
EPOCHS = 30
LR = 0.00005
TRAIN_BATCH_SIZE = 2
DEVICE = "cpu"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def train():
    ensure_dir(MODEL_DIR)

    dataset = PairDataset(
        root_dir=TRAIN_DIR,
        families=TRAIN_FAMILIES,
        image_size=IMAGE_SIZE,
        pairs_per_epoch=PAIRS_PER_EPOCH,
    )

    loader = DataLoader(
        dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    model = SiameseResNet34(
        embed_dim=128,
        pretrained=True,
        freeze_backbone=False
    ).to(DEVICE)

    # Freeze most of the backbone, train only the deeper part
    for name, param in model.encoder.feature_extractor.named_parameters():
        param.requires_grad = False
        if "7" in name:
            param.requires_grad = True

    # embedding head ہمیشہ trainable رہے
    for param in model.encoder.embedding_head.parameters():
        param.requires_grad = True

    criterion = ContrastiveLoss(margin=1.5)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR
    )

    best_loss = float("inf")
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for img1, img2, labels in loader:
            img1 = img1.to(DEVICE)
            img2 = img2.to(DEVICE)
            labels = labels.to(DEVICE).float()

            optimizer.zero_grad()

            emb1, emb2 = model(img1, img2)
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        history.append({"epoch": epoch + 1, "loss": epoch_loss})

        print(f"Epoch [{epoch + 1}/{EPOCHS}] Loss: {epoch_loss:.4f}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Best model saved at epoch {epoch + 1}")

    with open(HISTORY_PATH, "w") as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete.")
    print("Best model path:", MODEL_PATH)


if __name__ == "__main__":
    train()