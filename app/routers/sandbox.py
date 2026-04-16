import os
import shutil
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from app.models.resnet34_siamese import SiameseResNet34
from app.training.pair_dataset import PairDataset
from app.training.loss_functions import ContrastiveLoss
from app.settings import GRAY_IMAGES_DIR
import asyncio
import json
from fastapi.responses import StreamingResponse
from typing import Optional

router = APIRouter(prefix="/sandbox", tags=["Training Sandbox"])

# Sandbox directories (relative to backend root)
SANDBOX_DIR = "sandbox"
SANDBOX_TRAIN_DIR = os.path.join(SANDBOX_DIR, "data", "train")
SANDBOX_VAL_DIR = os.path.join(SANDBOX_DIR, "data", "val")
SANDBOX_MODELS_DIR = os.path.join(SANDBOX_DIR, "models")
SANDBOX_RESULTS_DIR = os.path.join(SANDBOX_DIR, "results")

# Ensure directories exist
os.makedirs(SANDBOX_TRAIN_DIR, exist_ok=True)
os.makedirs(SANDBOX_VAL_DIR, exist_ok=True)
os.makedirs(SANDBOX_MODELS_DIR, exist_ok=True)
os.makedirs(SANDBOX_RESULTS_DIR, exist_ok=True)

# Families (same as main)
FAMILIES = ["benign", "banking", "smsware", "adware", "riskware"]

# How many images to copy for sandbox (train and validation)
TRAIN_COUNT = 5   # per family
VAL_COUNT = 2     # per family

def ensure_sandbox_data():
    """Copy a small subset of grayscale images to sandbox data folders (if not already present)."""
    # Check if already copied (e.g., by checking one family's train folder)
    sample_path = os.path.join(SANDBOX_TRAIN_DIR, FAMILIES[0])
    if os.path.exists(sample_path) and len(os.listdir(sample_path)) >= TRAIN_COUNT:
        return  # already copied

    # Otherwise, copy fresh
    for family in FAMILIES:
        src_dir = os.path.join(GRAY_IMAGES_DIR, family)
        if not os.path.isdir(src_dir):
            print(f"Warning: {src_dir} not found, skipping {family}")
            continue

        # Get all PNG images, sorted
        images = [f for f in os.listdir(src_dir) if f.endswith(".png")]
        images.sort()
        if len(images) < TRAIN_COUNT + VAL_COUNT:
            print(f"Warning: Not enough images for {family}, need {TRAIN_COUNT+VAL_COUNT}, have {len(images)}")
            # Use what's available
            train_imgs = images[:TRAIN_COUNT]
            val_imgs = images[TRAIN_COUNT:TRAIN_COUNT+VAL_COUNT]
        else:
            train_imgs = images[:TRAIN_COUNT]
            val_imgs = images[TRAIN_COUNT:TRAIN_COUNT+VAL_COUNT]

        # Copy to train
        train_dst = os.path.join(SANDBOX_TRAIN_DIR, family)
        os.makedirs(train_dst, exist_ok=True)
        for img in train_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(train_dst, img))

        # Copy to val
        val_dst = os.path.join(SANDBOX_VAL_DIR, family)
        os.makedirs(val_dst, exist_ok=True)
        for img in val_imgs:
            shutil.copy(os.path.join(src_dir, img), os.path.join(val_dst, img))

    print("Sandbox data ready.")

class TrainRequest(BaseModel):
    batch_size: int = 4
    learning_rate: float = 0.0005
    epochs: int = 10
    loss_margin: float = 1.5

@router.post("/train")
async def sandbox_train(req: TrainRequest):
    # Ensure sandbox data exists
    ensure_sandbox_data()

    # Create dataset and dataloader from sandbox train folder
    train_dataset = PairDataset(
        root_dir=SANDBOX_TRAIN_DIR,
        families=FAMILIES,
        image_size=224,
        pairs_per_epoch=500  # smaller for speed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=req.batch_size,
        shuffle=True,
        num_workers=0
    )

    # Create a fresh model (not loading any pretrained weights)
    model = SiameseResNet34(
        embed_dim=128,
        pretrained=False,  # no pretrained weights to keep training fast
        freeze_backbone=False
    )
    # For speed, we can freeze most of the backbone except the last few layers
    # (similar to main training but we can be simpler)
    for name, param in model.encoder.feature_extractor.named_parameters():
        param.requires_grad = False
        if "7" in name:  # only train the last block
            param.requires_grad = True
    for param in model.encoder.embedding_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=req.learning_rate
    )
    criterion = ContrastiveLoss(margin=req.loss_margin)

    history = []  # list of {"epoch": e, "loss": l, "accuracy": a}

    for epoch in range(req.epochs):
        model.train()
        running_loss = 0.0
        correct_pairs = 0
        total_pairs = 0

        for img1, img2, labels in train_loader:
            # Move to device (CPU for simplicity; can be changed)
            img1 = img1
            img2 = img2
            labels = labels.float()

            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            emb1 = F.normalize(emb1, p=2, dim=1)
            emb2 = F.normalize(emb2, p=2, dim=1)

            loss = criterion(emb1, emb2, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute accuracy for this batch (simple threshold)
            distances = F.pairwise_distance(emb1, emb2)
            predictions = (distances < req.loss_margin).float()
            correct = (predictions == labels).sum().item()
            correct_pairs += correct
            total_pairs += len(labels)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = (correct_pairs / total_pairs) * 100.0
        history.append({"epoch": epoch + 1, "loss": round(epoch_loss, 4), "accuracy": round(epoch_acc, 2)})

        print(f"Sandbox Epoch [{epoch+1}/{req.epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # After training, evaluate on validation set (simple)
    val_dataset = PairDataset(
        root_dir=SANDBOX_VAL_DIR,
        families=FAMILIES,
        image_size=224,
        pairs_per_epoch=200  # small
    )
    val_loader = DataLoader(val_dataset, batch_size=req.batch_size, shuffle=False, num_workers=0)
    model.eval()
    correct_pairs = 0
    total_pairs = 0
    with torch.no_grad():
        for img1, img2, labels in val_loader:
            emb1, emb2 = model(img1, img2)
            distances = F.pairwise_distance(emb1, emb2)
            predictions = (distances < req.loss_margin).float()
            correct = (predictions == labels).sum().item()
            correct_pairs += correct
            total_pairs += len(labels)
    val_accuracy = (correct_pairs / total_pairs) * 100.0

    # Optionally, compute per-family accuracy on validation (more complex, can skip for now)
    # For simplicity, we just return overall validation accuracy.

    # Save the model (optional)
    model_path = os.path.join(SANDBOX_MODELS_DIR, f"student_model_epoch_{req.epochs}.pth")
    torch.save(model.state_dict(), model_path)

    # Save history to a JSON file
    history_path = os.path.join(SANDBOX_RESULTS_DIR, f"history_{req.epochs}_{req.learning_rate}.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return {
        "history": history,
        "final_loss": history[-1]["loss"] if history else 0,
        "val_accuracy": round(val_accuracy, 2),
        "message": f"Training completed on sandbox data. Validation accuracy: {val_accuracy:.2f}%"
    }



# ... (keep existing imports and helper functions)

@router.get("/train-stream")
async def sandbox_train_stream(
    batch_size: int = 4,
    learning_rate: float = 0.0005,
    epochs: int = 10,
    loss_margin: float = 1.5
):
    # Ensure sandbox data exists (same as before)
    ensure_sandbox_data()

    # Create a fresh model (same as in POST)
    model = SiameseResNet34(embed_dim=128, pretrained=False, freeze_backbone=False)
    for name, param in model.encoder.feature_extractor.named_parameters():
        param.requires_grad = False
        if "7" in name:
            param.requires_grad = True
    for param in model.encoder.embedding_head.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    criterion = ContrastiveLoss(margin=loss_margin)

    train_dataset = PairDataset(
        root_dir=SANDBOX_TRAIN_DIR,
        families=FAMILIES,
        image_size=224,
        pairs_per_epoch=500
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    async def event_generator():
        try:
            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct_pairs = 0
                total_pairs = 0
                for img1, img2, labels in train_loader:
                    img1 = img1
                    img2 = img2
                    labels = labels.float()
                    optimizer.zero_grad()
                    emb1, emb2 = model(img1, img2)
                    emb1 = F.normalize(emb1, p=2, dim=1)
                    emb2 = F.normalize(emb2, p=2, dim=1)
                    loss = criterion(emb1, emb2, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    distances = F.pairwise_distance(emb1, emb2)
                    predictions = (distances < loss_margin).float()
                    correct = (predictions == labels).sum().item()
                    correct_pairs += correct
                    total_pairs += len(labels)
                epoch_loss = running_loss / len(train_loader)
                epoch_acc = (correct_pairs / total_pairs) * 100.0

                # Send epoch update
                yield f"data: {json.dumps({'epoch': epoch+1, 'loss': round(epoch_loss, 4), 'accuracy': round(epoch_acc, 2)})}\n\n"
                await asyncio.sleep(0.01)  # allow cancellation

            # After all epochs, evaluate on validation set
            val_dataset = PairDataset(
                root_dir=SANDBOX_VAL_DIR,
                families=FAMILIES,
                image_size=224,
                pairs_per_epoch=200
            )
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            model.eval()
            correct_pairs = 0
            total_pairs = 0
            with torch.no_grad():
                for img1, img2, labels in val_loader:
                    emb1, emb2 = model(img1, img2)
                    distances = F.pairwise_distance(emb1, emb2)
                    predictions = (distances < loss_margin).float()
                    correct = (predictions == labels).sum().item()
                    correct_pairs += correct
                    total_pairs += len(labels)
            val_accuracy = (correct_pairs / total_pairs) * 100.0
            yield f"data: {json.dumps({'done': True, 'val_accuracy': round(val_accuracy, 2)})}\n\n"
        except asyncio.CancelledError:
            yield f"data: {json.dumps({'stopped': True, 'message': 'Training stopped by user'})}\n\n"
            raise

    return StreamingResponse(event_generator(), media_type="text/event-stream")