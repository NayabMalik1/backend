import os
import pickle
from collections import defaultdict

import torch.nn.functional as F

from app.constants import ALL_FAMILIES
from app.inference.make_embedding import load_model, make_embedding
from app.settings import (
    TRAIN_IMAGES_DIR,
    SUPPORT_SET_DIR,
    SUPPORT_EMBEDDINGS_PATH,
    SUPPORT_EMBEDDINGS_DIR,
)

# Seen families = jinke paas training images bhi hain
SEEN_FAMILIES = {"benign", "banking", "smsware"}

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def get_png_images(folder: str):
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(".png")
    )

def add_family_embeddings_from_folder(
    family: str,
    folder_path: str,
    source_tag: str,
    model,
    support_db,
    max_images: int = None,
):
    if not os.path.isdir(folder_path):
        print(f"[WARNING] Folder missing: {folder_path}")
        return 0, 0

    images = get_png_images(folder_path)

    if max_images is not None:
        images = images[:max_images]

    if not images:
        print(f"[WARNING] No PNG images found in: {folder_path}")
        return 0, 0

    print(f"\nProcessing family: {family} | source: {source_tag}")
    print(f"Folder: {folder_path}")
    print(f"Images found: {len(images)}")

    success_count = 0
    fail_count = 0

    for image_name in images:
        image_path = os.path.join(folder_path, image_name)

        try:
            emb = make_embedding(image_path, model)
            emb = F.normalize(emb.unsqueeze(0), p=2, dim=1).squeeze(0)

            support_db[family].append({
                "image_name": image_name,
                "embedding": emb.numpy(),
                "source": source_tag,
            })

            success_count += 1
            print(f"[OK] {source_tag}/{image_name}")

        except Exception as e:
            fail_count += 1
            print(f"[FAILED] {source_tag}/{image_name}: {e}")

    return success_count, fail_count

def save_support_embeddings():
    ensure_dir(SUPPORT_EMBEDDINGS_DIR)

    model = load_model()
    support_db = defaultdict(list)

    for family in ALL_FAMILIES:
        total_success = 0
        total_fail = 0

        # Always include support_set
        support_family_dir = os.path.join(SUPPORT_SET_DIR, family)
        s_ok, s_fail = add_family_embeddings_from_folder(
            family=family,
            folder_path=support_family_dir,
            source_tag="support",
            model=model,
            support_db=support_db,
            max_images=None,   # support images sab use hongi
        )
        total_success += s_ok
        total_fail += s_fail

        # For seen families, also include limited train_images
        if family in SEEN_FAMILIES:
            train_family_dir = os.path.join(TRAIN_IMAGES_DIR, family)
            t_ok, t_fail = add_family_embeddings_from_folder(
                family=family,
                folder_path=train_family_dir,
                source_tag="train",
                model=model,
                support_db=support_db,
                max_images=5,   # only limited train refs
            )
            total_success += t_ok
            total_fail += t_fail

        print(
            f"[SUMMARY] {family}: total_saved={len(support_db[family])}, "
            f"success={total_success}, failed={total_fail}"
        )

    with open(SUPPORT_EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(dict(support_db), f)

    print("\nExpanded gallery embeddings saved at:")
    print(SUPPORT_EMBEDDINGS_PATH)

    print("\nFinal family counts:")
    for family in ALL_FAMILIES:
        print(f"- {family}: {len(support_db[family])} embeddings")

if __name__ == "__main__":
    save_support_embeddings()
