import os
import shutil
from typing import List, Tuple

import numpy as np
from PIL import Image

# =========================
# SETTINGS
# =========================

SOURCE_DIR = "data/grayscale_images"

TRAIN_DIR = "data/train_images"
SUPPORT_DIR = "data/support_set"
TEST_DIR = "data/test_images"

TRAIN_FAMILIES = ["benign", "banking", "smsware"]
ALL_FAMILIES = ["benign", "banking", "smsware", "adware", "riskware"]

TRAIN_COUNT = 10
SUPPORT_COUNT = 5
TEST_COUNT = 5

# =========================
# MANUAL SUPPORT SELECTION
# =========================
# Agar kisi family ke liye yahan filenames di hui hon,
# to support set unhi files se banega.
# Baqi split automatically hoga.
#
# IMPORTANT:
# Ye filenames exact waise hi honi chahiye jaisi
# SOURCE_DIR/family folder me موجود hain.
#
# Abhi smsware ke liye manual support ON hai.
MANUAL_SUPPORT = {
     "smsware": [
        "020cdc2d622af016d7cbfcee797e078884380a6635ebe70b36a5c527608ec07f.png",
        "0221511d597a5ab7b6303e12675dabadf6f48db968fa26403ee70a041e3a6826.png",
        "043b4fbc2b58040754a20844e8bc85139ce38daffd40ce53ba0a91ba052ca84b.png",
        "0454a5c0ff9fea30a5084af2354ef142f0ee5dbbf3545edb4bf0d07b2242bbeb.png",
        "015b473e1d56054bed16899430ea95f9ac940a45ab0ec4888a119279667e7916.png",
    ]
}

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def reset_family_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def copy_files(files, src_dir, dst_dir):
    ensure_dir(dst_dir)

    for f in files:
        src = os.path.join(src_dir, f)
        dst = os.path.join(dst_dir, f)
        shutil.copy(src, dst)

def get_image_score(image_path: str) -> float:
    """
    Higher score = better / more informative image.
    Prefer images with reasonable contrast and non-extreme brightness.
    """
    try:
        img = Image.open(image_path).convert("L")
        arr = np.array(img, dtype=np.float32)

        mean_val = float(arr.mean())
        std_val = float(arr.std())

        mean_penalty = abs(mean_val - 127.5) / 127.5
        score = std_val - (mean_penalty * 20.0)
        return score
    except Exception:
        return -1e9

def get_ranked_images(family_src: str) -> List[str]:
    images = [
        f for f in os.listdir(family_src)
        if f.lower().endswith(".png")
    ]

    scored_images: List[Tuple[str, float]] = []

    for f in images:
        path = os.path.join(family_src, f)
        score = get_image_score(path)
        if score > -1e8:
            scored_images.append((f, score))

    scored_images.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in scored_images]

def pick_spread_items(images: List[str], count: int) -> List[str]:
    """
    Pick evenly spread samples from a ranked list so support is diverse.
    """
    if len(images) <= count:
        return images[:count]

    indices = np.linspace(0, len(images) - 1, count, dtype=int)
    picked = [images[i] for i in indices]

    unique_picked = []
    for item in picked:
        if item not in unique_picked:
            unique_picked.append(item)

    if len(unique_picked) < count:
        for item in images:
            if item not in unique_picked:
                unique_picked.append(item)
            if len(unique_picked) == count:
                break

    return unique_picked[:count]

def validate_manual_support(family: str, family_src: str, manual_files: List[str]) -> List[str]:
    """
    Keep only valid manual support files that actually exist.
    """
    valid = []
    missing = []

    for f in manual_files:
        full_path = os.path.join(family_src, f)
        if os.path.isfile(full_path):
            valid.append(f)
        else:
            missing.append(f)

    if missing:
        print(f"[WARNING] Missing manual support files for {family}:")
        for f in missing:
            print(f"  - {f}")

    if len(valid) < SUPPORT_COUNT:
        print(
            f"[WARNING] Manual support for {family} has only {len(valid)} valid files. "
            f"Need {SUPPORT_COUNT}. Falling back to auto-fill for remaining."
        )

    return valid

def split_seen_family(images: List[str], family: str, family_src: str):
    """
    Seen family split:
    - support = 5
    - train   = 10
    - test    = 5

    smsware ke liye manual support allow hai.
    """
    required = SUPPORT_COUNT + TRAIN_COUNT + TEST_COUNT
    if len(images) < required:
        print(f"[WARNING] Seen family has fewer than required images: {len(images)} < {required}")

    pool = images[:max(required, 20)]

    # =========================
    # Manual support mode
    # =========================
    if family in MANUAL_SUPPORT:
        manual_support = validate_manual_support(family, family_src, MANUAL_SUPPORT[family])

        remaining_candidates = [img for img in pool if img not in manual_support]

        # Agar manual support 5 se kam ho to auto-fill kar do
        if len(manual_support) < SUPPORT_COUNT:
            needed = SUPPORT_COUNT - len(manual_support)
            auto_fill = remaining_candidates[:needed]
            support = manual_support + auto_fill
        else:
            support = manual_support[:SUPPORT_COUNT]

        remaining = [img for img in pool if img not in support]
        train = remaining[:TRAIN_COUNT]
        test = remaining[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

        return train, support, test

    # =========================
    # Auto split for seen families
    # =========================
    support_candidates = pool[:15] if len(pool) >= 15 else pool
    support = pick_spread_items(support_candidates, SUPPORT_COUNT)

    remaining = [img for img in pool if img not in support]
    train = remaining[:TRAIN_COUNT]
    test = remaining[TRAIN_COUNT:TRAIN_COUNT + TEST_COUNT]

    return train, support, test

def split_unseen_family(images: List[str]):
    """
    Unseen family split:
    - train   = 0
    - support = 5
    - test    = 5
    """
    required = SUPPORT_COUNT + TEST_COUNT
    if len(images) < required:
        print(f"[WARNING] Unseen family has fewer than required images: {len(images)} < {required}")

    pool = images[:max(required, 15)]

    support_candidates = pool[:10] if len(pool) >= 10 else pool
    support = pick_spread_items(support_candidates, SUPPORT_COUNT)

    remaining = [img for img in pool if img not in support]
    test = remaining[:TEST_COUNT]

    train = []
    return train, support, test

def main():
    print("\n========== DATASET SPLIT START ==========\n")

    ensure_dir(TRAIN_DIR)
    ensure_dir(SUPPORT_DIR)
    ensure_dir(TEST_DIR)

    for family in ALL_FAMILIES:
        family_src = os.path.join(SOURCE_DIR, family)

        if not os.path.isdir(family_src):
            print(f"[WARNING] Missing source folder: {family_src}")
            continue

        images = get_ranked_images(family_src)

        print(f"\nProcessing: {family}")
        print("Valid images found:", len(images))

        if family in TRAIN_FAMILIES:
            train, support, test = split_seen_family(images, family, family_src)
        else:
            train, support, test = split_unseen_family(images)

        reset_family_dir(os.path.join(TRAIN_DIR, family))
        reset_family_dir(os.path.join(SUPPORT_DIR, family))
        reset_family_dir(os.path.join(TEST_DIR, family))

        copy_files(train, family_src, os.path.join(TRAIN_DIR, family))
        copy_files(support, family_src, os.path.join(SUPPORT_DIR, family))
        copy_files(test, family_src, os.path.join(TEST_DIR, family))

        print("Train  :", len(train))
        print("Support:", len(support))
        print("Test   :", len(test))

        if family in TRAIN_FAMILIES:
            if family in MANUAL_SUPPORT:
                print("Seen split  -> manual support(5), train(10), test(5)")
                print("Manual support files:")
                for f in support:
                    print(f"  - {f}")
            else:
                print("Seen split  -> support(diverse 5), train(10), test(5)")
        else:
            print("Unseen split -> support(diverse 5), test(5), no train")

    print("\n========== DATASET SPLIT DONE ==========\n")

if __name__ == "__main__":
    main()
