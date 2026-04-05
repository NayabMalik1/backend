import os
import zipfile
from app.preprocessing.apk_pipeline import apk_to_image_pipeline

# =========================
# SETTINGS
# =========================
RAW_APKS_DIR = "data/raw_apks"
GRAY_IMAGES_DIR = "data/grayscale_images"
TEMP_DIR = "temp"
FINAL_SIZE = (300, 300)

# Target number of grayscale images per family
TARGET_IMAGES_PER_FAMILY = 20

# Families to process
FAMILIES = ["benign", "banking", "smsware", "adware", "riskware"]

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def is_valid_apk(file_path: str) -> bool:
    """
    Check file validity by trying to open it as APK/ZIP
    and verifying that at least one .dex exists.
    Extension does not matter.
    """
    try:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            names = zip_ref.namelist()
            for name in names:
                if name.lower().endswith(".dex"):
                    return True
    except Exception:
        return False
    return False

def count_existing_images(folder: str) -> int:
    """Count PNG files in a directory."""
    if not os.path.isdir(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.lower().endswith(".png")])

def main():
    ensure_dir(RAW_APKS_DIR)
    ensure_dir(GRAY_IMAGES_DIR)
    ensure_dir(TEMP_DIR)

    total_files = 0
    total_converted = 0
    total_skipped = 0
    total_failed = 0

    print("\n========== APK TO GRAYSCALE DATASET SETUP ==========")
    print(f"Target images per family: {TARGET_IMAGES_PER_FAMILY}\n")

    for family in FAMILIES:
        input_family_dir = os.path.join(RAW_APKS_DIR, family)
        output_family_dir = os.path.join(GRAY_IMAGES_DIR, family)
        ensure_dir(output_family_dir)

        # Count existing images
        existing = count_existing_images(output_family_dir)
        needed = TARGET_IMAGES_PER_FAMILY - existing

        if needed <= 0:
            print(f"\n--- Family: {family} ---")
            print(f"Already have {existing} images (target {TARGET_IMAGES_PER_FAMILY}). Skipping.")
            continue

        if not os.path.isdir(input_family_dir):
            print(f"[WARNING] Family folder not found: {input_family_dir}")
            continue

        files = [
            f for f in os.listdir(input_family_dir)
            if os.path.isfile(os.path.join(input_family_dir, f))
        ]

        print(f"\n--- Processing family: {family} ---")
        print(f"Input folder : {input_family_dir}")
        print(f"Output folder: {output_family_dir}")
        print(f"Existing images: {existing}")
        print(f"Need {needed} more to reach {TARGET_IMAGES_PER_FAMILY}")
        print(f"Available raw APKs: {len(files)}")

        family_converted = 0
        family_skipped = 0
        family_failed = 0

        # Process files until we have enough successful conversions
        for file_name in files:
            # Stop if we already reached the target
            if family_converted >= needed:
                break

            total_files += 1
            file_path = os.path.join(input_family_dir, file_name)
            image_name = os.path.splitext(file_name)[0] + ".png"
            output_image_path = os.path.join(output_family_dir, image_name)

            # Skip if output already exists (to avoid reprocessing)
            if os.path.isfile(output_image_path):
                print(f"[SKIP] Already exists: {image_name}")
                continue

            if not is_valid_apk(file_path):
                print(f"[SKIPPED] Not a valid APK: {file_name}")
                total_skipped += 1
                family_skipped += 1
                continue

            try:
                apk_to_image_pipeline(
                    apk_path=file_path,
                    temp_dir=TEMP_DIR,
                    output_image_path=output_image_path,
                    final_size=FINAL_SIZE,
                )
                print(f"[OK] {file_name} -> {image_name}")
                total_converted += 1
                family_converted += 1
            except Exception as e:
                print(f"[FAILED] {file_name}: {e}")
                total_failed += 1
                family_failed += 1

        print(f"\nFamily summary: {family}")
        print(f"Converted this run: {family_converted}")
        print(f"Skipped (invalid APK): {family_skipped}")
        print(f"Failed (conversion error): {family_failed}")
        print(f"Total images now in output: {existing + family_converted} / {TARGET_IMAGES_PER_FAMILY}")

    print("\n========== FINAL DATASET REPORT ==========")
    print(f"Total files processed (across all families): {total_files}")
    print(f"Total converted (new images): {total_converted}")
    print(f"Total skipped (invalid APKs): {total_skipped}")
    print(f"Total failed (conversion errors): {total_failed}")
    print("\nDone.")

if __name__ == "__main__":
    main()