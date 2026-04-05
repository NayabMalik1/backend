import os
import shutil
from typing import Tuple

from .dex_utils import extract_primary_dex_from_apk
from .apk_to_grayscale import dex_to_grayscale_image
from .resize_utils import resize_image

def apk_to_image_pipeline(
    apk_path: str,
    temp_dir: str,
    output_image_path: str,
    final_size: Tuple[int, int] = (300, 300),
) -> str:
    """
    Full pipeline:
    APK -> extract classes.dex -> raw grayscale image -> resized grayscale image
    """

    if not os.path.isfile(apk_path):
        raise FileNotFoundError(f"APK not found: {apk_path}")

    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    dex_extract_dir = os.path.join(temp_dir, "unzipped")
    raw_gray_path = os.path.join(temp_dir, "raw_gray.png")

    # Clean previous temp data
    if os.path.exists(dex_extract_dir):
        shutil.rmtree(dex_extract_dir, ignore_errors=True)

    dex_path = extract_primary_dex_from_apk(apk_path, dex_extract_dir)
    dex_to_grayscale_image(dex_path, raw_gray_path)
    resize_image(raw_gray_path, output_image_path, size=final_size)

    return output_image_path
