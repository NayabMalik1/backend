import math
import os
from typing import Tuple

import numpy as np
from PIL import Image

def calculate_image_width(file_size: int) -> int:
    """
    Common practical width selection based on file size.
    This gives more stable image shapes than pure sqrt for many APKs.
    """

    if file_size < 10_000:
        return 32
    if file_size < 30_000:
        return 64
    if file_size < 60_000:
        return 128
    if file_size < 100_000:
        return 256
    if file_size < 200_000:
        return 384
    if file_size < 500_000:
        return 512
    if file_size < 1_000_000:
        return 768
    return 1024

def dex_to_grayscale_array(dex_path: str) -> np.ndarray:
    """
    Read a .dex file as raw bytes and convert to 2D uint8 grayscale array.
    """

    if not os.path.isfile(dex_path):
        raise FileNotFoundError(f"DEX file not found: {dex_path}")

    with open(dex_path, "rb") as f:
        byte_data = f.read()

    if not byte_data:
        raise Exception(f"DEX file is empty: {dex_path}")

    byte_array = np.frombuffer(byte_data, dtype=np.uint8)

    width = calculate_image_width(len(byte_array))
    height = math.ceil(len(byte_array) / width)

    padded_size = width * height
    padded_array = np.pad(
        byte_array,
        (0, padded_size - len(byte_array)),
        mode="constant",
        constant_values=0,
    )

    image_array = padded_array.reshape((height, width))
    return image_array

def dex_to_grayscale_image(dex_path: str, output_image_path: str) -> str:
    """
    Convert DEX file to grayscale PNG image and save it.
    """

    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)

    image_array = dex_to_grayscale_array(dex_path)
    img = Image.fromarray(image_array, mode="L")
    img.save(output_image_path)

    return output_image_path
