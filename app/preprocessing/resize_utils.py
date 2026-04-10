import os
from typing import Tuple

from PIL import Image


def resize_image(
    input_path: str,
    output_path: str,
    size: Tuple[int, int] = (300, 300),
) -> str:
    """
    Resize grayscale image to fixed size for model input.
    """

    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Image not found: {input_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with Image.open(input_path) as img:
        img = img.convert("L")
        img = img.resize(size)
        img.save(output_path)

    return output_path