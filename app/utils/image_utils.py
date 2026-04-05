from PIL import Image

def open_grayscale(image_path: str):
    return Image.open(image_path).convert("L")
