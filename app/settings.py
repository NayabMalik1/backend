import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(APP_DIR)

DATA_DIR = os.path.join(BACKEND_DIR, "data")

RAW_APKS_DIR = os.path.join(DATA_DIR, "raw_apks")
GRAY_IMAGES_DIR = os.path.join(DATA_DIR, "grayscale_images")
TRAIN_IMAGES_DIR = os.path.join(DATA_DIR, "train_images")
SUPPORT_SET_DIR = os.path.join(DATA_DIR, "support_set")
TEST_IMAGES_DIR = os.path.join(DATA_DIR, "test_images")

UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")

TRAINED_MODELS_DIR = os.path.join(BACKEND_DIR, "trained_models")
SUPPORT_EMBEDDINGS_DIR = os.path.join(BACKEND_DIR, "support_embeddings")

MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, "resnet34_best.pth")
SUPPORT_EMBEDDINGS_PATH = os.path.join(SUPPORT_EMBEDDINGS_DIR, "support_embeddings.pkl")

IMAGE_SIZE = 224
EMBED_DIM = 128

DEVICE = "cpu"