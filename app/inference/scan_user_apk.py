import os
import uuid
import shutil
import numpy as np

from app.preprocessing.apk_pipeline import apk_to_image_pipeline
from app.inference.make_embedding import load_model, make_embedding
from app.inference.compare_support import compare_with_support
from app.inference.decide_result import decide_prediction
from app.settings import UPLOADS_DIR, OUTPUTS_DIR

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def scan_user_apk(apk_file_path: str):
    ensure_dir(UPLOADS_DIR)
    ensure_dir(OUTPUTS_DIR)

    base_id = str(uuid.uuid4())
    temp_dir = os.path.join(UPLOADS_DIR, f"temp_{base_id}")
    output_image_path = os.path.join(OUTPUTS_DIR, f"{base_id}.png")

    processing_steps = [
        "APK uploaded successfully",
        "Extracting classes.dex",
        "Generating grayscale image",
        "Generating embedding",
        "Comparing with support families",
        "Preparing final prediction",
    ]

    try:
        apk_to_image_pipeline(
            apk_path=apk_file_path,
            temp_dir=temp_dir,
            output_image_path=output_image_path,
            final_size=(224, 224),
        )

        model = load_model()
        query_embedding = make_embedding(output_image_path, model).numpy().astype(np.float32).flatten()

        similarity_scores = compare_with_support(query_embedding)
        result = decide_prediction(similarity_scores)

        return {
            "success": True,
            "file_name": os.path.basename(apk_file_path),
            "predicted_family": result["predicted_family"],
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "danger_score": result["danger_score"],
            "model_name": "Siamese ResNet34",
            "grayscale_image_url": f"/outputs/{os.path.basename(output_image_path)}",
            "family_matches": result["family_matches"],
            "processing_steps": processing_steps,
        }

    finally:
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
