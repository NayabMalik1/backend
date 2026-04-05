import os
from typing import Dict, List, Tuple

import numpy as np

from app.inference.make_embedding import load_model, make_embedding
from app.inference.compare_support import compare_with_support
from app.inference.decide_result import decide_prediction
from app.settings import TEST_IMAGES_DIR

SEEN_FAMILIES = ["benign", "banking", "smsware"]
UNSEEN_FAMILIES = ["adware", "riskware"]

def get_family_image_paths(root_dir: str) -> Dict[str, List[str]]:
    family_to_images = {}

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Test directory not found: {root_dir}")

    for family in sorted(os.listdir(root_dir)):
        family_dir = os.path.join(root_dir, family)

        if not os.path.isdir(family_dir):
            continue

        images = sorted(
            os.path.join(family_dir, f)
            for f in os.listdir(family_dir)
            if f.lower().endswith(".png")
        )

        if images:
            family_to_images[family] = images

    return family_to_images

def evaluate_family(
    family: str,
    image_paths: List[str],
    model,
) -> Tuple[int, int, List[dict]]:
    correct = 0
    total = 0
    details = []

    for image_path in image_paths:
        try:
            query_embedding = make_embedding(image_path, model).numpy().astype(np.float32).flatten()
            similarity_scores = compare_with_support(query_embedding)
            result = decide_prediction(similarity_scores)

            predicted_family = result["predicted_family"]
            is_correct = predicted_family == family

            total += 1
            if is_correct:
                correct += 1

            details.append({
                "file_name": os.path.basename(image_path),
                "actual_family": family,
                "predicted_family": predicted_family,
                "predicted_label": result["predicted_label"],
                "confidence": result["confidence"],
                "danger_score": result["danger_score"],
                "is_correct": is_correct,
            })

            status = "OK" if is_correct else "WRONG"
            print(
                f"[{status}] {family} | {os.path.basename(image_path)} "
                f"-> predicted: {predicted_family}, confidence: {result['confidence']}%"
            )

        except Exception as e:
            print(f"[FAILED] {family} | {os.path.basename(image_path)}: {e}")

    return correct, total, details

def safe_accuracy(correct: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round((correct / total) * 100, 2)

def evaluate():
    print("Loading model...")
    model = load_model()

    print(f"Reading test images from: {TEST_IMAGES_DIR}")
    family_to_images = get_family_image_paths(TEST_IMAGES_DIR)

    if not family_to_images:
        raise ValueError("No test images found.")

    overall_correct = 0
    overall_total = 0

    seen_correct = 0
    seen_total = 0

    unseen_correct = 0
    unseen_total = 0

    family_results = {}
    all_details = []

    print("\nStarting evaluation...\n")

    for family, image_paths in family_to_images.items():
        print(f"Evaluating family: {family}")
        correct, total, details = evaluate_family(family, image_paths, model)

        acc = safe_accuracy(correct, total)
        family_results[family] = {
            "correct": correct,
            "total": total,
            "accuracy": acc,
        }
        all_details.extend(details)

        overall_correct += correct
        overall_total += total

        if family in SEEN_FAMILIES:
            seen_correct += correct
            seen_total += total
        elif family in UNSEEN_FAMILIES:
            unseen_correct += correct
            unseen_total += total

        print(f"Family accuracy for {family}: {acc}% ({correct}/{total})\n")

    overall_accuracy = safe_accuracy(overall_correct, overall_total)
    seen_accuracy = safe_accuracy(seen_correct, seen_total)
    unseen_accuracy = safe_accuracy(unseen_correct, unseen_total)

        # ... (your existing evaluation code)

    print("\n" + "=" * 60)
    print("FINAL EVALUATION RESULTS")
    print("=" * 60)
    print(f"Overall Accuracy : {overall_accuracy}% ({overall_correct}/{overall_total})")
    print(f"Seen Accuracy    : {seen_accuracy}% ({seen_correct}/{seen_total})")
    print(f"Unseen Accuracy  : {unseen_accuracy}% ({unseen_correct}/{unseen_total})")
    print("\nPer-Family Accuracy:")
    for family, stats in family_results.items():
        print(f"- {family}: {stats['accuracy']}% ({stats['correct']}/{stats['total']})")

    # ========== SAVE JSON ==========
    from app.settings import TRAINED_MODELS_DIR
    import json
    import os
    results = {
        "overall_accuracy": overall_accuracy,
        "seen_accuracy": seen_accuracy,
        "unseen_accuracy": unseen_accuracy,
        "family_accuracy": {fam: stats["accuracy"] for fam, stats in family_results.items()},
        "final_loss": None
    }
    os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
    output_path = os.path.join(TRAINED_MODELS_DIR, "evaluation_metrics.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation metrics saved to: {output_path}")

    return results

if __name__ == "__main__":
    evaluate()
