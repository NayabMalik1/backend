from typing import Dict
import numpy as np

def softmax(x: np.ndarray, temperature: float = 0.30) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    if x.size == 0:
        return x

    temperature = max(temperature, 1e-6)
    x = x / temperature
    x = x - np.max(x)

    exp_x = np.exp(x)
    denom = np.sum(exp_x)

    if denom < 1e-8:
        return np.ones_like(x) / len(x)

    return exp_x / denom

def decide_prediction(similarity_scores: Dict[str, float]) -> Dict[str, float]:
    if not similarity_scores:
        return {
            "predicted_family": "unknown",
            "predicted_label": "unknown",
            "confidence": 0.0,
            "danger_score": 0.0,
            "family_matches": {},
        }

    families = list(similarity_scores.keys())
    scores = np.array([similarity_scores[f] for f in families], dtype=np.float32)

    probs = softmax(scores, temperature=0.30)

    ranked = sorted(
        zip(families, scores, probs),
        key=lambda x: x[1],
        reverse=True
    )

    best_family, best_score, best_prob = ranked[0]

    if len(ranked) > 1:
        second_family, second_score, second_prob = ranked[1]
        margin = float(best_score - second_score)
    else:
        second_family, second_score, second_prob = "none", 0.0, 0.0
        margin = float(best_score)

    margin_conf = max(0.0, min(1.0, (margin + 0.05) / 0.15))

    confidence = (0.75 * float(best_prob)) + (0.25 * margin_conf)
    confidence = float(max(0.0, min(1.0, confidence))) * 100.0

    if best_family == "benign":
        predicted_label = "benign"
        danger_score = max(0.0, 100.0 - confidence)
    else:
        predicted_label = "malware"
        danger_score = confidence

    family_matches = {
        family: round(float(prob) * 100.0, 2)
        for family, prob in zip(families, probs)
    }

    return {
        "predicted_family": best_family,
        "predicted_label": predicted_label,
        "confidence": round(confidence, 2),
        "danger_score": round(danger_score, 2),
        "family_matches": family_matches,
    }

if __name__ == "__main__":
    scores = {
        "trojan": 0.82,
        "ransomware": 0.61,
        "benign": 0.12
    }

    result = decide_prediction(scores)
    print(result)
