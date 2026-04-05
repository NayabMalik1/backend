from typing import Dict, List
from pydantic import BaseModel

class ScanResponse(BaseModel):
    success: bool
    file_name: str
    predicted_family: str
    predicted_label: str
    confidence: float
    danger_score: float
    model_name: str
    grayscale_image_url: str
    family_matches: Dict[str, float]
    processing_steps: List[str]
