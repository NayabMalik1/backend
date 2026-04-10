from typing import Dict, List, Optional
from pydantic import BaseModel, Field

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
    permissions: List[str] = []
    api_calls: List[str] = []

# ========== AUTH MODELS ==========
class UserCreate(BaseModel):
    email: str = Field(..., pattern=r'^[^@]+@[^@]+\.[^@]+$')
    password: str
    full_name: str                     # ADDED
    role: str
    university: Optional[str] = None
    organization: Optional[str] = None
    org_details: Optional[str] = None

class UserLogin(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class PasswordResetRequest(BaseModel):
    email: str

class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str