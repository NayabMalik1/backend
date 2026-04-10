from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=True)   # NEW
    role = Column(String(20), nullable=False)
    university = Column(String(200), nullable=True)
    organization = Column(String(200), nullable=True)
    org_details = Column(String(500), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class PasswordResetToken(Base):
    __tablename__ = "password_reset_tokens"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    token = Column(String(255), unique=True, index=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)

class ScanHistory(Base):
    __tablename__ = "scan_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    file_name = Column(String(255), nullable=False)
    predicted_family = Column(String(50))
    predicted_label = Column(String(20))
    confidence = Column(Float)
    danger_score = Column(Float)
    permissions = Column(JSON)      # store list
    api_calls = Column(JSON)        # store list
    full_response = Column(JSON)    # store the whole scan result
    scanned_at = Column(DateTime(timezone=True), server_default=func.now())