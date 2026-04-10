from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.db_models import ScanHistory, User
from app.auth import get_current_user

router = APIRouter(prefix="/history", tags=["history"])

@router.get("/")
def get_history(
    skip: int = 0,
    limit: int = 50,
    threat_type: str = Query(None, regex="^(malware|benign)$"),
    search: str = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    query = db.query(ScanHistory).filter(ScanHistory.user_id == current_user.id)
    if threat_type:
        query = query.filter(ScanHistory.predicted_label == threat_type)
    if search:
        query = query.filter(ScanHistory.file_name.contains(search))
    results = query.order_by(ScanHistory.scanned_at.desc()).offset(skip).limit(limit).all()
    return results

@router.post("/save")
def save_scan(scan_data: dict, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    history = ScanHistory(
        user_id=current_user.id,
        file_name=scan_data.get("file_name"),
        predicted_family=scan_data.get("predicted_family"),
        predicted_label=scan_data.get("predicted_label"),
        confidence=scan_data.get("confidence"),
        danger_score=scan_data.get("danger_score"),
        permissions=scan_data.get("permissions", []),
        api_calls=scan_data.get("api_calls", []),
        full_response=scan_data
    )
    db.add(history)
    db.commit()
    return {"message": "Saved"}