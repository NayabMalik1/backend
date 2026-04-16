import os
import shutil
import uuid
import json
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ScanResponse
from app.settings import UPLOADS_DIR, OUTPUTS_DIR, TRAINED_MODELS_DIR, DATA_DIR, SUPPORT_EMBEDDINGS_PATH
from app.inference.scan_user_apk import scan_user_apk
from app.database import engine, Base
from app.routers import auth, history, local_report
from app.routers import sandbox
from app.routers import sandbox

# ... after other routers


app = FastAPI(title="Android Malware FSL API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Static files
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Create database tables
Base.metadata.create_all(bind=engine)

# Include routers (only once each)
app.include_router(auth.router)
app.include_router(history.router)
app.include_router(local_report.router)
app.include_router(sandbox.router)

@app.get("/")
def root():
    return {"message": "Android Malware FSL backend is running"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/scan", response_model=ScanResponse)
async def scan(apk: UploadFile = File(...)):
    if not apk.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")
    unique_name = f"{uuid.uuid4()}_{apk.filename}"
    apk_path = os.path.join(UPLOADS_DIR, unique_name)
    try:
        with open(apk_path, "wb") as f:
            shutil.copyfileobj(apk.file, f)
        result = scan_user_apk(apk_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ========== DASHBOARD ENDPOINTS ==========
@app.get("/api/train-history")
async def get_train_history():
    history_path = os.path.join(TRAINED_MODELS_DIR, "train_history.json")
    if not os.path.isfile(history_path):
        return []
    with open(history_path, "r") as f:
        return json.load(f)

@app.get("/api/evaluation-metrics")
async def get_evaluation_metrics():
    metrics_path = os.path.join(TRAINED_MODELS_DIR, "evaluation_metrics.json")
    if not os.path.isfile(metrics_path):
        return {
            "overall_accuracy": 0.0,
            "seen_accuracy": 0.0,
            "unseen_accuracy": 0.0,
            "family_accuracy": {},
            "final_loss": None
        }
    with open(metrics_path, "r") as f:
        return json.load(f)

@app.get("/api/dashboard-stats")
async def dashboard_stats():
    try:
        raw_dir = os.path.join(DATA_DIR, "raw_apks")
        families = ["benign", "banking", "smsware", "adware", "riskware"]
        raw_apks = {}
        for family in families:
            family_dir = os.path.join(raw_dir, family)
            if os.path.isdir(family_dir):
                raw_apks[family] = len([f for f in os.listdir(family_dir) if os.path.isfile(os.path.join(family_dir, f))])
            else:
                raw_apks[family] = 0

        splits = ["train_images", "test_images", "support_set"]
        image_splits = {}
        split_totals = {}
        for split in splits:
            split_dir = os.path.join(DATA_DIR, split)
            image_splits[split] = {}
            total = 0
            for family in families:
                family_dir = os.path.join(split_dir, family)
                if os.path.isdir(family_dir):
                    count = len([f for f in os.listdir(family_dir) if f.endswith(".png")])
                else:
                    count = 0
                image_splits[split][family] = count
                total += count
            split_totals[split] = total

        embedding_count = 0
        if os.path.isfile(SUPPORT_EMBEDDINGS_PATH):
            try:
                with open(SUPPORT_EMBEDDINGS_PATH, "rb") as f:
                    data = pickle.load(f)
                    embedding_count = sum(len(v) for v in data.values())
            except:
                pass

        training_history = await get_train_history()
        evaluation = await get_evaluation_metrics()

        return {
            "success": True,
            "raw_apks": raw_apks,
            "image_splits": image_splits,
            "split_totals": split_totals,
            "embedding_count": embedding_count,
            "training_history": training_history,
            "evaluation": evaluation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))