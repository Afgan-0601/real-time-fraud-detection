"""
FastAPI Real-Time Fraud Detection API

Endpoints
─────────
POST  /predict          — Score a single transaction
POST  /predict/batch    — Score up to 500 transactions
GET   /pipeline/stats   — Live pipeline statistics
GET   /pipeline/alerts  — Recent fraud alerts
GET   /models/metrics   — Leaderboard from last training run
GET   /health           — Health check
"""

import os
import json
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

import yaml
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from fraud_detection.models.predictor import FraudPredictor
from fraud_detection.pipeline.stream_processor import RealTimePipeline, AlertManager


# ─────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    user_id: str = Field(..., description="Cardholder user ID")
    merchant_id: str = Field(..., description="Merchant identifier")
    merchant_category: str = Field(..., description="Merchant category code")
    timestamp: str = Field(..., description="ISO-8601 transaction timestamp")
    amount: float = Field(..., gt=0, le=1_000_000, description="Transaction amount in USD")
    transaction_type: str = Field(..., description="purchase / online_purchase / atm_withdrawal etc.")
    device_type: str = Field(..., description="mobile_app / web_browser / pos_terminal / atm")
    is_online: int = Field(default=0, ge=0, le=1)
    card_present: int = Field(default=1, ge=0, le=1)
    is_international: int = Field(default=0, ge=0, le=1)
    distance_from_home_km: float = Field(default=0.0, ge=0.0, description="Distance from cardholder home")
    failed_attempts_before: int = Field(default=0, ge=0, le=20)
    account_age_days: int = Field(default=365, ge=0)

    @field_validator("amount")
    @classmethod
    def amount_positive(cls, v):
        if v <= 0:
            raise ValueError("amount must be positive")
        return round(v, 2)


class TransactionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float
    fraud_prediction: int
    risk_tier: str
    risk_description: str
    top_risk_factors: List[str]
    latency_ms: float
    scored_at: str


class BatchRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    results: List[Dict[str, Any]]
    total: int
    flagged: int
    batch_latency_ms: float
    scored_at: str


# ─────────────────────────────────────────────────────────────────────
# App state (loaded at startup)
# ─────────────────────────────────────────────────────────────────────

class AppState:
    predictor: Optional[FraudPredictor] = None
    pipeline: Optional[RealTimePipeline] = None
    alerts: Optional[AlertManager] = None
    config: Optional[dict] = None
    startup_time: float = 0.0


app_state = AppState()


def _load_config() -> dict:
    for path in ["config.yaml", "../config.yaml", "../../config.yaml"]:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError("config.yaml not found")


# ─────────────────────────────────────────────────────────────────────
# App lifecycle
# ─────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load models and start pipeline on startup."""
    app_state.startup_time = time.time()
    app_state.config = _load_config()
    cfg = app_state.config

    model_path = cfg["api"]["model_path"]
    preprocessor_path = "models/artifacts/preprocessor.pkl"

    if os.path.exists(model_path) and os.path.exists(preprocessor_path):
        app_state.predictor = FraudPredictor(
            model_path=model_path,
            preprocessor_path=preprocessor_path,
            config=cfg,
        )
        app_state.pipeline = RealTimePipeline(
            predictor=app_state.predictor,
            config=cfg,
        )
        app_state.pipeline.start(n_workers=cfg["api"].get("workers", 4))
        app_state.alerts = app_state.pipeline.alerts
        print("✅ Fraud Detection API ready.")
    else:
        print("⚠️  Model artefacts not found. Run `python scripts/train_models.py` first.")

    yield

    # Shutdown
    if app_state.pipeline:
        app_state.pipeline.stop()
    print("API shutdown complete.")


# ─────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description=(
        "AI/ML-powered real-time transaction fraud scoring engine. "
        "Returns fraud probability, risk tier, and human-readable risk factors."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────
# Request timing middleware
# ─────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(
        round((time.perf_counter() - t0) * 1000, 2)
    )
    return response


# ─────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────

def _require_predictor():
    if app_state.predictor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run `python scripts/train_models.py` to train.",
        )


# ─────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check():
    return {
        "status": "ok",
        "model_loaded": app_state.predictor is not None,
        "pipeline_running": app_state.pipeline is not None and app_state.pipeline._running,
        "uptime_seconds": round(time.time() - app_state.startup_time, 1),
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/predict", response_model=TransactionResponse, tags=["Prediction"])
def predict_single(request: TransactionRequest):
    """Score a single transaction for fraud."""
    _require_predictor()
    txn_dict = request.model_dump()
    result = app_state.predictor.score_transaction(txn_dict)

    # Also ingest into live pipeline
    if app_state.pipeline:
        app_state.pipeline.ingest(txn_dict)

    return TransactionResponse(
        **result,
        scored_at=datetime.utcnow().isoformat(),
    )


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
def predict_batch(request: BatchRequest):
    """Score a batch of up to 500 transactions."""
    _require_predictor()
    t0 = time.perf_counter()
    txns = [t.model_dump() for t in request.transactions]
    results = app_state.predictor.score_batch(txns)

    flagged = sum(1 for r in results if r["fraud_prediction"] == 1)
    batch_ms = round((time.perf_counter() - t0) * 1000, 2)

    return BatchResponse(
        results=results,
        total=len(results),
        flagged=flagged,
        batch_latency_ms=batch_ms,
        scored_at=datetime.utcnow().isoformat(),
    )


@app.get("/pipeline/stats", tags=["Pipeline"])
def pipeline_stats():
    """Return live pipeline performance statistics."""
    if app_state.pipeline is None:
        return {"message": "Pipeline not running"}
    return app_state.pipeline.stats.summary()


@app.get("/pipeline/alerts", tags=["Pipeline"])
def recent_alerts(limit: int = 50):
    """Return the most recent fraud alerts."""
    if app_state.alerts is None:
        return {"alerts": [], "total": 0}
    alerts = app_state.alerts.get_recent(limit)
    return {"alerts": alerts, "total": app_state.alerts.total_alerts}


@app.get("/models/metrics", tags=["Models"])
def model_metrics():
    """Return evaluation metrics from the last training run."""
    leaderboard_path = "models/reports/all_models_metrics.csv"
    if not os.path.exists(leaderboard_path):
        # Fall back to leaderboard saved by trainer
        leaderboard_path = "models/artifacts/leaderboard.csv"
    if not os.path.exists(leaderboard_path):
        return {"message": "No training results found. Run train_models.py first."}
    df = pd.read_csv(leaderboard_path, index_col=0)
    return df.to_dict(orient="index")


@app.get("/models/features", tags=["Models"])
def feature_list():
    """Return the feature names used by the loaded model."""
    _require_predictor()
    return {"features": app_state.predictor._feature_names}


# ─────────────────────────────────────────────────────────────────────
# Dev server entrypoint
# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    cfg = _load_config()
    uvicorn.run(
        "fraud_detection.api.server:app",
        host=cfg["api"]["host"],
        port=cfg["api"]["port"],
        reload=False,
        workers=1,
    )
