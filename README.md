# 🛡️ Real-Time Fraud Detection System

An end-to-end AI/ML system for detecting fraudulent financial transactions in real time.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Transaction Stream                           │
│        (Card payments, ATM, Online, Mobile)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              Feature Engineering Pipeline                       │
│  • Velocity features  • Amount deviation  • Geo features        │
│  • Time-based         • User behaviour    • Risk signals        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML Model Ensemble                            │
│  XGBoost │ LightGBM │ Random Forest │ Neural Network │ IsoForest│
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────┐       ┌─────────────────────┐
   │  FastAPI Server │       │  Streamlit Dashboard │
   │  /predict       │       │  Live monitoring     │
   │  /batch_predict │       │  Alert management    │
   │  /alerts        │       │  Model performance   │
   └─────────────────┘       └─────────────────────┘
```

## Fraud Patterns Detected

| Pattern | Description | Typical Signals |
|---------|-------------|-----------------|
| **Card Testing** | Fraudster tests stolen card with micro-charges | Small amounts < $5, rapid succession, online merchants |
| **Account Takeover** | Compromised account used for large purchases | Large amount, unusual location, failed auth attempts |
| **CNP Fraud** | Card-Not-Present: stolen details used online | Online purchase, high-value, new device |
| **Velocity Abuse** | Many transactions in short time window | txn_velocity_1h > 5, crypto/money transfer category |
| **Geographic Anomaly** | Impossible travel patterns | distance > 1000km, international flag |

## ML Models

| Model | Type | Strength |
|-------|------|---------|
| **XGBoost** | Gradient Boosting | Best overall performance |
| **LightGBM** | Fast Gradient Boosting | Speed + accuracy balance |
| **Random Forest** | Bagging Ensemble | Robust, interpretable |
| **Neural Network** | MLP (3 hidden layers) | Complex pattern recognition |
| **Logistic Regression** | Linear | Fast baseline |
| **Isolation Forest** | Unsupervised Anomaly | No labels required |

## Quick Start

### 1. Install dependencies

```bash
cd real-time-fraud-detection
pip install -r requirements.txt
```

### 2. Train all models

```bash
# Full training (500k transactions, all 6 models)
python scripts/train_models.py

# Quick mode (50k transactions, fewer estimators)
python scripts/train_models.py --quick
```

### 3. Launch the REST API

```bash
uvicorn fraud_detection.api.server:app --host 0.0.0.0 --port 8000 --reload
```

API docs available at: http://localhost:8000/docs

### 4. Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard at: http://localhost:8501

### 5. Run the real-time pipeline simulation

```bash
python scripts/run_pipeline.py --tps 100 --n 500
```

### 6. Run tests

```bash
pytest tests/ -v
```

## API Reference

### Score a single transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_001",
    "user_id": "USR_000001",
    "merchant_id": "MRC_00001",
    "merchant_category": "electronics",
    "timestamp": "2025-01-15 14:32:00",
    "amount": 1500.00,
    "transaction_type": "online_purchase",
    "device_type": "web_browser",
    "is_online": 1,
    "card_present": 0,
    "is_international": 1,
    "distance_from_home_km": 2500,
    "failed_attempts_before": 3,
    "account_age_days": 45
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN_001",
  "fraud_probability": 0.9432,
  "fraud_prediction": 1,
  "risk_tier": "CRITICAL",
  "risk_description": "🔴 Critical fraud signal — block immediately",
  "top_risk_factors": [
    "High transaction amount: $1500.00",
    "Far from home location: 2500 km",
    "International transaction",
    "3 failed auth attempts before this transaction",
    "Online CNP purchase pattern detected"
  ],
  "latency_ms": 3.2,
  "scored_at": "2025-05-12T09:45:22"
}
```

### Batch scoring

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"transactions": [...]}'
```

## Feature Engineering

The system generates 20+ engineered features from raw transactions:

| Feature | Description |
|---------|-------------|
| `txn_velocity_1h` | Transactions by this user in last 1 hour |
| `txn_velocity_24h` | Transactions by this user in last 24 hours |
| `txn_velocity_7d` | Transactions by this user in last 7 days |
| `amount_deviation` | Z-score of amount vs user historical average |
| `amount_to_avg_ratio` | Current amount / user avg amount |
| `unique_merchants_24h` | Unique merchants in last 24h |
| `distance_from_home_km` | Geographic distance from home |
| `is_large_distance` | Flag: distance > 200km |
| `is_very_large_distance` | Flag: distance > 1000km |
| `time_since_last_txn_min` | Minutes since user's last transaction |
| `is_rapid_succession` | Flag: < 2 min since last transaction |
| `is_night` | Flag: transaction between 11pm-6am |
| `amount_log` | Log-transformed amount |
| `amount_rounded` | Flag: round number amount |
| `is_weekend` | Flag: Saturday or Sunday |

## Project Structure

```
real-time-fraud-detection/
├── fraud_detection/
│   ├── data/
│   │   ├── generator.py        # Synthetic fraud data generation
│   │   └── preprocessor.py     # Feature engineering + preprocessing
│   ├── models/
│   │   ├── trainer.py          # Multi-model training pipeline
│   │   ├── evaluator.py        # Metrics, ROC/PR curves, SHAP plots
│   │   └── predictor.py        # Real-time inference engine
│   ├── pipeline/
│   │   └── stream_processor.py # Real-time stream processing
│   └── api/
│       └── server.py           # FastAPI REST endpoints
├── dashboard/
│   └── app.py                  # Streamlit monitoring dashboard
├── scripts/
│   ├── train_models.py         # End-to-end training script
│   └── run_pipeline.py         # Pipeline simulation
├── tests/
│   └── test_fraud_detection.py # Pytest test suite
├── data/                       # Generated datasets
├── models/                     # Saved model artefacts + plots
├── config.yaml                 # System configuration
├── requirements.txt
└── setup.py
```

## Model Performance (Typical Results)

| Model | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|-------|---------|--------|----|-----------|--------|
| XGBoost | 0.992 | 0.877 | 0.823 | 0.877 | 0.777 |
| LightGBM | 0.988 | 0.865 | 0.812 | 0.865 | 0.765 |
| Random Forest | 0.985 | 0.854 | 0.804 | 0.843 | 0.765 |
| Neural Network | 0.981 | 0.843 | 0.797 | 0.832 | 0.764 |
| Logistic Regression | 0.963 | 0.787 | 0.754 | 0.812 | 0.701 |
| Isolation Forest | 0.872 | 0.654 | — | — | — |

> Evaluated on 500k synthetic transactions with 1.5% fraud rate, SMOTE-balanced training.

## Technologies

- **Python 3.9+**
- **scikit-learn** — core ML algorithms
- **XGBoost / LightGBM** — gradient boosting
- **imbalanced-learn** — SMOTE oversampling
- **FastAPI** — high-performance REST API
- **Streamlit** — interactive dashboard
- **Plotly** — interactive visualizations
- **SHAP** — model explainability
- **Pydantic v2** — request/response validation
- **Pytest** — test suite
