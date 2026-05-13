"""
Fraud Detection System — Test Suite

Tests
─────
- Data generator produces correct schema and fraud rate
- Feature engineering creates expected columns
- Preprocessor fit/transform produces numeric features
- Predictor returns valid risk tiers and probabilities
- FastAPI endpoints return correct response shapes
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

with open("config.yaml") as f:
    CONFIG = yaml.safe_load(f)

# ─────────────────────────────────────────────────────────────────────
# Data Generator Tests
# ─────────────────────────────────────────────────────────────────────

class TestFraudDataGenerator:
    def setup_method(self):
        cfg = CONFIG.copy()
        cfg["data"]["n_samples"] = 2000   # small for fast tests
        cfg["data"]["n_users"] = 100
        cfg["data"]["n_merchants"] = 50
        from fraud_detection.data.generator import FraudDataGenerator
        self.gen = FraudDataGenerator(cfg)

    def test_generates_correct_row_count(self):
        df = self.gen.generate(save=False)
        assert len(df) == 2000

    def test_required_columns_present(self):
        df = self.gen.generate(save=False)
        required = [
            "transaction_id", "user_id", "merchant_id", "merchant_category",
            "timestamp", "amount", "is_fraud", "fraud_type",
            "distance_from_home_km", "is_international",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_fraud_rate_approximately_correct(self):
        df = self.gen.generate(save=False)
        target = CONFIG["data"]["fraud_ratio"]
        actual = df["is_fraud"].mean()
        assert abs(actual - target) < 0.01, f"Fraud rate {actual:.3f} far from target {target}"

    def test_amounts_positive(self):
        df = self.gen.generate(save=False)
        assert (df["amount"] > 0).all()

    def test_fraud_types_valid(self):
        df = self.gen.generate(save=False)
        valid_types = {
            "none", "card_testing", "account_takeover",
            "cnp_fraud", "velocity_abuse", "geographic_anomaly"
        }
        assert set(df["fraud_type"].unique()).issubset(valid_types)

    def test_is_fraud_binary(self):
        df = self.gen.generate(save=False)
        assert set(df["is_fraud"].unique()).issubset({0, 1})


# ─────────────────────────────────────────────────────────────────────
# Preprocessor Tests
# ─────────────────────────────────────────────────────────────────────

class TestFraudPreprocessor:
    def setup_method(self):
        cfg = CONFIG.copy()
        cfg["data"]["n_samples"] = 3000
        cfg["data"]["n_users"] = 200
        cfg["data"]["n_merchants"] = 100
        from fraud_detection.data.generator import FraudDataGenerator
        from fraud_detection.data.preprocessor import FraudPreprocessor

        gen = FraudDataGenerator(cfg)
        self.df = gen.generate(save=False)
        self.prep = FraudPreprocessor(cfg)

    def test_fit_transform_returns_dataframe_and_series(self):
        X, y = self.prep.fit_transform(self.df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_no_nan_after_preprocessing(self):
        X, y = self.prep.fit_transform(self.df)
        assert not X.isnull().any().any(), "NaN values found after preprocessing"

    def test_all_columns_numeric(self):
        X, y = self.prep.fit_transform(self.df)
        assert X.select_dtypes(exclude=[np.number]).empty

    def test_target_is_binary(self):
        _, y = self.prep.fit_transform(self.df)
        assert set(y.unique()).issubset({0, 1})

    def test_feature_names_stored(self):
        self.prep.fit_transform(self.df)
        assert len(self.prep.feature_names_) > 5

    def test_transform_new_data(self):
        X_train, y = self.prep.fit_transform(self.df)
        # Transform a subset (simulating inference)
        X_new = self.prep.transform(self.df.iloc[:20])
        assert X_new.shape[1] == X_train.shape[1]
        assert not X_new.isnull().any().any()

    def test_split_returns_six_arrays(self):
        X, y = self.prep.fit_transform(self.df)
        result = self.prep.split(X, y, apply_smote=False)
        assert len(result) == 6


# ─────────────────────────────────────────────────────────────────────
# Predictor Tests (with a simple trained model)
# ─────────────────────────────────────────────────────────────────────

class TestFraudPredictor:
    """Integration test: train a tiny model, then test predictor."""

    @pytest.fixture(autouse=True, scope="class")
    def train_small_model(self, tmp_path_factory):
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from fraud_detection.data.generator import FraudDataGenerator
        from fraud_detection.data.preprocessor import FraudPreprocessor

        cfg = CONFIG.copy()
        cfg["data"]["n_samples"] = 4000
        cfg["data"]["n_users"] = 300
        cfg["data"]["n_merchants"] = 100

        gen = FraudDataGenerator(cfg)
        df = gen.generate(save=False)

        prep = FraudPreprocessor(cfg)
        X, y = prep.fit_transform(df)

        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)

        tmp = tmp_path_factory.mktemp("models")
        model_path = str(tmp / "best_model.pkl")
        prep_path = str(tmp / "preprocessor.pkl")
        joblib.dump(model, model_path)
        prep.save(prep_path)

        self.__class__.model_path = model_path
        self.__class__.prep_path = prep_path
        self.__class__.cfg = cfg

    def _make_predictor(self):
        from fraud_detection.models.predictor import FraudPredictor
        return FraudPredictor(
            model_path=self.model_path,
            preprocessor_path=self.prep_path,
            config=self.cfg,
        )

    def _sample_txn(self, **overrides):
        base = {
            "transaction_id": "TXN_TEST_001",
            "user_id": "USR_000001",
            "merchant_id": "MRC_00001",
            "merchant_category": "retail",
            "timestamp": "2025-06-01 14:30:00",
            "amount": 75.00,
            "transaction_type": "purchase",
            "device_type": "pos_terminal",
            "is_online": 0,
            "card_present": 1,
            "is_international": 0,
            "distance_from_home_km": 5.0,
            "failed_attempts_before": 0,
            "account_age_days": 500,
        }
        base.update(overrides)
        return base

    def test_score_transaction_returns_dict(self):
        pred = self._make_predictor()
        result = pred.score_transaction(self._sample_txn())
        assert isinstance(result, dict)

    def test_fraud_probability_in_range(self):
        pred = self._make_predictor()
        result = pred.score_transaction(self._sample_txn())
        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_fraud_prediction_binary(self):
        pred = self._make_predictor()
        result = pred.score_transaction(self._sample_txn())
        assert result["fraud_prediction"] in (0, 1)

    def test_risk_tier_valid(self):
        pred = self._make_predictor()
        result = pred.score_transaction(self._sample_txn())
        assert result["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")

    def test_latency_ms_positive(self):
        pred = self._make_predictor()
        result = pred.score_transaction(self._sample_txn())
        assert result["latency_ms"] > 0

    def test_top_risk_factors_list(self):
        pred = self._make_predictor()
        result = pred.score_transaction(self._sample_txn())
        assert isinstance(result["top_risk_factors"], list)

    def test_high_risk_transaction_gets_higher_score(self):
        pred = self._make_predictor()
        legit = pred.score_transaction(self._sample_txn())
        suspicious = pred.score_transaction(self._sample_txn(
            amount=3500.0,
            is_international=1,
            distance_from_home_km=5000,
            failed_attempts_before=4,
            merchant_category="crypto_exchange",
            device_type="web_browser",
            is_online=1,
            card_present=0,
        ))
        assert suspicious["fraud_probability"] >= legit["fraud_probability"]

    def test_batch_scoring(self):
        pred = self._make_predictor()
        batch = [self._sample_txn(transaction_id=f"TXN_{i:03d}") for i in range(10)]
        results = pred.score_batch(batch)
        assert len(results) == 10
        for r in results:
            assert 0 <= r["fraud_probability"] <= 1


# ─────────────────────────────────────────────────────────────────────
# API Tests
# ─────────────────────────────────────────────────────────────────────

class TestFraudAPI:
    """Tests FastAPI endpoints using TestClient (no running server needed)."""

    @pytest.fixture(autouse=True, scope="class")
    def setup_api(self):
        """Skip API tests if model artefacts are not available."""
        model_path = CONFIG["api"]["model_path"]
        if not os.path.exists(model_path):
            pytest.skip("Model artefacts not found — run train_models.py first")

    def _client(self):
        from fastapi.testclient import TestClient
        from fraud_detection.api.server import app
        return TestClient(app)

    def test_health_endpoint(self):
        client = self._client()
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"

    def test_predict_endpoint_valid_transaction(self):
        client = self._client()
        payload = {
            "transaction_id": "TXN_API_001",
            "user_id": "USR_000001",
            "merchant_id": "MRC_00001",
            "merchant_category": "retail",
            "timestamp": "2025-06-01 14:30:00",
            "amount": 75.0,
            "transaction_type": "purchase",
            "device_type": "pos_terminal",
            "is_online": 0,
            "card_present": 1,
            "is_international": 0,
            "distance_from_home_km": 5.0,
            "failed_attempts_before": 0,
            "account_age_days": 500,
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        body = r.json()
        assert "fraud_probability" in body
        assert "risk_tier" in body
        assert 0 <= body["fraud_probability"] <= 1

    def test_predict_invalid_amount_rejected(self):
        client = self._client()
        payload = {
            "transaction_id": "TXN_BAD",
            "user_id": "USR_000001",
            "merchant_id": "MRC_00001",
            "merchant_category": "retail",
            "timestamp": "2025-06-01 14:30:00",
            "amount": -50.0,   # INVALID
            "transaction_type": "purchase",
            "device_type": "pos_terminal",
        }
        r = client.post("/predict", json=payload)
        assert r.status_code == 422  # Validation error

    def test_batch_endpoint(self):
        client = self._client()
        txn = {
            "transaction_id": "TXN_B001",
            "user_id": "USR_000001",
            "merchant_id": "MRC_00001",
            "merchant_category": "retail",
            "timestamp": "2025-06-01 14:30:00",
            "amount": 75.0,
            "transaction_type": "purchase",
            "device_type": "pos_terminal",
        }
        r = client.post("/predict/batch", json={"transactions": [txn, txn]})
        assert r.status_code == 200
        body = r.json()
        assert body["total"] == 2
        assert "flagged" in body


# ─────────────────────────────────────────────────────────────────────
# Pipeline Tests
# ─────────────────────────────────────────────────────────────────────

class TestPipelineStats:
    def test_initial_state(self):
        from fraud_detection.pipeline.stream_processor import PipelineStats
        stats = PipelineStats()
        assert stats.total_processed == 0
        assert stats.fraud_rate == 0.0

    def test_record_updates_counts(self):
        from fraud_detection.pipeline.stream_processor import PipelineStats
        stats = PipelineStats()
        stats.record(0.9, 1, 5.0)
        stats.record(0.1, 0, 2.0)
        assert stats.total_processed == 2
        assert stats.total_flagged == 1
        assert abs(stats.fraud_rate - 0.5) < 0.01

    def test_summary_returns_dict(self):
        from fraud_detection.pipeline.stream_processor import PipelineStats
        stats = PipelineStats()
        stats.record(0.5, 1, 3.0)
        summary = stats.summary()
        assert isinstance(summary, dict)
        assert "throughput_tps" in summary


class TestAlertManager:
    def test_raise_and_retrieve_alert(self):
        from fraud_detection.pipeline.stream_processor import AlertManager
        am = AlertManager()
        am.raise_alert({
            "transaction_id": "TXN_001",
            "fraud_probability": 0.95,
            "risk_tier": "CRITICAL",
            "risk_description": "Test alert",
            "top_risk_factors": ["test factor"],
        })
        alerts = am.get_recent(10)
        assert len(alerts) == 1
        assert alerts[0]["transaction_id"] == "TXN_001"

    def test_max_alerts_respected(self):
        from fraud_detection.pipeline.stream_processor import AlertManager
        am = AlertManager(max_alerts=5)
        for i in range(10):
            am.raise_alert({"transaction_id": f"T{i}", "fraud_probability": 0.9,
                            "risk_tier": "HIGH", "risk_description": "x", "top_risk_factors": []})
        assert len(am.get_recent(100)) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
