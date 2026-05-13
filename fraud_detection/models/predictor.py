"""
Real-Time Fraud Predictor

Loads trained model + preprocessor and scores individual transactions
or batches, returning fraud probability, risk tier, and explanation.
"""

import os
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


RISK_TIERS = {
    (0.0, 0.30): ("LOW",      "✅ Transaction appears normal"),
    (0.30, 0.60): ("MEDIUM",  "⚠️  Elevated risk — soft review recommended"),
    (0.60, 0.80): ("HIGH",    "🚨 High fraud risk — manual review required"),
    (0.80, 1.01): ("CRITICAL","🔴 Critical fraud signal — block immediately"),
}


def _risk_tier(prob: float) -> tuple:
    for (lo, hi), (tier, desc) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier, desc
    return "CRITICAL", "🔴 Critical fraud signal — block immediately"


class FraudPredictor:
    """
    Real-time fraud scoring engine.

    Parameters
    ----------
    model_path      : path to the trained sklearn model (.pkl)
    preprocessor_path : path to the fitted FraudPreprocessor (.pkl)
    threshold       : decision threshold (overrides config)
    config          : config dict (loads config.yaml if None)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        preprocessor_path: Optional[str] = None,
        threshold: Optional[float] = None,
        config: Optional[dict] = None,
    ):
        if config is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
        self.cfg = config

        mp = model_path or config["api"]["model_path"]
        pp = preprocessor_path or "models/artifacts/preprocessor.pkl"
        self.threshold = threshold or config["api"]["threshold"]
        self.high_risk_threshold = config["api"].get("high_risk_threshold", 0.8)

        self.model = joblib.load(mp)
        self.preprocessor = joblib.load(pp)
        self._feature_names = self.preprocessor.feature_names_
        print(f"Predictor loaded — model: {mp}  preprocessor: {pp}")

    # ─────────────────────────────────────────────────────────────────
    # Core scoring
    # ─────────────────────────────────────────────────────────────────

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return fraud probabilities for a DataFrame of raw transactions."""
        X = self.preprocessor.transform(df)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        scores = self.model.decision_function(X)
        return 1 / (1 + np.exp(scores * 3))

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(df)
        return (probs >= self.threshold).astype(int)

    # ─────────────────────────────────────────────────────────────────
    # Single-transaction enriched response
    # ─────────────────────────────────────────────────────────────────

    def score_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score a single transaction dict.

        Returns
        -------
        dict with keys:
          transaction_id, fraud_probability, fraud_prediction,
          risk_tier, risk_description, latency_ms, top_risk_factors
        """
        t0 = time.perf_counter()
        df = pd.DataFrame([transaction])

        prob = float(self.predict_proba(df)[0])
        pred = int(prob >= self.threshold)
        tier, desc = _risk_tier(prob)

        risk_factors = self._explain_transaction(transaction, prob)

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        return {
            "transaction_id": transaction.get("transaction_id", "unknown"),
            "fraud_probability": round(prob, 6),
            "fraud_prediction": pred,
            "risk_tier": tier,
            "risk_description": desc,
            "top_risk_factors": risk_factors,
            "latency_ms": latency_ms,
        }

    def score_batch(
        self, transactions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score a list of transactions."""
        t0 = time.perf_counter()
        df = pd.DataFrame(transactions)
        probs = self.predict_proba(df)
        results = []
        for i, (txn, prob) in enumerate(zip(transactions, probs)):
            prob = float(prob)
            pred = int(prob >= self.threshold)
            tier, desc = _risk_tier(prob)
            results.append({
                "transaction_id": txn.get("transaction_id", f"txn_{i}"),
                "fraud_probability": round(prob, 6),
                "fraud_prediction": pred,
                "risk_tier": tier,
                "risk_description": desc,
            })
        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        print(f"Batch scored: {len(transactions)} txns in {total_ms}ms "
              f"({total_ms/len(transactions):.2f}ms/txn)")
        return results

    # ─────────────────────────────────────────────────────────────────
    # Simple rule-based explanation
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _explain_transaction(txn: Dict[str, Any], prob: float) -> List[str]:
        """Produce human-readable risk factor notes."""
        factors = []
        amount = txn.get("amount", 0)
        dist = txn.get("distance_from_home_km", 0)
        velocity_1h = txn.get("txn_velocity_1h", 0)
        velocity_24h = txn.get("txn_velocity_24h", 0)
        is_international = txn.get("is_international", 0)
        failed = txn.get("failed_attempts_before", 0)
        is_night = txn.get("is_night", 0)
        cat = txn.get("merchant_category", "")
        device = txn.get("device_type", "")

        if amount > 1000:
            factors.append(f"High transaction amount: ${amount:.2f}")
        if dist > 500:
            factors.append(f"Far from home location: {dist:.0f} km")
        if is_international:
            factors.append("International transaction")
        if velocity_1h >= 3:
            factors.append(f"High velocity: {velocity_1h} txns in last hour")
        if velocity_24h >= 10:
            factors.append(f"Elevated velocity: {velocity_24h} txns in 24h")
        if failed >= 2:
            factors.append(f"{failed} failed auth attempts before this transaction")
        if is_night:
            factors.append("Transaction at unusual hour (night time)")
        if cat in ("crypto_exchange", "money_transfer", "gambling"):
            factors.append(f"High-risk merchant category: {cat}")
        if device in ("web_browser",) and cat in ("online_shopping", "electronics"):
            factors.append("Online CNP purchase pattern detected")

        if not factors:
            factors.append("Model score elevated — no single dominant factor")

        return factors[:5]  # top 5 factors only


if __name__ == "__main__":
    predictor = FraudPredictor()
    sample = {
        "transaction_id": "TXN_TEST_001",
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
        "account_age_days": 45,
    }
    result = predictor.score_transaction(sample)
    print(result)
