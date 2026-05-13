"""
Real-Time Stream Processing Pipeline

Simulates a streaming fraud detection pipeline that:
  1. Ingests transactions from an in-memory queue
  2. Applies feature engineering
  3. Scores each transaction with the trained model
  4. Emits alerts for high-risk transactions
  5. Maintains a rolling statistics store
"""

import asyncio
import queue
import threading
import time
import uuid
import random
from datetime import datetime
from collections import deque
from typing import Optional, Callable, List, Dict, Any
import yaml
import os
import numpy as np
import pandas as pd

from fraud_detection.models.predictor import FraudPredictor, _risk_tier


class TransactionQueue:
    """Thread-safe transaction ingestion queue."""

    def __init__(self, maxsize: int = 10000):
        self._q = queue.Queue(maxsize=maxsize)

    def put(self, transaction: Dict[str, Any], block: bool = True) -> bool:
        try:
            self._q.put(transaction, block=block, timeout=1.0)
            return True
        except queue.Full:
            return False

    def get(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def size(self) -> int:
        return self._q.qsize()


class AlertManager:
    """
    Manages fraud alerts and maintains an alert log.
    """

    def __init__(self, max_alerts: int = 1000):
        self._alerts: deque = deque(maxlen=max_alerts)
        self._lock = threading.Lock()

    def raise_alert(self, result: Dict[str, Any]):
        alert = {
            "alert_id": str(uuid.uuid4())[:8],
            "timestamp": datetime.utcnow().isoformat(),
            "transaction_id": result.get("transaction_id"),
            "fraud_probability": result.get("fraud_probability"),
            "risk_tier": result.get("risk_tier"),
            "risk_description": result.get("risk_description"),
            "risk_factors": result.get("top_risk_factors", []),
        }
        with self._lock:
            self._alerts.appendleft(alert)

        tier = alert["risk_tier"]
        print(f"  🚨 ALERT [{tier}] txn={alert['transaction_id']} "
              f"prob={alert['fraud_probability']:.4f}")

    def get_recent(self, n: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._alerts)[:n]

    @property
    def total_alerts(self) -> int:
        with self._lock:
            return len(self._alerts)


class PipelineStats:
    """Rolling statistics for pipeline monitoring."""

    def __init__(self, window: int = 1000):
        self._window = window
        self._scores: deque = deque(maxlen=window)
        self._labels: deque = deque(maxlen=window)
        self._latencies: deque = deque(maxlen=window)
        self._lock = threading.Lock()
        self.total_processed: int = 0
        self.total_flagged: int = 0
        self.start_time: float = time.time()

    def record(self, score: float, label: int, latency_ms: float):
        with self._lock:
            self._scores.append(score)
            self._labels.append(label)
            self._latencies.append(latency_ms)
            self.total_processed += 1
            self.total_flagged += label

    @property
    def fraud_rate(self) -> float:
        with self._lock:
            if not self._labels:
                return 0.0
            return sum(self._labels) / len(self._labels)

    @property
    def avg_score(self) -> float:
        with self._lock:
            return float(np.mean(self._scores)) if self._scores else 0.0

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            return float(np.mean(self._latencies)) if self._latencies else 0.0

    @property
    def throughput_tps(self) -> float:
        elapsed = time.time() - self.start_time
        return self.total_processed / max(elapsed, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_processed": self.total_processed,
            "total_flagged": self.total_flagged,
            "fraud_rate_rolling": round(self.fraud_rate * 100, 3),
            "avg_fraud_score": round(self.avg_score, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "throughput_tps": round(self.throughput_tps, 1),
            "uptime_seconds": round(time.time() - self.start_time, 1),
        }


class RealTimePipeline:
    """
    Real-time fraud detection pipeline.

    Architecture
    ────────────
    TransactionProducer
          ↓
    TransactionQueue (in-memory, thread-safe)
          ↓
    Worker threads: feature engineering → model scoring → alert
          ↓
    AlertManager + PipelineStats
    """

    def __init__(
        self,
        predictor: Optional[FraudPredictor] = None,
        config: Optional[dict] = None,
        on_alert: Optional[Callable] = None,
    ):
        if config is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
        self.cfg = config["pipeline"]
        self.predictor = predictor
        self.on_alert = on_alert  # optional callback for external notification

        self.queue = TransactionQueue(maxsize=self.cfg["max_queue_size"])
        self.alerts = AlertManager()
        self.stats = PipelineStats()
        self._running = False
        self._workers: List[threading.Thread] = []
        self._n_workers = 4

    # ─────────────────────────────────────────────────────────────────
    # Producer
    # ─────────────────────────────────────────────────────────────────

    def ingest(self, transaction: Dict[str, Any]) -> bool:
        """Push a single transaction onto the processing queue."""
        return self.queue.put(transaction)

    def ingest_batch(self, transactions: List[Dict[str, Any]]) -> int:
        """Push a batch and return number successfully enqueued."""
        return sum(self.ingest(t) for t in transactions)

    # ─────────────────────────────────────────────────────────────────
    # Processing worker
    # ─────────────────────────────────────────────────────────────────

    def _process_transaction(self, txn: Dict[str, Any]):
        """Score one transaction and handle alerts."""
        result = self.predictor.score_transaction(txn)
        prob = result["fraud_probability"]
        pred = result["fraud_prediction"]

        self.stats.record(prob, pred, result.get("latency_ms", 0))

        if prob >= self.cfg["alert_threshold"]:
            self.alerts.raise_alert(result)
            if self.on_alert:
                self.on_alert(result)

        return result

    def _worker_loop(self):
        """Worker thread: pull from queue and score."""
        while self._running:
            txn = self.queue.get(timeout=0.5)
            if txn is None:
                continue
            try:
                self._process_transaction(txn)
            except Exception as e:
                print(f"  [Worker error] {e}")

    # ─────────────────────────────────────────────────────────────────
    # Start / stop
    # ─────────────────────────────────────────────────────────────────

    def start(self, n_workers: int = 4):
        """Start the pipeline worker threads."""
        self._running = True
        self._n_workers = n_workers
        for i in range(n_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True,
                                 name=f"fraud-worker-{i}")
            t.start()
            self._workers.append(t)
        print(f"Pipeline started with {n_workers} workers. Queue capacity: {self.cfg['max_queue_size']}")

    def stop(self):
        """Gracefully stop the pipeline."""
        self._running = False
        for t in self._workers:
            t.join(timeout=3)
        self._workers.clear()
        print(f"Pipeline stopped. Stats: {self.stats.summary()}")

    # ─────────────────────────────────────────────────────────────────
    # Simulation helpers
    # ─────────────────────────────────────────────────────────────────

    def simulate_stream(
        self,
        n_transactions: int = 500,
        tps: float = 50.0,
        fraud_inject_rate: float = 0.02,
    ):
        """
        Simulate a live transaction stream.

        Parameters
        ----------
        n_transactions  : total transactions to simulate
        tps             : target transactions per second
        fraud_inject_rate : fraction of transactions to inject as fraud-like
        """
        sleep_interval = 1.0 / tps

        USER_IDS = [f"USR_{i:06d}" for i in range(1000)]
        MERCHANT_IDS = [f"MRC_{i:05d}" for i in range(500)]
        CATS = ["grocery", "gas_station", "restaurant", "retail", "electronics",
                "online_shopping", "jewelry", "travel", "crypto_exchange"]
        DEVICES = ["mobile_app", "web_browser", "pos_terminal", "atm"]
        TXN_TYPES = ["purchase", "online_purchase", "atm_withdrawal", "contactless"]

        print(f"\nSimulating {n_transactions:,} transactions at {tps} TPS...")
        enqueued = 0

        for i in range(n_transactions):
            is_fraud_sim = random.random() < fraud_inject_rate

            if is_fraud_sim:
                txn = {
                    "transaction_id": f"SIM_{i:07d}",
                    "user_id": random.choice(USER_IDS),
                    "merchant_id": random.choice(MERCHANT_IDS),
                    "merchant_category": random.choice(["crypto_exchange", "jewelry", "electronics"]),
                    "timestamp": datetime.utcnow().isoformat(),
                    "amount": round(random.uniform(500, 4000), 2),
                    "transaction_type": "online_purchase",
                    "device_type": "web_browser",
                    "is_online": 1,
                    "card_present": 0,
                    "is_international": random.choice([0, 1]),
                    "distance_from_home_km": round(random.uniform(1000, 8000), 1),
                    "failed_attempts_before": random.randint(0, 4),
                    "account_age_days": random.randint(10, 120),
                    "hour_of_day": random.choice([0, 1, 2, 3, 4]),
                }
            else:
                txn = {
                    "transaction_id": f"SIM_{i:07d}",
                    "user_id": random.choice(USER_IDS),
                    "merchant_id": random.choice(MERCHANT_IDS),
                    "merchant_category": random.choice(CATS),
                    "timestamp": datetime.utcnow().isoformat(),
                    "amount": round(random.lognormal(4.0, 0.8), 2),
                    "transaction_type": random.choice(TXN_TYPES),
                    "device_type": random.choice(DEVICES),
                    "is_online": random.choice([0, 1]),
                    "card_present": random.choice([0, 1]),
                    "is_international": int(random.random() < 0.03),
                    "distance_from_home_km": round(random.expovariate(1/50), 1),
                    "failed_attempts_before": 0,
                    "account_age_days": random.randint(180, 3600),
                    "hour_of_day": random.randint(8, 21),
                }

            if self.ingest(txn):
                enqueued += 1

            time.sleep(sleep_interval)

        # Drain remaining
        drain_timeout = 10
        t0 = time.time()
        while self.queue.size > 0 and time.time() - t0 < drain_timeout:
            time.sleep(0.1)

        print(f"Simulation complete — enqueued: {enqueued}")
        print(f"Stats: {self.stats.summary()}")
        print(f"Alerts raised: {self.alerts.total_alerts}")
