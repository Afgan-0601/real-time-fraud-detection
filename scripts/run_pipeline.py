"""
Run Real-Time Pipeline Simulation

Loads the trained model, starts the pipeline, simulates a live
transaction stream, and prints performance statistics.

Usage
─────
  python scripts/run_pipeline.py
  python scripts/run_pipeline.py --tps 100 --n 1000
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def parse_args():
    p = argparse.ArgumentParser(description="Run Real-Time Fraud Detection Pipeline")
    p.add_argument("--tps", type=float, default=50.0, help="Transactions per second")
    p.add_argument("--n", type=int, default=200, help="Total transactions to simulate")
    p.add_argument("--fraud-rate", type=float, default=0.02, help="Fraud injection rate")
    p.add_argument("--workers", type=int, default=4, help="Number of pipeline workers")
    return p.parse_args()


def main():
    args = parse_args()

    import yaml
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    model_path = config["api"]["model_path"]
    preprocessor_path = "models/artifacts/preprocessor.pkl"

    if not (os.path.exists(model_path) and os.path.exists(preprocessor_path)):
        print("❌ Model artefacts not found.")
        print("   Run `python scripts/train_models.py` first.")
        sys.exit(1)

    print("\n" + "="*60)
    print("  REAL-TIME FRAUD DETECTION — PIPELINE SIMULATION")
    print("="*60)
    print(f"  TPS       : {args.tps}")
    print(f"  Txns      : {args.n}")
    print(f"  Fraud rate: {args.fraud_rate*100:.1f}%")
    print(f"  Workers   : {args.workers}")

    from fraud_detection.models.predictor import FraudPredictor
    from fraud_detection.pipeline.stream_processor import RealTimePipeline

    predictor = FraudPredictor(
        model_path=model_path,
        preprocessor_path=preprocessor_path,
        config=config,
    )

    pipeline = RealTimePipeline(predictor=predictor, config=config)
    pipeline.start(n_workers=args.workers)

    try:
        pipeline.simulate_stream(
            n_transactions=args.n,
            tps=args.tps,
            fraud_inject_rate=args.fraud_rate,
        )
    finally:
        time.sleep(2)  # drain queue
        pipeline.stop()

    print("\nRecent alerts:")
    for alert in pipeline.alerts.get_recent(10):
        print(f"  [{alert['risk_tier']}] {alert['transaction_id']} "
              f"score={alert['fraud_probability']:.4f}")


if __name__ == "__main__":
    main()
