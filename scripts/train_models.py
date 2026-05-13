"""
Full Training Pipeline Script

Runs end-to-end:
  1. Generate synthetic transaction data
  2. Feature engineering & preprocessing
  3. Train all models (LR, RF, XGBoost, LightGBM, NN, IsoForest)
  4. Evaluate on held-out test set
  5. Save all artefacts

Usage
─────
  cd real-time-fraud-detection
  python scripts/train_models.py
  python scripts/train_models.py --quick     # smaller dataset for quick test
  python scripts/train_models.py --skip-gen  # reuse existing data/raw/transactions.csv
"""

import argparse
import os
import sys
import time

# Make sure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fraud Detection Models")
    parser.add_argument("--quick", action="store_true",
                        help="Generate smaller dataset for quick testing")
    parser.add_argument("--skip-gen", action="store_true",
                        help="Skip data generation, use existing CSV")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────────────
    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.quick:
        config["data"]["n_samples"] = 50_000
        config["training"]["cross_validation_folds"] = 3
        config["models"]["random_forest"]["n_estimators"] = 100
        config["models"]["xgboost"]["n_estimators"] = 100
        config["models"]["lightgbm"]["n_estimators"] = 100
        config["models"]["neural_network"]["max_iter"] = 50
        print("⚡ QUICK mode: reduced dataset and model sizes")

    print("\n" + "="*60)
    print("  REAL-TIME FRAUD DETECTION — TRAINING PIPELINE")
    print("="*60)
    t_start = time.time()

    # ── Step 1: Data Generation ───────────────────────────────────────
    from fraud_detection.data.generator import FraudDataGenerator

    raw_path = config["data"]["raw_path"]
    if args.skip_gen and os.path.exists(raw_path):
        print(f"\n[1/4] Loading existing data from {raw_path}...")
        df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    else:
        print("\n[1/4] Generating synthetic transaction data...")
        gen = FraudDataGenerator(config)
        df = gen.generate(save=True)

    print(f"  Dataset: {len(df):,} transactions, "
          f"{df['is_fraud'].sum():,} fraud ({df['is_fraud'].mean()*100:.2f}%)")

    # ── Step 2: Feature Engineering & Preprocessing ───────────────────
    print("\n[2/4] Feature engineering & preprocessing...")
    from fraud_detection.data.preprocessor import FraudPreprocessor

    prep = FraudPreprocessor(config)
    X, y = prep.fit_transform(df)
    print(f"  Feature matrix: {X.shape}")
    print(f"  Features: {X.columns.tolist()[:10]} ...")

    X_train, X_val, X_test, y_train, y_val, y_test = prep.split(X, y, apply_smote=True)

    os.makedirs(os.path.dirname(config["data"]["train_path"]), exist_ok=True)
    train_df = X_train.copy(); train_df["is_fraud"] = y_train.values
    test_df  = X_test.copy();  test_df["is_fraud"]  = y_test.values
    train_df.to_csv(config["data"]["train_path"], index=False)
    test_df.to_csv(config["data"]["test_path"], index=False)

    prep.save("models/artifacts/preprocessor.pkl")

    # ── Step 3: Model Training ────────────────────────────────────────
    print("\n[3/4] Training models...")
    from fraud_detection.models.trainer import FraudModelTrainer

    trainer = FraudModelTrainer(config)
    trainer.train_all(X_train, y_train, X_val, y_val)
    trainer.save_all(config["models"]["output_dir"])

    # ── Step 4: Evaluation ───────────────────────────────────────────
    print("\n[4/4] Evaluating on test set...")
    from fraud_detection.models.evaluator import FraudModelEvaluator

    evaluator = FraudModelEvaluator(config)
    scores_df = evaluator.evaluate_all(
        models=trainer.models,
        X_test=X_test,
        y_test=y_test,
        feature_importances=trainer.feature_importances_,
    )

    # ── Summary ──────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    print("\n" + "="*60)
    print(f"  TRAINING COMPLETE in {elapsed/60:.1f} min")
    print("="*60)
    best = trainer.best_model_name
    bs = scores_df.loc[best] if best in scores_df.index else {}
    print(f"  Best model : {best}")
    if len(bs):
        print(f"  ROC-AUC    : {bs.get('roc_auc', 'N/A')}")
        print(f"  PR-AUC     : {bs.get('pr_auc', 'N/A')}")
        print(f"  F1         : {bs.get('f1', 'N/A')}")
    print()
    print("  Artefacts saved:")
    print(f"    Models    → {config['models']['output_dir']}/")
    print(f"    Plots     → {config['evaluation']['plots_dir']}/")
    print(f"    Reports   → {config['evaluation']['reports_dir']}/")
    print()
    print("  To launch the API:")
    print("    uvicorn fraud_detection.api.server:app --host 0.0.0.0 --port 8000")
    print()
    print("  To launch the dashboard:")
    print("    streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
