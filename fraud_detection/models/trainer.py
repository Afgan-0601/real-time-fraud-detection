"""
Multi-Model Training Pipeline for Fraud Detection

Trains, tunes, and persists six diverse classifiers:
  1. Logistic Regression  (linear baseline)
  2. Random Forest        (bagging ensemble)
  3. XGBoost              (gradient boosting)
  4. LightGBM             (fast gradient boosting)
  5. Neural Network       (MLP)
  6. Isolation Forest     (unsupervised anomaly detection)

Plus a Soft-Voting Ensemble of the top supervised models.
"""

import os
import time
import yaml
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class FraudModelTrainer:
    """
    Orchestrates training of all fraud detection models.

    Usage
    -----
    trainer = FraudModelTrainer()
    trainer.train_all(X_train, y_train, X_val, y_val)
    trainer.save_all()
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.mcfg = config["models"]
        self.tcfg = config["training"]
        self.models: Dict[str, Any] = {}
        self.val_scores: Dict[str, Dict[str, float]] = {}
        self.best_model_name: str = ""
        self.feature_importances_: Dict[str, pd.Series] = {}

    # ─────────────────────────────────────────────────────────────────
    # Model definitions
    # ─────────────────────────────────────────────────────────────────

    def _build_models(self) -> Dict[str, Any]:
        mc = self.mcfg
        return {
            "logistic_regression": LogisticRegression(
                C=mc["logistic_regression"]["C"],
                max_iter=mc["logistic_regression"]["max_iter"],
                class_weight=mc["logistic_regression"]["class_weight"],
                solver="saga",
                n_jobs=-1,
                random_state=self.tcfg["random_state"],
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=mc["random_forest"]["n_estimators"],
                max_depth=mc["random_forest"]["max_depth"],
                min_samples_split=mc["random_forest"]["min_samples_split"],
                class_weight=mc["random_forest"]["class_weight"],
                n_jobs=-1,
                random_state=mc["random_forest"]["random_state"],
            ),
            "xgboost": XGBClassifier(
                n_estimators=mc["xgboost"]["n_estimators"],
                max_depth=mc["xgboost"]["max_depth"],
                learning_rate=mc["xgboost"]["learning_rate"],
                subsample=mc["xgboost"]["subsample"],
                colsample_bytree=mc["xgboost"]["colsample_bytree"],
                scale_pos_weight=mc["xgboost"]["scale_pos_weight"],
                random_state=mc["xgboost"]["random_state"],
                eval_metric="aucpr",
                verbosity=0,
                n_jobs=-1,
            ),
            "lightgbm": LGBMClassifier(
                n_estimators=mc["lightgbm"]["n_estimators"],
                max_depth=mc["lightgbm"]["max_depth"],
                learning_rate=mc["lightgbm"]["learning_rate"],
                num_leaves=mc["lightgbm"]["num_leaves"],
                class_weight=mc["lightgbm"]["class_weight"],
                random_state=mc["lightgbm"]["random_state"],
                n_jobs=-1,
                verbose=-1,
            ),
            "neural_network": MLPClassifier(
                hidden_layer_sizes=tuple(mc["neural_network"]["hidden_layer_sizes"]),
                activation=mc["neural_network"]["activation"],
                max_iter=mc["neural_network"]["max_iter"],
                early_stopping=True,
                validation_fraction=0.1,
                random_state=mc["neural_network"]["random_state"],
            ),
        }

    # ─────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────

    def train_single(
        self,
        name: str,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        cv: bool = True,
    ) -> Dict[str, float]:
        """Train one model and return validation metrics."""
        print(f"\n  [{name}] Training...")
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob >= self.tcfg["threshold"]).astype(int)

        from sklearn.metrics import (
            precision_score, recall_score, f1_score, roc_auc_score,
            average_precision_score,
        )
        scores = {
            "roc_auc": roc_auc_score(y_val, y_prob),
            "pr_auc": average_precision_score(y_val, y_prob),
            "f1": f1_score(y_val, y_pred, zero_division=0),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred, zero_division=0),
            "train_time_s": elapsed,
        }

        if cv:
            skf = StratifiedKFold(n_splits=self.tcfg["cross_validation_folds"],
                                  shuffle=True, random_state=self.tcfg["random_state"])
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=skf, scoring="roc_auc", n_jobs=-1
            )
            scores["cv_roc_auc_mean"] = cv_scores.mean()
            scores["cv_roc_auc_std"] = cv_scores.std()

        self.models[name] = model
        self.val_scores[name] = scores

        print(f"    ROC-AUC : {scores['roc_auc']:.4f}")
        print(f"    PR-AUC  : {scores['pr_auc']:.4f}")
        print(f"    F1      : {scores['f1']:.4f}")
        print(f"    Precision / Recall: {scores['precision']:.4f} / {scores['recall']:.4f}")
        print(f"    Time    : {elapsed:.1f}s")

        # Store feature importances
        self._extract_feature_importances(name, model, X_train.columns.tolist())
        return scores

    def train_isolation_forest(
        self, X_train: pd.DataFrame, X_val: pd.DataFrame, y_val: pd.Series
    ) -> Dict[str, float]:
        """Train Isolation Forest (unsupervised). Scores inverted so higher = more fraudulent."""
        print("\n  [isolation_forest] Training...")
        ifc = self.mcfg["isolation_forest"]
        iso = IsolationForest(
            n_estimators=ifc["n_estimators"],
            contamination=ifc["contamination"],
            random_state=ifc["random_state"],
            n_jobs=-1,
        )
        t0 = time.time()
        iso.fit(X_train)
        elapsed = time.time() - t0

        # Convert decision_function scores: lower = more anomalous → invert
        scores_raw = iso.decision_function(X_val)
        y_prob = 1 / (1 + np.exp(scores_raw * 3))  # sigmoid inversion

        from sklearn.metrics import roc_auc_score, average_precision_score
        scores = {
            "roc_auc": roc_auc_score(y_val, y_prob),
            "pr_auc": average_precision_score(y_val, y_prob),
            "train_time_s": elapsed,
        }
        self.models["isolation_forest"] = iso
        self.val_scores["isolation_forest"] = scores
        print(f"    ROC-AUC : {scores['roc_auc']:.4f}")
        print(f"    PR-AUC  : {scores['pr_auc']:.4f}")
        print(f"    Time    : {elapsed:.1f}s")
        return scores

    def build_ensemble(self, X_val: pd.DataFrame, y_val: pd.Series):
        """Soft-voting ensemble of top supervised models."""
        supervised = {
            k: v for k, v in self.models.items()
            if k != "isolation_forest"
            and hasattr(v, "predict_proba")
        }
        estimators = list(supervised.items())
        ensemble = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        # VotingClassifier needs refitting
        print("\n  [ensemble] Building soft-voting ensemble (requires refit)...")
        # We'll fit lazily during training script; store components only
        self.models["ensemble"] = ensemble
        return ensemble

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ):
        """Train every model."""
        print("\n" + "="*60)
        print("  FRAUD DETECTION MODEL TRAINING")
        print("="*60)
        print(f"  Train size : {len(X_train):,}  (fraud: {y_train.sum():,})")
        print(f"  Val size   : {len(X_val):,}   (fraud: {y_val.sum():,})")
        print(f"  Features   : {X_train.shape[1]}")

        defined = self._build_models()
        for name, model in defined.items():
            self.train_single(name, model, X_train, y_train, X_val, y_val, cv=False)

        self.train_isolation_forest(X_train, X_val, y_val)

        # Pick best by PR-AUC (more relevant for imbalanced data)
        supervised = {
            k: v for k, v in self.val_scores.items()
            if k != "isolation_forest" and "pr_auc" in v
        }
        self.best_model_name = max(supervised, key=lambda k: supervised[k]["pr_auc"])
        print(f"\n  Best model: {self.best_model_name} "
              f"(PR-AUC={supervised[self.best_model_name]['pr_auc']:.4f})")

        self._print_leaderboard()

    def _print_leaderboard(self):
        print("\n" + "="*60)
        print("  LEADERBOARD (sorted by PR-AUC)")
        print("="*60)
        rows = []
        for name, s in self.val_scores.items():
            rows.append({
                "model": name,
                "ROC-AUC": round(s.get("roc_auc", 0), 4),
                "PR-AUC": round(s.get("pr_auc", 0), 4),
                "F1": round(s.get("f1", 0), 4),
                "Precision": round(s.get("precision", 0), 4),
                "Recall": round(s.get("recall", 0), 4),
                "Time(s)": round(s.get("train_time_s", 0), 1),
            })
        lb = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
        print(lb.to_string(index=False))

    # ─────────────────────────────────────────────────────────────────
    # Feature importances
    # ─────────────────────────────────────────────────────────────────

    def _extract_feature_importances(self, name: str, model, feature_names: list):
        if hasattr(model, "feature_importances_"):
            self.feature_importances_[name] = pd.Series(
                model.feature_importances_, index=feature_names
            ).sort_values(ascending=False)
        elif hasattr(model, "coef_"):
            self.feature_importances_[name] = pd.Series(
                np.abs(model.coef_[0]), index=feature_names
            ).sort_values(ascending=False)

    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.Series:
        name = model_name or self.best_model_name
        return self.feature_importances_.get(name, pd.Series(dtype=float))

    # ─────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────

    def save_all(self, output_dir: Optional[str] = None):
        out = output_dir or self.mcfg["output_dir"]
        os.makedirs(out, exist_ok=True)

        for name, model in self.models.items():
            path = os.path.join(out, f"{name}.pkl")
            joblib.dump(model, path)

        # Save leaderboard
        scores_df = pd.DataFrame(self.val_scores).T
        scores_df.to_csv(os.path.join(out, "leaderboard.csv"))

        # Save best model alias
        best_path = os.path.join(out, "best_model.pkl")
        joblib.dump(self.models[self.best_model_name], best_path)
        print(f"\nAll models saved to: {out}")
        print(f"Best model ({self.best_model_name}) → {best_path}")

    def save_model(self, name: str, output_dir: Optional[str] = None):
        out = output_dir or self.mcfg["output_dir"]
        os.makedirs(out, exist_ok=True)
        path = os.path.join(out, f"{name}.pkl")
        joblib.dump(self.models[name], path)
        return path

    @staticmethod
    def load_model(name: str, output_dir: str = "models/artifacts"):
        path = os.path.join(output_dir, f"{name}.pkl")
        return joblib.load(path)
