"""
Model Evaluator — Comprehensive Performance Analysis

Generates:
  - Classification report
  - Confusion matrix
  - ROC & Precision-Recall curves
  - Feature importance plot
  - SHAP summary plot
  - Per-fraud-type breakdown
  - HTML/CSV evaluation reports
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Dict, Optional, Any
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve,
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score,
)
import yaml

sns.set_theme(style="whitegrid", palette="muted")


class FraudModelEvaluator:
    """
    Evaluates one or more fraud detection models and produces rich reports.
    """

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            cfg_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
            with open(cfg_path) as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.plots_dir = config["evaluation"]["plots_dir"]
        self.reports_dir = config["evaluation"]["reports_dir"]
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
    # Core metrics
    # ─────────────────────────────────────────────────────────────────

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        y_pred = (y_prob >= threshold).astype(int)
        return {
            "threshold": threshold,
            "accuracy": float((y_true == y_pred).mean()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob)),
            "pr_auc": float(average_precision_score(y_true, y_prob)),
            "false_positive_rate": float(
                confusion_matrix(y_true, y_pred)[0, 1] /
                max((y_true == 0).sum(), 1)
            ),
            "false_negative_rate": float(
                confusion_matrix(y_true, y_pred)[1, 0] /
                max((y_true == 1).sum(), 1)
            ),
        }

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> float:
        """Find threshold that maximises F1 score."""
        thresholds = np.linspace(0.1, 0.9, 161)
        best_f1, best_t = 0.0, 0.5
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            f = f1_score(y_true, y_pred, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        return float(best_t)

    # ─────────────────────────────────────────────────────────────────
    # Visualisations
    # ─────────────────────────────────────────────────────────────────

    def plot_confusion_matrix(
        self, y_true, y_pred, model_name: str = "model"
    ):
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, data, fmt, title in zip(
            axes,
            [cm, cm_norm],
            ["d", ".2%"],
            ["Counts", "Normalised"],
        ):
            sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", ax=ax,
                        xticklabels=["Legit", "Fraud"],
                        yticklabels=["Legit", "Fraud"])
            ax.set_title(f"Confusion Matrix — {title}", fontweight="bold")
            ax.set_ylabel("True label")
            ax.set_xlabel("Predicted label")
        fig.suptitle(f"{model_name.replace('_', ' ').title()}", fontsize=14)
        fig.tight_layout()
        path = os.path.join(self.plots_dir, f"{model_name}_confusion_matrix.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_roc_curves(
        self,
        models_probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
        filename: str = "roc_curves.png",
    ):
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.tab10.colors

        for idx, (name, y_prob) in enumerate(models_probs.items()):
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors[idx % 10], lw=2,
                    label=f"{name.replace('_', ' ').title()} (AUC={roc:.4f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax.set_xlim([-0.01, 1.01])
        ax.set_ylim([-0.01, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=10)
        fig.tight_layout()
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_pr_curves(
        self,
        models_probs: Dict[str, np.ndarray],
        y_true: np.ndarray,
        filename: str = "pr_curves.png",
    ):
        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.tab10.colors
        baseline = y_true.mean()

        for idx, (name, y_prob) in enumerate(models_probs.items()):
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            ax.plot(rec, prec, color=colors[idx % 10], lw=2,
                    label=f"{name.replace('_', ' ').title()} (AP={ap:.4f})")

        ax.axhline(baseline, color="k", linestyle="--", lw=1,
                   label=f"Baseline (fraud rate={baseline:.3f})")
        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=10)
        fig.tight_layout()
        path = os.path.join(self.plots_dir, filename)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_feature_importance(
        self,
        importances: pd.Series,
        model_name: str = "model",
        top_n: int = 25,
    ):
        top = importances.head(top_n).sort_values()
        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top)))
        top.plot(kind="barh", ax=ax, color=colors)
        ax.set_title(f"Top {top_n} Feature Importances — {model_name.replace('_', ' ').title()}",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        path = os.path.join(self.plots_dir, f"{model_name}_feature_importance.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_score_distribution(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "model",
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(y_prob[y_true == 0], bins=80, alpha=0.6, color="steelblue",
                label="Legitimate", density=True)
        ax.hist(y_prob[y_true == 1], bins=80, alpha=0.7, color="crimson",
                label="Fraud", density=True)
        ax.set_xlabel("Fraud Probability Score", fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.set_title(f"Score Distribution — {model_name.replace('_', ' ').title()}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11)
        ax.axvline(0.5, color="orange", linestyle="--", lw=1.5, label="Threshold 0.5")
        fig.tight_layout()
        path = os.path.join(self.plots_dir, f"{model_name}_score_distribution.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_model_comparison(self, scores_df: pd.DataFrame):
        metrics = ["roc_auc", "pr_auc", "f1", "precision", "recall"]
        present = [m for m in metrics if m in scores_df.columns]
        df = scores_df[present].copy()

        fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 5))
        if len(present) == 1:
            axes = [axes]
        colors = plt.cm.Set2.colors

        for ax, metric in zip(axes, present):
            vals = df[metric].sort_values(ascending=True)
            bars = ax.barh(vals.index, vals.values,
                           color=[colors[i % len(colors)] for i in range(len(vals))])
            ax.set_xlim(0, 1.05)
            ax.set_title(metric.replace("_", " ").upper(), fontweight="bold")
            for bar, val in zip(bars, vals.values):
                ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{val:.3f}", va="center", fontsize=9)

        fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")
        fig.tight_layout()
        path = os.path.join(self.plots_dir, "model_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    # ─────────────────────────────────────────────────────────────────
    # SHAP explanations
    # ─────────────────────────────────────────────────────────────────

    def plot_shap_summary(
        self,
        model,
        X_sample: pd.DataFrame,
        model_name: str = "model",
    ):
        try:
            import shap
            print(f"  Computing SHAP values for {model_name}...")

            if hasattr(model, "predict_proba"):
                explainer = shap.TreeExplainer(model) if hasattr(model, "feature_importances_") \
                    else shap.KernelExplainer(model.predict_proba, shap.sample(X_sample, 100))
                shap_values = explainer.shap_values(X_sample)

                # For binary classifiers TreeExplainer returns list[2]
                sv = shap_values[1] if isinstance(shap_values, list) else shap_values

                fig, ax = plt.subplots(figsize=(10, 8))
                shap.summary_plot(sv, X_sample, plot_type="bar", show=False, max_display=20)
                path = os.path.join(self.plots_dir, f"{model_name}_shap_summary.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                plt.close()
                return path
        except Exception as e:
            print(f"  SHAP plot skipped: {e}")
        return None

    # ─────────────────────────────────────────────────────────────────
    # Full evaluation report
    # ─────────────────────────────────────────────────────────────────

    def evaluate_model(
        self,
        model_name: str,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_importances: Optional[pd.Series] = None,
        original_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Run full evaluation and save all artefacts."""
        print(f"\nEvaluating: {model_name}")

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            scores = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(scores * 3))

        # Optimal threshold
        if self.cfg["training"].get("optimal_threshold"):
            threshold = self.find_optimal_threshold(y_test.values, y_prob)
            print(f"  Optimal threshold: {threshold:.3f}")
        else:
            threshold = self.cfg["training"]["threshold"]

        metrics = self.compute_metrics(y_test.values, y_prob, threshold)
        y_pred = (y_prob >= threshold).astype(int)

        print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC    : {metrics['pr_auc']:.4f}")
        print(f"  F1        : {metrics['f1']:.4f}")
        print(f"  Precision : {metrics['precision']:.4f}")
        print(f"  Recall    : {metrics['recall']:.4f}")
        print(f"  FPR       : {metrics['false_positive_rate']:.4f}")
        print(f"  FNR       : {metrics['false_negative_rate']:.4f}")

        # Classification report
        report = classification_report(y_test, y_pred,
                                       target_names=["Legitimate", "Fraud"])
        print(report)

        # Plots
        self.plot_confusion_matrix(y_test, y_pred, model_name)
        self.plot_score_distribution(y_test.values, y_prob, model_name)

        if feature_importances is not None and len(feature_importances) > 0:
            self.plot_feature_importance(feature_importances, model_name)

        # Save report
        report_data = {
            "model": model_name,
            "metrics": metrics,
            "classification_report": report,
        }
        report_path = os.path.join(self.reports_dir, f"{model_name}_report.json")
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        return {"metrics": metrics, "y_prob": y_prob, "threshold": threshold}

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_importances: Optional[Dict[str, pd.Series]] = None,
    ) -> pd.DataFrame:
        """Evaluate all models and produce comparison plots."""
        all_probs = {}
        all_metrics = {}

        for name, model in models.items():
            result = self.evaluate_model(
                name, model, X_test, y_test,
                feature_importances=(feature_importances or {}).get(name),
            )
            all_probs[name] = result["y_prob"]
            all_metrics[name] = result["metrics"]

        # Comparison plots
        self.plot_roc_curves(all_probs, y_test.values)
        self.plot_pr_curves(all_probs, y_test.values)

        scores_df = pd.DataFrame(all_metrics).T
        scores_df.to_csv(os.path.join(self.reports_dir, "all_models_metrics.csv"))
        self.plot_model_comparison(scores_df)

        print(f"\nAll evaluation artefacts saved:")
        print(f"  Plots   → {self.plots_dir}")
        print(f"  Reports → {self.reports_dir}")

        return scores_df
