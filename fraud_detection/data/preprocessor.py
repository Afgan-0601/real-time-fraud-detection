"""
Feature Engineering & Preprocessing Pipeline

Transforms raw transaction records into a rich feature set ready for
ML model training and real-time inference.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import yaml
import os
from typing import Tuple, Optional


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes behaviour-based and aggregate features from raw transactions.

    Features created
    ----------------
    Temporal   : hour_of_day, day_of_week, is_weekend, is_night
    Amount     : amount_log, amount_zscore, amount_to_avg_ratio, amount_deviation
    Velocity   : txn_velocity_1h, txn_velocity_24h, txn_velocity_7d
    Diversity  : unique_merchants_24h, unique_categories_7d
    Risk       : high_risk_merchant, failed_attempts_24h
    Geography  : distance_from_home_km, is_large_distance
    Time gap   : time_since_last_txn_min
    """

    def __init__(self):
        self.user_stats_ = None  # fitted per-user stats

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Per-user historical statistics (from training data only)
        legit = df[df.get("is_fraud", pd.Series(0, index=df.index)) == 0] if "is_fraud" in df.columns else df
        self.user_stats_ = (
            legit.groupby("user_id")["amount"]
            .agg(user_avg_amount="mean", user_std_amount="std", user_txn_count="count")
            .fillna({"user_std_amount": 1.0})
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")

        # ── Temporal features ──────────────────────────────────────────
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 23)).astype(int)
        df["month"] = df["timestamp"].dt.month

        # ── Amount features ────────────────────────────────────────────
        df["amount_log"] = np.log1p(df["amount"])
        df["amount_rounded"] = (df["amount"] % 1 == 0).astype(int)  # round number amounts

        # Per-user amount statistics
        df = df.merge(self.user_stats_, on="user_id", how="left")
        df["user_avg_amount"] = df["user_avg_amount"].fillna(df["amount"])
        df["user_std_amount"] = df["user_std_amount"].fillna(df["amount"] * 0.3 + 1)
        df["amount_to_avg_ratio"] = df["amount"] / (df["user_avg_amount"] + 1e-6)
        df["amount_deviation"] = (df["amount"] - df["user_avg_amount"]) / (df["user_std_amount"] + 1e-6)

        # ── Velocity & diversity features (rolling window per user) ────
        df = df.sort_values(["user_id", "timestamp"])
        df = self._add_velocity_features(df)

        # ── Geographic features ────────────────────────────────────────
        df["is_large_distance"] = (df["distance_from_home_km"] > 200).astype(int)
        df["is_very_large_distance"] = (df["distance_from_home_km"] > 1000).astype(int)

        # ── Time since last transaction ────────────────────────────────
        df["time_since_last_txn_min"] = (
            df.groupby("user_id")["timestamp"]
            .diff()
            .dt.total_seconds()
            .div(60)
            .fillna(9999)
        )
        df["is_rapid_succession"] = (df["time_since_last_txn_min"] < 2).astype(int)

        # ── Drop helper columns ────────────────────────────────────────
        df = df.drop(columns=["user_avg_amount", "user_std_amount", "user_txn_count"],
                     errors="ignore")

        return df

    @staticmethod
    def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
        """Rolling-window velocity counts per user."""
        # Work on a sorted copy
        df = df.copy()
        df = df.sort_values(["user_id", "timestamp"])

        windows = {
            "txn_velocity_1h": pd.Timedelta(hours=1),
            "txn_velocity_24h": pd.Timedelta(hours=24),
            "txn_velocity_7d": pd.Timedelta(days=7),
        }

        for col, window in windows.items():
            counts = []
            for uid, grp in df.groupby("user_id"):
                grp = grp.sort_values("timestamp")
                ts = grp["timestamp"].values
                c = []
                for i, t in enumerate(ts):
                    start = t - np.timedelta64(int(window.total_seconds()), 's')
                    c.append(((ts[:i] >= start) & (ts[:i] < t)).sum())
                counts.extend(c)
            df[col] = counts

        # Unique merchants in 24h
        umerchants = []
        for uid, grp in df.groupby("user_id"):
            grp = grp.sort_values("timestamp")
            ts = grp["timestamp"].values
            mids = grp["merchant_id"].values
            counts = []
            for i, t in enumerate(ts):
                start = t - np.timedelta64(24 * 3600, 's')
                mask = (ts[:i] >= start) & (ts[:i] < t)
                counts.append(len(set(mids[:i][mask])))
            umerchants.extend(counts)
        df["unique_merchants_24h"] = umerchants

        return df


class FraudPreprocessor:
    """
    Full preprocessing pipeline: feature engineering → encoding → scaling → split.
    """

    CATEGORICAL_COLS = ["merchant_category", "transaction_type", "device_type"]
    DROP_COLS = [
        "transaction_id", "user_id", "merchant_id", "merchant_lat",
        "merchant_lon", "fraud_type", "timestamp",
    ]

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)
        self.cfg = config
        self.feature_engineer = TransactionFeatureEngineer()
        self.label_encoders: dict = {}
        self.scaler = StandardScaler()
        self.feature_names_: list = []
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Full fit + transform on training data."""
        print("Running feature engineering...")
        df_fe = self.feature_engineer.fit(df, df["is_fraud"]).transform(df)

        y = df_fe["is_fraud"].astype(int)
        X = df_fe.drop(columns=["is_fraud"] + self.DROP_COLS, errors="ignore")

        # Encode categoricals
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le

        # Fill any remaining NaN
        X = X.fillna(X.median(numeric_only=True))
        X = X.select_dtypes(include=[np.number])

        self.feature_names_ = list(X.columns)

        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=self.feature_names_,
            index=X.index,
        )
        self._fitted = True
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new / inference data using fitted parameters."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        df_fe = self.feature_engineer.transform(df)
        X = df_fe.drop(
            columns=["is_fraud"] + self.DROP_COLS, errors="ignore"
        )
        for col in self.CATEGORICAL_COLS:
            if col in X.columns:
                le = self.label_encoders.get(col)
                if le:
                    known = set(le.classes_)
                    X[col] = X[col].astype(str).apply(
                        lambda v: v if v in known else le.classes_[0]
                    )
                    X[col] = le.transform(X[col])
        X = X.fillna(X.median(numeric_only=True))
        X = X.reindex(columns=self.feature_names_, fill_value=0)
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names_,
            index=X.index,
        )
        return X_scaled

    def split(
        self, X: pd.DataFrame, y: pd.Series, apply_smote: bool = True
    ) -> Tuple:
        """Train/validation/test split with optional SMOTE oversampling."""
        cfg_t = self.cfg["training"]
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=cfg_t["test_size"],
            random_state=cfg_t["random_state"],
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=cfg_t["validation_size"],
            random_state=cfg_t["random_state"],
            stratify=y_trainval,
        )

        if apply_smote and cfg_t.get("smote", True):
            print(f"  Before SMOTE — fraud: {y_train.sum():,} / legit: {(y_train==0).sum():,}")
            smote = SMOTE(random_state=cfg_t["random_state"], k_neighbors=5)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  After  SMOTE — fraud: {y_train.sum():,} / legit: {(y_train==0).sum():,}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def save(self, path: str = "models/artifacts/preprocessor.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
        print(f"Preprocessor saved → {path}")

    @staticmethod
    def load(path: str = "models/artifacts/preprocessor.pkl") -> "FraudPreprocessor":
        return joblib.load(path)


if __name__ == "__main__":
    from fraud_detection.data.generator import FraudDataGenerator

    gen = FraudDataGenerator()
    df = gen.generate(save=False)

    prep = FraudPreprocessor()
    X, y = prep.fit_transform(df)
    print(f"Feature matrix: {X.shape}")
    print(f"Features: {X.columns.tolist()}")
