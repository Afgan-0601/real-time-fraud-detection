"""
Synthetic Fraud Transaction Data Generator

Generates realistic credit/debit card transaction data with embedded
fraud patterns including card testing, account takeover, CNP fraud,
and velocity abuse attacks.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import yaml
import os


class FraudDataGenerator:
    """
    Generates realistic synthetic transaction datasets with fraud patterns.
    
    Fraud patterns implemented:
    - Card Testing: multiple small rapid transactions
    - Account Takeover: large purchases in unusual locations
    - Card-Not-Present (CNP): online transactions with mismatched details
    - Velocity Abuse: high-frequency transaction bursts
    - Geographic Anomalies: impossible travel patterns
    """

    MERCHANT_CATEGORIES = [
        "grocery", "gas_station", "restaurant", "retail", "entertainment",
        "travel", "healthcare", "online_shopping", "electronics", "jewelry",
        "atm_withdrawal", "money_transfer", "gambling", "crypto_exchange",
        "subscription", "utility", "insurance", "education"
    ]

    TRANSACTION_TYPES = [
        "purchase", "atm_withdrawal", "online_purchase",
        "contactless", "recurring", "refund"
    ]

    DEVICE_TYPES = ["mobile_app", "web_browser", "pos_terminal", "atm", "phone_order"]

    # Average spend per category (mean, std)
    CATEGORY_SPEND = {
        "grocery": (65, 40), "gas_station": (55, 25), "restaurant": (45, 35),
        "retail": (80, 70), "entertainment": (60, 50), "travel": (350, 300),
        "healthcare": (120, 100), "online_shopping": (90, 80), "electronics": (250, 200),
        "jewelry": (500, 400), "atm_withdrawal": (200, 100), "money_transfer": (500, 400),
        "gambling": (100, 80), "crypto_exchange": (800, 600), "subscription": (15, 10),
        "utility": (100, 50), "insurance": (200, 100), "education": (300, 200),
    }

    def __init__(self, config: Optional[dict] = None):
        if config is None:
            config_path = os.path.join(os.path.dirname(__file__), "../../config.yaml")
            with open(config_path) as f:
                config = yaml.safe_load(f)
        self.cfg = config["data"]
        self.random_state = self.cfg["random_state"]
        np.random.seed(self.random_state)

    def _generate_users(self) -> pd.DataFrame:
        """Create a user population with home locations and spending profiles."""
        n = self.cfg["n_users"]
        users = pd.DataFrame({
            "user_id": [f"USR_{i:06d}" for i in range(n)],
            "home_lat": np.random.uniform(25.0, 48.0, n),
            "home_lon": np.random.uniform(-122.0, -70.0, n),
            "avg_monthly_spend": np.random.lognormal(7.5, 0.8, n),
            "account_age_days": np.random.randint(30, 3650, n),
            "credit_limit": np.random.choice([500, 1000, 2000, 5000, 10000, 25000], n,
                                             p=[0.05, 0.15, 0.25, 0.30, 0.20, 0.05]),
            "risk_score_base": np.random.beta(2, 8, n),  # most users are low risk
        })
        return users.set_index("user_id")

    def _generate_merchants(self) -> pd.DataFrame:
        """Create a merchant registry."""
        n = self.cfg["n_merchants"]
        categories = np.random.choice(self.MERCHANT_CATEGORIES, n)
        merchants = pd.DataFrame({
            "merchant_id": [f"MRC_{i:05d}" for i in range(n)],
            "merchant_category": categories,
            "merchant_lat": np.random.uniform(25.0, 48.0, n),
            "merchant_lon": np.random.uniform(-122.0, -70.0, n),
            "is_online": np.random.choice([0, 1], n, p=[0.7, 0.3]),
            "is_high_risk": np.random.choice([0, 1], n, p=[0.92, 0.08]),
        })
        return merchants.set_index("merchant_id")

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        """Compute great-circle distance in km."""
        R = 6371.0
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlam = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
        return R * 2 * np.arcsin(np.sqrt(a))

    # ------------------------------------------------------------------
    # Legitimate transaction generation
    # ------------------------------------------------------------------

    def _legitimate_transaction(self, txn_id, user_id, user, merchant_id, merchant, ts) -> dict:
        cat = merchant["merchant_category"]
        mean_amt, std_amt = self.CATEGORY_SPEND[cat]
        amount = max(0.5, np.random.lognormal(np.log(mean_amt), 0.5))

        dist_km = self._haversine_km(
            user["home_lat"], user["home_lon"],
            merchant["merchant_lat"], merchant["merchant_lon"]
        )

        return {
            "transaction_id": txn_id,
            "user_id": user_id,
            "merchant_id": merchant_id,
            "merchant_category": cat,
            "timestamp": ts,
            "amount": round(amount, 2),
            "transaction_type": np.random.choice(
                self.TRANSACTION_TYPES, p=[0.45, 0.08, 0.20, 0.15, 0.10, 0.02]
            ),
            "device_type": np.random.choice(
                self.DEVICE_TYPES, p=[0.30, 0.25, 0.35, 0.07, 0.03]
            ),
            "is_online": int(merchant["is_online"]),
            "card_present": int(not merchant["is_online"]),
            "is_international": int(np.random.random() < 0.03),
            "merchant_lat": merchant["merchant_lat"],
            "merchant_lon": merchant["merchant_lon"],
            "distance_from_home_km": round(dist_km, 2),
            "failed_attempts_before": int(np.random.choice([0, 1, 2], p=[0.95, 0.04, 0.01])),
            "is_fraud": 0,
            "fraud_type": "none",
        }

    # ------------------------------------------------------------------
    # Fraud transaction generation patterns
    # ------------------------------------------------------------------

    def _card_testing_fraud(self, base_txn, user, merchant) -> dict:
        """Small amounts, rapid succession - testing stolen card validity."""
        txn = base_txn.copy()
        txn["amount"] = round(np.random.uniform(0.01, 5.00), 2)
        txn["transaction_type"] = "online_purchase"
        txn["device_type"] = "web_browser"
        txn["is_online"] = 1
        txn["card_present"] = 0
        txn["merchant_category"] = np.random.choice(["subscription", "charity"])
        txn["is_fraud"] = 1
        txn["fraud_type"] = "card_testing"
        return txn

    def _account_takeover_fraud(self, base_txn, user, merchant) -> dict:
        """Large purchase in unusual location after account compromise."""
        txn = base_txn.copy()
        txn["amount"] = round(np.random.uniform(500, 5000), 2)
        # Place transaction far from home
        angle = np.random.uniform(0, 2 * np.pi)
        dist_deg = np.random.uniform(15, 40)
        txn["merchant_lat"] = user["home_lat"] + dist_deg * np.sin(angle)
        txn["merchant_lon"] = user["home_lon"] + dist_deg * np.cos(angle)
        txn["distance_from_home_km"] = round(
            self._haversine_km(user["home_lat"], user["home_lon"],
                               txn["merchant_lat"], txn["merchant_lon"]), 2
        )
        txn["merchant_category"] = np.random.choice(["jewelry", "electronics"])
        txn["is_international"] = int(np.random.random() < 0.4)
        txn["device_type"] = np.random.choice(["mobile_app", "web_browser"])
        txn["failed_attempts_before"] = np.random.randint(1, 5)
        txn["is_fraud"] = 1
        txn["fraud_type"] = "account_takeover"
        return txn

    def _cnp_fraud(self, base_txn, user, merchant) -> dict:
        """Card-not-present fraud: online purchase with stolen card details."""
        txn = base_txn.copy()
        txn["amount"] = round(np.random.uniform(100, 2000), 2)
        txn["transaction_type"] = "online_purchase"
        txn["device_type"] = np.random.choice(["web_browser", "mobile_app"])
        txn["is_online"] = 1
        txn["card_present"] = 0
        txn["merchant_category"] = np.random.choice(["online_shopping", "electronics", "travel"])
        txn["is_fraud"] = 1
        txn["fraud_type"] = "cnp_fraud"
        return txn

    def _velocity_abuse_fraud(self, base_txn, user, merchant) -> dict:
        """Many transactions in a short window - smurfing / money mule."""
        txn = base_txn.copy()
        txn["amount"] = round(np.random.uniform(50, 500), 2)
        txn["merchant_category"] = np.random.choice(["money_transfer", "crypto_exchange"])
        txn["transaction_type"] = "online_purchase"
        txn["is_online"] = 1
        txn["card_present"] = 0
        txn["is_fraud"] = 1
        txn["fraud_type"] = "velocity_abuse"
        return txn

    def _geographic_anomaly_fraud(self, base_txn, user, merchant) -> dict:
        """Impossible travel - two transactions in impossible geographic sequence."""
        txn = base_txn.copy()
        txn["amount"] = round(np.random.uniform(200, 3000), 2)
        txn["is_international"] = 1
        txn["merchant_lat"] = np.random.uniform(-50, 70)
        txn["merchant_lon"] = np.random.uniform(-180, 180)
        txn["distance_from_home_km"] = round(
            self._haversine_km(user["home_lat"], user["home_lon"],
                               txn["merchant_lat"], txn["merchant_lon"]), 2
        )
        txn["is_fraud"] = 1
        txn["fraud_type"] = "geographic_anomaly"
        return txn

    # ------------------------------------------------------------------
    # Main generation method
    # ------------------------------------------------------------------

    def generate(self, save: bool = True) -> pd.DataFrame:
        """Generate the full synthetic transaction dataset."""
        print(f"Generating {self.cfg['n_samples']:,} transactions...")

        users = self._generate_users()
        merchants = self._generate_merchants()

        n_total = self.cfg["n_samples"]
        fraud_ratio = self.cfg["fraud_ratio"]
        n_fraud = int(n_total * fraud_ratio)
        n_legit = n_total - n_fraud

        # --- Time range: last 365 days ---
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)

        def random_timestamp():
            # Simulate realistic transaction time distribution
            day_offset = np.random.exponential(30)  # more recent txns more common
            day_offset = min(day_offset, 365)
            ts = end_date - timedelta(days=day_offset)
            # Peak hours: 8am-9pm
            hour_weights = np.array([
                0.005, 0.003, 0.002, 0.002, 0.003, 0.010,
                0.025, 0.055, 0.065, 0.070, 0.072, 0.075,
                0.072, 0.068, 0.065, 0.062, 0.060, 0.062,
                0.060, 0.055, 0.050, 0.040, 0.030, 0.018
            ], dtype=float)
            hour_weights /= hour_weights.sum()  # normalize to exactly 1.0
            hour = int(np.random.choice(range(24), p=hour_weights))
            return ts.replace(hour=hour, minute=np.random.randint(0, 60),
                              second=np.random.randint(0, 60))

        records = []
        user_ids = list(users.index)
        merchant_ids = list(merchants.index)

        # Legitimate transactions
        print("  Generating legitimate transactions...")
        for i in range(n_legit):
            uid = np.random.choice(user_ids)
            mid = np.random.choice(merchant_ids)
            ts = random_timestamp()
            txn = self._legitimate_transaction(
                f"TXN_{i:08d}", uid, users.loc[uid], mid, merchants.loc[mid], ts
            )
            records.append(txn)

        # Fraud transactions
        print(f"  Generating {n_fraud:,} fraud transactions...")
        fraud_patterns = {
            "card_testing": self._card_testing_fraud,
            "account_takeover": self._account_takeover_fraud,
            "cnp_fraud": self._cnp_fraud,
            "velocity_abuse": self._velocity_abuse_fraud,
            "geographic_anomaly": self._geographic_anomaly_fraud,
        }
        pattern_names = list(fraud_patterns.keys())
        # Distribution: CNP most common, geo anomaly rarest
        pattern_probs = [0.25, 0.20, 0.30, 0.15, 0.10]

        for i in range(n_fraud):
            uid = np.random.choice(user_ids)
            mid = np.random.choice(merchant_ids)
            ts = random_timestamp()

            base = self._legitimate_transaction(
                f"TXN_{n_legit + i:08d}", uid, users.loc[uid],
                mid, merchants.loc[mid], ts
            )
            pattern = np.random.choice(pattern_names, p=pattern_probs)
            txn = fraud_patterns[pattern](base, users.loc[uid], merchants.loc[mid])
            records.append(txn)

        df = pd.DataFrame(records)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)

        # Add account age from users
        df["account_age_days"] = df["user_id"].map(users["account_age_days"])

        print(f"  Total records: {len(df):,}")
        print(f"  Fraud rate:    {df['is_fraud'].mean()*100:.2f}%")
        print(f"  Fraud types:   {df[df.is_fraud==1].fraud_type.value_counts().to_dict()}")

        if save:
            os.makedirs(os.path.dirname(self.cfg["raw_path"]), exist_ok=True)
            df.to_csv(self.cfg["raw_path"], index=False)
            print(f"  Saved to: {self.cfg['raw_path']}")

        return df


if __name__ == "__main__":
    gen = FraudDataGenerator()
    df = gen.generate(save=True)
    print(df.head())
