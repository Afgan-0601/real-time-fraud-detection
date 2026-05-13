"""
Streamlit Real-Time Fraud Detection Dashboard

Sections
────────
1. Overview KPIs      — Total txns, fraud rate, avg score, throughput
2. Live Score Feed    — Scrolling table of recent scored transactions
3. Alert Monitor      — High-risk alert log with drill-down
4. Model Performance  — ROC / PR curves, leaderboard comparison
5. Feature Importance — Top features by model
6. Transaction Tester — Manual transaction scoring UI
"""

import os
import time
import json
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import requests
import yaml


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────

def load_config():
    for p in ["config.yaml", "../config.yaml"]:
        if os.path.exists(p):
            with open(p) as f:
                return yaml.safe_load(f)
    return {}


CONFIG = load_config()
API_URL = CONFIG.get("dashboard", {}).get("api_url", "http://localhost:8000")
REFRESH_INTERVAL = CONFIG.get("dashboard", {}).get("refresh_interval_seconds", 5)

RISK_COLORS = {
    "LOW":      "#28a745",
    "MEDIUM":   "#ffc107",
    "HIGH":     "#fd7e14",
    "CRITICAL": "#dc3545",
}


# ─────────────────────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────────────────────

def api_get(endpoint: str, default=None):
    try:
        r = requests.get(f"{API_URL}{endpoint}", timeout=3)
        if r.ok:
            return r.json()
    except Exception:
        pass
    return default


def api_post(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=5)
        if r.ok:
            return r.json()
        return {"error": r.text}
    except Exception as e:
        return {"error": str(e)}


def check_api_alive() -> bool:
    h = api_get("/health")
    return h is not None and h.get("status") == "ok"


# ─────────────────────────────────────────────────────────────────────
# Synthetic live feed generator (when API unavailable)
# ─────────────────────────────────────────────────────────────────────

CATS = ["grocery", "gas_station", "restaurant", "retail", "electronics",
        "online_shopping", "jewelry", "travel", "crypto_exchange"]
TIERS = ["LOW", "LOW", "LOW", "LOW", "LOW", "MEDIUM", "MEDIUM", "HIGH", "CRITICAL"]


def generate_demo_transaction() -> dict:
    is_fraud = random.random() < 0.03
    tier = random.choice(["HIGH", "CRITICAL"]) if is_fraud else random.choice(["LOW", "LOW", "LOW", "MEDIUM"])
    prob = random.uniform(0.6, 0.99) if is_fraud else random.uniform(0.0, 0.45)
    return {
        "transaction_id": f"SIM_{random.randint(100000, 999999)}",
        "timestamp": datetime.utcnow().isoformat(),
        "merchant_category": random.choice(CATS),
        "amount": round(random.lognormal(4, 0.8), 2),
        "fraud_probability": round(prob, 4),
        "fraud_prediction": int(is_fraud),
        "risk_tier": tier,
    }


def get_demo_stats() -> dict:
    return {
        "total_processed": random.randint(50000, 200000),
        "total_flagged": random.randint(500, 3000),
        "fraud_rate_rolling": round(random.uniform(1.0, 3.5), 3),
        "avg_fraud_score": round(random.uniform(0.05, 0.12), 4),
        "avg_latency_ms": round(random.uniform(1.5, 8.0), 2),
        "throughput_tps": round(random.uniform(40, 120), 1),
        "uptime_seconds": random.randint(300, 7200),
    }


def get_demo_alerts() -> dict:
    alerts = []
    for _ in range(random.randint(5, 20)):
        prob = round(random.uniform(0.7, 0.99), 4)
        alerts.append({
            "alert_id": f"ALT_{random.randint(1000, 9999)}",
            "timestamp": (datetime.utcnow() - timedelta(seconds=random.randint(0, 3600))).isoformat(),
            "transaction_id": f"SIM_{random.randint(100000, 999999)}",
            "fraud_probability": prob,
            "risk_tier": "CRITICAL" if prob > 0.85 else "HIGH",
            "risk_factors": random.sample([
                "High transaction amount",
                "Far from home location",
                "International transaction",
                "High velocity: 5 txns in 1h",
                "Failed auth attempts detected",
                "High-risk merchant category",
            ], k=random.randint(2, 4)),
        })
    return {"alerts": alerts, "total": len(alerts)}


# ─────────────────────────────────────────────────────────────────────
# Streamlit page config
# ─────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        color: white;
        border-left: 4px solid #0f3460;
    }
    .fraud-alert {
        background: #2d0000;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        margin: 0.3rem 0;
    }
    .stMetric { background: #f8f9fa; border-radius: 8px; padding: 0.5rem; }
    div[data-testid="stSidebarNav"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank-cards.png", width=64)
    st.title("🛡️ Fraud Detection")
    st.caption("Real-Time AI Monitoring System")
    st.divider()

    api_alive = check_api_alive()
    if api_alive:
        st.success("🟢 API Online", icon="✅")
    else:
        st.warning("🟡 Demo Mode (API offline)", icon="⚠️")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔴 Live Alerts", "🤖 Model Performance",
         "🔍 Transaction Tester", "📈 Analytics"],
        label_visibility="collapsed",
    )
    st.divider()
    auto_refresh = st.toggle("Auto-refresh", value=True)
    refresh_sec = st.slider("Refresh interval (s)", 2, 30, REFRESH_INTERVAL)
    if st.button("🔄 Refresh Now"):
        st.rerun()

    st.divider()
    st.caption(f"API: `{API_URL}`")
    st.caption(f"Last update: {datetime.now().strftime('%H:%M:%S')}")


# ─────────────────────────────────────────────────────────────────────
# Data fetch (live or demo)
# ─────────────────────────────────────────────────────────────────────

stats = api_get("/pipeline/stats", get_demo_stats()) if api_alive else get_demo_stats()
alerts_data = api_get("/pipeline/alerts?limit=100", get_demo_alerts()) if api_alive else get_demo_alerts()
alerts = alerts_data.get("alerts", [])


# ─────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("📊 Real-Time Fraud Detection — Overview")

    # KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Processed", f"{stats['total_processed']:,}")
    col2.metric("Total Flagged", f"{stats['total_flagged']:,}",
                delta=f"{stats['fraud_rate_rolling']:.2f}% fraud rate",
                delta_color="inverse")
    col3.metric("Avg Fraud Score", f"{stats['avg_fraud_score']:.4f}")
    col4.metric("Throughput", f"{stats['throughput_tps']} TPS")
    col5.metric("Avg Latency", f"{stats['avg_latency_ms']} ms")

    st.divider()

    # Simulated time series
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Transaction Volume & Fraud Rate (Last 24h)")
        hours = pd.date_range(end=datetime.utcnow(), periods=24, freq="h")
        vol = np.random.poisson(lam=3000, size=24)
        fraud_rate = np.random.beta(2, 80, 24) * 100

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=hours, y=vol, name="Transactions",
                             marker_color="steelblue", opacity=0.7), secondary_y=False)
        fig.add_trace(go.Scatter(x=hours, y=fraud_rate, name="Fraud Rate %",
                                 line=dict(color="crimson", width=2.5),
                                 mode="lines+markers"), secondary_y=True)
        fig.update_layout(height=350, legend=dict(orientation="h"),
                          margin=dict(l=0, r=0, t=20, b=0))
        fig.update_yaxes(title_text="Transactions", secondary_y=False)
        fig.update_yaxes(title_text="Fraud Rate %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Risk Tier Distribution")
        tier_counts = {"LOW": 8500, "MEDIUM": 950, "HIGH": 320, "CRITICAL": 130}
        colors = [RISK_COLORS[t] for t in tier_counts]
        fig_pie = go.Figure(go.Pie(
            labels=list(tier_counts.keys()),
            values=list(tier_counts.values()),
            marker=dict(colors=colors),
            hole=0.45,
        ))
        fig_pie.update_layout(height=350, margin=dict(l=0, r=0, t=20, b=0),
                              showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    # Recent transactions feed
    st.subheader("Recent Transaction Feed")
    feed = [generate_demo_transaction() for _ in range(20)]
    feed_df = pd.DataFrame(feed)[
        ["transaction_id", "timestamp", "merchant_category", "amount",
         "fraud_probability", "risk_tier"]
    ]
    feed_df["amount"] = feed_df["amount"].apply(lambda x: f"${x:,.2f}")
    feed_df["fraud_probability"] = feed_df["fraud_probability"].apply(lambda x: f"{x:.4f}")

    def color_tier(val):
        c = RISK_COLORS.get(val, "#6c757d")
        return f"color: {c}; font-weight: bold"

    styled = feed_df.style.applymap(color_tier, subset=["risk_tier"])
    st.dataframe(styled, use_container_width=True, height=400)


# ─────────────────────────────────────────────────────────────────────
# PAGE: Live Alerts
# ─────────────────────────────────────────────────────────────────────

elif page == "🔴 Live Alerts":
    st.title("🔴 Live Fraud Alerts")
    total_alerts = alerts_data.get("total", len(alerts))

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Alerts", total_alerts)
    critical = sum(1 for a in alerts if a.get("risk_tier") == "CRITICAL")
    high = sum(1 for a in alerts if a.get("risk_tier") == "HIGH")
    col2.metric("CRITICAL", critical, delta_color="inverse")
    col3.metric("HIGH", high, delta_color="inverse")

    st.divider()

    tier_filter = st.multiselect(
        "Filter by risk tier",
        ["CRITICAL", "HIGH", "MEDIUM"],
        default=["CRITICAL", "HIGH"],
    )

    filtered = [a for a in alerts if a.get("risk_tier") in tier_filter]

    if not filtered:
        st.info("No alerts matching the selected filters.")
    else:
        for alert in filtered:
            tier = alert.get("risk_tier", "HIGH")
            color = RISK_COLORS.get(tier, "#fd7e14")
            prob = alert.get("fraud_probability", 0)

            with st.expander(
                f"🚨 [{tier}] {alert.get('transaction_id', 'unknown')} — "
                f"Fraud Score: {prob:.4f} — {alert.get('timestamp', '')[:19]}",
                expanded=(tier == "CRITICAL"),
            ):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"**Alert ID:** `{alert.get('alert_id', 'N/A')}`")
                    st.markdown(f"**Transaction:** `{alert.get('transaction_id', 'N/A')}`")
                    st.markdown(f"**Timestamp:** {alert.get('timestamp', 'N/A')[:19]}")
                    st.markdown(f"**Risk Tier:** <span style='color:{color}; font-weight:bold'>{tier}</span>",
                                unsafe_allow_html=True)
                    st.markdown(f"**Fraud Probability:** `{prob:.4f}`")

                with col_b:
                    factors = alert.get("risk_factors", alert.get("risk_factors", []))
                    if factors:
                        st.markdown("**Risk Factors:**")
                        for fac in factors:
                            st.markdown(f"  - {fac}")

                # Probability gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Fraud Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 30], "color": "#e8f5e9"},
                            {"range": [30, 60], "color": "#fff9c4"},
                            {"range": [60, 80], "color": "#ffe0b2"},
                            {"range": [80, 100], "color": "#ffcdd2"},
                        ],
                    },
                    number={"suffix": "%"},
                ))
                fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=0))
                st.plotly_chart(fig_gauge, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: Model Performance
# ─────────────────────────────────────────────────────────────────────

elif page == "🤖 Model Performance":
    st.title("🤖 Model Performance")

    leaderboard_paths = [
        "models/reports/all_models_metrics.csv",
        "models/artifacts/leaderboard.csv",
    ]
    lb_df = None
    for p in leaderboard_paths:
        if os.path.exists(p):
            lb_df = pd.read_csv(p, index_col=0)
            break

    if lb_df is None:
        # Demo leaderboard
        lb_df = pd.DataFrame({
            "roc_auc":    [0.9921, 0.9876, 0.9845, 0.9812, 0.9634, 0.8721],
            "pr_auc":     [0.8765, 0.8654, 0.8543, 0.8432, 0.7865, 0.6543],
            "f1":         [0.8234, 0.8121, 0.8043, 0.7965, 0.7543, 0.6234],
            "precision":  [0.8765, 0.8654, 0.8432, 0.8321, 0.8123, 0.6789],
            "recall":     [0.7765, 0.7654, 0.7654, 0.7643, 0.7012, 0.5765],
        }, index=["xgboost", "lightgbm", "random_forest", "neural_network",
                  "logistic_regression", "isolation_forest"])
        st.info("Demo data shown. Run `python scripts/train_models.py` to populate real metrics.")

    numeric_cols = lb_df.select_dtypes(include=[float, int]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["📋 Leaderboard", "📈 Metric Comparison", "🎯 ROC / PR Curves"])

    with tab1:
        st.dataframe(
            lb_df[numeric_cols].style.background_gradient(cmap="RdYlGn", axis=0),
            use_container_width=True,
        )

    with tab2:
        metrics_to_plot = [c for c in ["roc_auc", "pr_auc", "f1", "precision", "recall"]
                           if c in lb_df.columns]
        selected = st.multiselect("Select metrics", metrics_to_plot, default=metrics_to_plot[:3])
        if selected:
            fig_bar = go.Figure()
            for metric in selected:
                fig_bar.add_trace(go.Bar(
                    name=metric.upper().replace("_", "-"),
                    x=lb_df.index.tolist(),
                    y=lb_df[metric].tolist(),
                ))
            fig_bar.update_layout(
                barmode="group", height=400,
                xaxis_title="Model", yaxis_title="Score",
                yaxis_range=[0, 1],
                title="Model Comparison by Metric",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        # Simulated ROC curves
        fig_roc = go.Figure()
        colors_roc = px.colors.qualitative.Plotly
        models_demo = {
            "XGBoost":           0.9921,
            "LightGBM":          0.9876,
            "Random Forest":     0.9845,
            "Neural Network":    0.9812,
            "Logistic Regression": 0.9634,
        }
        for i, (mname, auc_val) in enumerate(models_demo.items()):
            fpr = np.linspace(0, 1, 200)
            # Simulate ROC from AUC
            tpr = 1 - np.exp(-fpr * 8 * auc_val)
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, name=f"{mname} (AUC={auc_val:.4f})",
                line=dict(color=colors_roc[i], width=2),
            ))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random",
                                     line=dict(color="gray", dash="dash")))
        fig_roc.update_layout(
            title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR",
            height=450, xaxis_range=[-0.01, 1.01], yaxis_range=[-0.01, 1.02],
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # Feature importance
    st.divider()
    st.subheader("Feature Importance (Best Model)")
    features = [
        "txn_velocity_1h", "amount_deviation", "distance_from_home_km",
        "amount_log", "txn_velocity_24h", "time_since_last_txn_min",
        "failed_attempts_before", "is_night", "unique_merchants_24h",
        "is_international", "amount_to_avg_ratio", "is_rapid_succession",
        "account_age_days", "hour_of_day", "is_large_distance",
        "merchant_category", "transaction_type", "card_present",
        "is_online", "amount_rounded",
    ]
    importances = np.random.dirichlet(np.ones(len(features)) * 2)
    importances = pd.Series(importances, index=features).sort_values(ascending=True)

    fig_fi = px.bar(
        importances, orientation="h",
        title="Top Feature Importances",
        color=importances.values,
        color_continuous_scale="RdYlGn",
        labels={"value": "Importance", "index": "Feature"},
    )
    fig_fi.update_layout(height=500, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig_fi, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# PAGE: Transaction Tester
# ─────────────────────────────────────────────────────────────────────

elif page == "🔍 Transaction Tester":
    st.title("🔍 Transaction Scorer")
    st.markdown("Manually score a transaction in real time.")

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            txn_id = st.text_input("Transaction ID", value=f"TXN_{random.randint(100000,999999)}")
            user_id = st.text_input("User ID", value="USR_000001")
            merchant_id = st.text_input("Merchant ID", value="MRC_00001")
            amount = st.number_input("Amount ($)", min_value=0.01, max_value=100000.0, value=250.0)

        with col2:
            merchant_category = st.selectbox("Merchant Category", CATS, index=4)
            txn_type = st.selectbox("Transaction Type",
                                    ["purchase", "online_purchase", "atm_withdrawal",
                                     "contactless", "recurring"])
            device_type = st.selectbox("Device Type",
                                       ["mobile_app", "web_browser", "pos_terminal", "atm"])
            timestamp = st.text_input("Timestamp", value=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))

        with col3:
            is_online = st.checkbox("Online Transaction", value=False)
            is_international = st.checkbox("International", value=False)
            dist_home = st.number_input("Distance from Home (km)", 0.0, 20000.0, 50.0)
            failed_attempts = st.slider("Failed Auth Attempts", 0, 10, 0)
            account_age = st.number_input("Account Age (days)", 0, 5000, 365)

        submitted = st.form_submit_button("🔍 Score Transaction", type="primary")

    if submitted:
        payload = {
            "transaction_id": txn_id,
            "user_id": user_id,
            "merchant_id": merchant_id,
            "merchant_category": merchant_category,
            "timestamp": timestamp,
            "amount": float(amount),
            "transaction_type": txn_type,
            "device_type": device_type,
            "is_online": int(is_online),
            "card_present": int(not is_online),
            "is_international": int(is_international),
            "distance_from_home_km": float(dist_home),
            "failed_attempts_before": int(failed_attempts),
            "account_age_days": int(account_age),
        }

        if api_alive:
            result = api_post("/predict", payload)
        else:
            # Simulate scoring
            risk_score = (
                (amount / 5000) * 0.25 +
                (dist_home / 10000) * 0.20 +
                int(is_international) * 0.15 +
                (failed_attempts / 10) * 0.20 +
                int(is_online and merchant_category in ["crypto_exchange", "jewelry"]) * 0.20
            )
            risk_score = min(risk_score + random.uniform(0, 0.05), 0.99)
            tier = ("CRITICAL" if risk_score > 0.8 else
                    "HIGH" if risk_score > 0.6 else
                    "MEDIUM" if risk_score > 0.3 else "LOW")
            result = {
                "transaction_id": txn_id,
                "fraud_probability": round(risk_score, 4),
                "fraud_prediction": int(risk_score > 0.5),
                "risk_tier": tier,
                "risk_description": f"Demo scoring result — risk tier: {tier}",
                "top_risk_factors": ["Demo mode: connect API for real ML scoring"],
                "latency_ms": 2.3,
                "scored_at": datetime.utcnow().isoformat(),
            }

        if "error" in result:
            st.error(f"API Error: {result['error']}")
        else:
            prob = result.get("fraud_probability", 0)
            tier = result.get("risk_tier", "LOW")
            color = RISK_COLORS.get(tier, "#28a745")

            st.divider()
            rc1, rc2, rc3 = st.columns([1, 1, 2])
            with rc1:
                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob * 100,
                    title={"text": "Fraud Score", "font": {"size": 16}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": color, "thickness": 0.3},
                        "steps": [
                            {"range": [0, 30], "color": "#e8f5e9"},
                            {"range": [30, 60], "color": "#fff9c4"},
                            {"range": [60, 80], "color": "#ffe0b2"},
                            {"range": [80, 100], "color": "#ffcdd2"},
                        ],
                        "threshold": {"line": {"color": "red"}, "value": 50},
                    },
                    number={"suffix": "%", "font": {"size": 28}},
                ))
                fig_g.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=0))
                st.plotly_chart(fig_g, use_container_width=True)

            with rc2:
                st.markdown(f"### Risk Tier")
                st.markdown(f"<h1 style='color:{color}'>{tier}</h1>", unsafe_allow_html=True)
                verdict = "🚨 FRAUD LIKELY" if result["fraud_prediction"] == 1 else "✅ LEGITIMATE"
                st.markdown(f"**Verdict:** {verdict}")
                st.markdown(f"**Latency:** {result.get('latency_ms', 0):.1f} ms")
                st.markdown(f"**Scored at:** {result.get('scored_at', '')[:19]}")

            with rc3:
                st.markdown("### Risk Factors")
                for factor in result.get("top_risk_factors", []):
                    st.markdown(f"  - {factor}")
                st.markdown("### Raw Response")
                st.json(result)


# ─────────────────────────────────────────────────────────────────────
# PAGE: Analytics
# ─────────────────────────────────────────────────────────────────────

elif page == "📈 Analytics":
    st.title("📈 Fraud Analytics")

    tab1, tab2, tab3 = st.tabs(["By Category", "By Hour", "Score Distribution"])

    with tab1:
        st.subheader("Fraud Rate by Merchant Category")
        categories = CATS
        fraud_rates = np.random.beta(1, 20, len(categories)) * 100
        legit_rates = 100 - fraud_rates
        df_cat = pd.DataFrame({"category": categories,
                               "fraud_rate": fraud_rates,
                               "legit_rate": legit_rates}).sort_values("fraud_rate", ascending=False)
        fig_cat = px.bar(df_cat, x="category", y=["fraud_rate", "legit_rate"],
                         title="Transaction Mix by Category",
                         color_discrete_map={"fraud_rate": "crimson", "legit_rate": "steelblue"},
                         barmode="stack")
        st.plotly_chart(fig_cat, use_container_width=True)

    with tab2:
        st.subheader("Fraud Volume by Hour of Day")
        hours = list(range(24))
        fraud_vol = np.random.poisson(lam=[5 if h < 6 or h > 22 else 2 for h in hours])
        legit_vol = np.random.poisson(lam=[10 if h < 6 else 150 for h in hours])
        fig_hour = go.Figure()
        fig_hour.add_trace(go.Bar(x=hours, y=legit_vol, name="Legitimate", marker_color="steelblue"))
        fig_hour.add_trace(go.Bar(x=hours, y=fraud_vol, name="Fraud", marker_color="crimson"))
        fig_hour.update_layout(barmode="group", xaxis_title="Hour (UTC)",
                               yaxis_title="Transaction Count",
                               title="Transactions by Hour — Fraud vs Legitimate")
        st.plotly_chart(fig_hour, use_container_width=True)

    with tab3:
        st.subheader("Fraud Score Distribution")
        legit_scores = np.random.beta(1, 15, 10000)
        fraud_scores = np.random.beta(8, 3, 500)

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=legit_scores, name="Legitimate", nbinsx=60,
            opacity=0.6, marker_color="steelblue", histnorm="probability density",
        ))
        fig_dist.add_trace(go.Histogram(
            x=fraud_scores, name="Fraud", nbinsx=60,
            opacity=0.7, marker_color="crimson", histnorm="probability density",
        ))
        fig_dist.add_vline(x=0.5, line_dash="dash", line_color="orange",
                           annotation_text="Decision Threshold")
        fig_dist.update_layout(barmode="overlay", title="Fraud Score Distribution",
                               xaxis_title="Fraud Probability", yaxis_title="Density")
        st.plotly_chart(fig_dist, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────
# Auto-refresh
# ─────────────────────────────────────────────────────────────────────

if auto_refresh and page in ["📊 Overview", "🔴 Live Alerts"]:
    time.sleep(refresh_sec)
    st.rerun()
