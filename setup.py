from setuptools import setup, find_packages

setup(
    name="real-time-fraud-detection",
    version="1.0.0",
    description="Real-Time AI/ML Fraud Detection System",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=1.7.6",
        "lightgbm>=4.0.0",
        "imbalanced-learn>=0.11.0",
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.23.0",
        "pydantic>=2.0.0",
        "streamlit>=1.25.0",
        "plotly>=5.15.0",
        "joblib>=1.3.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
        "shap>=0.42.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-asyncio>=0.21.0", "httpx>=0.24.0"],
    },
    entry_points={
        "console_scripts": [
            "fraud-train=scripts.train_models:main",
            "fraud-pipeline=scripts.run_pipeline:main",
            "fraud-api=fraud_detection.api.server:app",
        ]
    },
)
