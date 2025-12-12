"""
Optimized LightGBM Training Script - GitHub Deployment Ready
- Compact model (< 25MB for GitHub without LFS)
- Production-ready artifacts
- Comprehensive evaluation
"""

import os
import json
import warnings
from pathlib import Path
from datetime import datetime
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
from lightgbm import LGBMRegressor


class Config:
    BASE_DIR = Path(__file__).parent
    DATA_PATH = BASE_DIR / "data" / "data.csv"
    MODEL_DIR = BASE_DIR / "models"
    MODEL_OUT = MODEL_DIR / "model.joblib"
    METRICS_OUT = MODEL_DIR / "metrics.json"
    FEATURE_INFO_OUT = MODEL_DIR / "feature_info.json"
    
    RANDOM_STATE = 42
    TEST_SIZE = 0.20
    VAL_SIZE = 0.15
    
    MOOD_MAPPING = {"Happy": 1, "Neutral": 2, "Anxious": 3, "Sad": 4}
    CITY_MAPPING = {
        'Delhi': 1, 'Mumbai': 2, 'Bengaluru': 3, 'Kolkata': 4,
        'Chennai': 5, 'Pune': 6, 'Hyderabad': 7
    }
    
    INSIGNIFICANT_FEATURES = [
        "gender", "num_checkout_visits", "default_payment_method",
        "account_age_days", "num_cart_visits", "device_preference"
    ]


def log(msg, level="INFO"):
    """Simple logging"""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {level}: {msg}")


def load_and_prepare_data():
    """Load and prepare dataset"""
    log("Loading data...")
    
    if not Config.DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {Config.DATA_PATH}")
    
    df = pd.read_csv(Config.DATA_PATH)
    
    if df.empty:
        raise ValueError("Dataset is empty!")
    
    log(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # Apply mappings
    if "mood_last_week" in df.columns:
        df["mood_last_week"] = df["mood_last_week"].map(Config.MOOD_MAPPING)
    
    if "city" in df.columns:
        df["city"] = df["city"].map(Config.CITY_MAPPING)
    
    # Drop insignificant features
    drop_cols = [c for c in Config.INSIGNIFICANT_FEATURES if c in df.columns]
    if drop_cols:
        log(f"  Dropping {len(drop_cols)} insignificant features")
        df = df.drop(columns=drop_cols)
    
    return df


def build_pipeline(numeric_cols, categorical_cols):
    """Build optimized pipeline"""
    
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop")
    
    # Compact LightGBM configuration
    lgbm = LGBMRegressor(
        n_estimators=300,              # Reduced for smaller size
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=Config.RANDOM_STATE,
        n_jobs=1,
        verbosity=-1,
        force_col_wise=True,
        importance_type='gain'
    )
    
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", lgbm)
    ])
    
    return pipeline


def evaluate(y_true, y_pred, set_name="Test"):
    """Evaluate model performance"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE with zero handling
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
    
    results = {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape)
    }
    
    log(f"\n{set_name} Performance:")
    for metric, value in results.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    return results


def main():
    log("=" * 70)
    log("IMPULSE BUYING PREDICTION MODEL TRAINING")
    log("=" * 70)
    
    # Create directories
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_prepare_data()
    
    # Separate features and target
    if "impulse_buy_score" not in df.columns:
        raise KeyError("Target 'impulse_buy_score' not found")
    
    X = df.drop(columns=["impulse_buy_score", "user_id"], errors='ignore')
    y = df["impulse_buy_score"]
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    log(f"\nFeatures: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    # Save feature info for Streamlit
    feature_info = {
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "all_features": numeric_cols + categorical_cols,
        "mood_mapping": Config.MOOD_MAPPING,
        "city_mapping": Config.CITY_MAPPING
    }
    
    with open(Config.FEATURE_INFO_OUT, "w") as f:
        json.dump(feature_info, f, indent=2)
    
    # Split data
    log("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=Config.VAL_SIZE, random_state=Config.RANDOM_STATE
    )
    
    log(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    
    # Build and train pipeline
    log("\nTraining model...")
    pipeline = build_pipeline(numeric_cols, categorical_cols)
    
    pipeline.fit(X_train, y_train)
    
    log("=" * 70)
    log("TRAINING COMPLETE")
    log("=" * 70)
    
    # Evaluate on all sets
    train_preds = pipeline.predict(X_train)
    val_preds = pipeline.predict(X_val)
    test_preds = pipeline.predict(X_test)
    
    train_results = evaluate(y_train, train_preds, "Train")
    val_results = evaluate(y_val, val_preds, "Validation")
    test_results = evaluate(y_test, test_preds, "Test")
    
    # Save model
    log("\nSaving model...")
    joblib.dump(pipeline, Config.MODEL_OUT, compress=3)
    
    model_size_mb = Config.MODEL_OUT.stat().st_size / (1024 * 1024)
    log(f"  Model saved: {Config.MODEL_OUT}")
    log(f"  Model size: {model_size_mb:.2f} MB")
    
    if model_size_mb > 95:
        log("  ⚠️  WARNING: Model > 95MB - consider Git LFS!", "WARNING")
    else:
        log("  ✅ Model size is GitHub-friendly!")
    
    # Save metrics
    metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train": train_results,
        "validation": val_results,
        "test": test_results,
        "model_size_mb": float(model_size_mb)
    }
    
    with open(Config.METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)
    
    log("\n" + "=" * 70)
    log("✅ ALL DONE!")
    log("=" * 70)
    log("\nNext step: Run Streamlit app")
    log("  streamlit run app.py")


if __name__ == "__main__":
    main()