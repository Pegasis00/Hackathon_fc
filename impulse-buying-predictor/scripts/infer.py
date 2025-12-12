# scripts/infer.py
import json
from pathlib import Path
import joblib
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
MODEL = BASE/"models"/"model.joblib"
FEATURE_INFO = BASE/"models"/"feature_info.json"
DATA = BASE/"data"/"data.csv"

model = joblib.load(MODEL)
feature_info = json.load(open(FEATURE_INFO))

# Load and prepare
df = pd.read_csv(DATA)

# Apply mappings if present in feature_info
mood_map = feature_info.get("mood_mapping", {})
city_map = feature_info.get("city_mapping", {})

if "mood_last_week" in df.columns and mood_map:
    df["mood_last_week"] = df["mood_last_week"].map(mood_map)

if "city" in df.columns and city_map:
    df["city"] = df["city"].astype(str).str.strip().map(city_map)

X = df.drop(columns=["impulse_buy_score", "user_id"], errors="ignore")

# Quick NaN check
nan_cols = X.columns[X.isnull().any()].tolist()
if nan_cols:
    print("Columns with NaNs after mapping:", nan_cols)
    # You can choose to impute here if you want to auto-fix; for safety we stop.
    X[nan_cols] = X[nan_cols].fillna(0)  # quick fallback but inspect recommended

preds = model.predict(X.head(10))
print("Preds:", preds)
