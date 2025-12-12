# quick_predict_fix.py
import joblib
import pandas as pd
from pathlib import Path

# load config-like mappings (hardcode or import from your script)
MOOD_MAPPING = {"Happy": 1, "Neutral": 2, "Anxious": 3, "Sad": 4}
CITY_MAPPING = {
    'Delhi': 1, 'Mumbai': 2, 'Bengaluru': 3, 'Kolkata': 4,
    'Chennai': 5, 'Pune': 6, 'Hyderabad': 7
}

MODEL_PATH = Path("models/model.joblib")
DATA_PATH = Path("data/data.csv")

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# Apply the same preprocessing that training used
if "mood_last_week" in df.columns:
    df["mood_last_week"] = df["mood_last_week"].map(MOOD_MAPPING)

if "city" in df.columns:
    # strip whitespace and map (safer)
    df["city"] = df["city"].astype(str).str.strip().map(CITY_MAPPING)

X = df.drop(columns=["impulse_buy_score", "user_id"], errors="ignore")

# If mapping produced NaNs for unseen categories, either fill or inspect:
if X.isnull().any().any():
    print("Warning: NaNs present after mapping. Showing columns with NaNs:")
    print(X.columns[X.isnull().any()].tolist())
    # Optionally: X.fillna(0, inplace=True)  # not ideal - inspect instead

print("Sample predictions:", model.predict(X.head(5)))
