import joblib
import pandas as pd
from pathlib import Path


def test_model_predicts():
    base = Path(__file__).resolve().parents[1]
    model = joblib.load(base/"models"/"model.joblib")
    df = pd.read_csv(base/"data"/"data.csv")
    X = df.drop(columns=["impulse_buy_score", "user_id"], errors="ignore")
    # quick sample
    preds = model.predict(X.head(3))
    assert len(preds) == 3