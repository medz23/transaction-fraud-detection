
from functools import lru_cache

import joblib
import pandas as pd

from app.data import FEATURES, add_features
from app.paths import MODEL_FILE


@lru_cache
def load_model(path=MODEL_FILE):
    return joblib.load(path)


def score(transaction: dict) -> dict:
    saved = load_model()
    row = add_features(pd.DataFrame([transaction]))
    probability = float(saved["model"].predict_proba(row[FEATURES])[:, 1][0])
    return {
        "fraud_probability": round(probability, 4),
        "is_fraud": probability >= saved["threshold"],
        "threshold": round(saved["threshold"], 4),
    }
