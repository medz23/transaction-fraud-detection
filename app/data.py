"""Load the raw CSV, clean it, add features, split it by time."""

import numpy as np
import pandas as pd

from app.paths import TRANSACTIONS_CSV

LABELS = {"1": 1, "yes": 1, "y": 1, "fraud": 1, "true": 1, "0": 0, "no": 0, "n": 0, "legit": 0, "false": 0}
YES_NO = {"1": 1, "yes": 1, "y": 1, "true": 1, "t": 1, "0": 0, "no": 0, "n": 0, "false": 0, "f": 0}
CHANNELS = {"online": "online", "web": "online", "pos": "pos", "in-store": "pos", "instore": "pos", "atm": "atm"}
CATEGORY_TYPOS = {
    "grocerys": "grocery", "restaurants": "restaurant", "electronic": "electronics", "fuel/gas": "fuel",
    "clothes": "clothing", "entertainmnet": "entertainment", "utility": "utilities",
    "online services": "online_services",
}  # fmt: skip
COUNTRIES = {"usa": "US", "united states": "US", "uk": "GB", "united kingdom": "GB", "germany": "DE",
             "france": "FR", "romania": "RO", "spain": "ES"}  # fmt: skip
MISSING_WORDS = ["", "n/a", "na", "null", "none", "nan", "?", "-", "--", "unknown", "error"]

NUMERIC = ["amount_log", "amount_ratio", "account_age_days", "num_txn_24h", "distance_log", "customer_age",
           "hour", "is_night", "card_present", "device_missing", "is_refund"]  # fmt: skip
CATEGORICAL = ["channel", "merchant_category", "country"]
FEATURES = NUMERIC + CATEGORICAL


def load(path=TRANSACTIONS_CSV):
    return pd.read_csv(path)


def text(col):
    """Lower-case, trim, and turn 'N/A', '?', 'unknown'... into real missing values."""
    s = col.astype("string").str.strip().str.lower()
    return s.mask(s.isin(MISSING_WORDS))


def parse_time(col):
    """The file mixes three formats: '2024-03-21 07:35:50', '03/21/2024 07:35 AM' and 1711006550."""
    s = col.astype("string").str.strip()
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    us = pd.to_datetime(s, format="%m/%d/%Y %I:%M %p", errors="coerce")
    epoch = pd.to_datetime(pd.to_numeric(s.where(s.str.fullmatch(r"\d{9,11}")), errors="coerce"), unit="s")
    return iso.fillna(us).fillna(epoch)


def clean(raw):
    df = raw.copy()

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df["amount"] = pd.to_numeric(df["amount"].astype(str).str.replace(r"[^\d.\-]", "", regex=True), errors="coerce")
    df["is_fraud"] = text(df["is_fraud"]).map(LABELS)
    df = df.drop_duplicates()
    df = df.sort_values("is_fraud", ascending=False, na_position="last")
    df = df.drop_duplicates(subset="transaction_id").sort_index()
    df = df.dropna(subset=["is_fraud"])
    df["is_fraud"] = df["is_fraud"].astype(int)
    df = df.drop(columns=["fraud_review_outcome", "record_source"])
    for c in ["customer_age", "account_age_days", "avg_amount_30d", "num_txn_24h", "distance_from_home_km"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["customer_age"] = df["customer_age"].where(df["customer_age"].between(18, 120))
    df["distance_from_home_km"] = df["distance_from_home_km"].where(df["distance_from_home_km"] >= 0)
    df["account_age_days"] = df["account_age_days"].where(df["account_age_days"] >= 0)
    df["amount"] = df["amount"].where(df["amount"] <= 50_000)
    df["is_refund"] = (df["amount"] < 0).astype(int)
    df["amount"] = df["amount"].abs()

    df["channel"] = text(df["channel"]).map(CHANNELS).fillna("unknown")
    df["merchant_category"] = text(df["merchant_category"]).replace(CATEGORY_TYPOS).fillna("unknown")
    df["country"] = text(df["country"]).replace(COUNTRIES).str.upper().fillna("unknown")
    df["card_present"] = text(df["card_present"]).map(YES_NO)

    eur = text(df["currency"]) == "eur"
    df.loc[eur, ["amount", "avg_amount_30d"]] *= 1.08

    df["timestamp"] = parse_time(df["timestamp"])
    df["timestamp"] = df["timestamp"].where(df["timestamp"].between("2024-01-01", "2024-06-30 23:59:59"))

    df["device_missing"] = df["device_id"].isna().astype(int)
    df = df.drop(columns=["device_id"])
    return df.reset_index(drop=True)


def add_features(df):
    """Used by training AND by the API, so a request gets exactly the same features."""
    df = df.copy()
    for c in ["amount", "avg_amount_30d", "distance_from_home_km", "customer_age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")  # an API request may send None
    df["amount_log"] = np.log1p(df["amount"])
    df["amount_ratio"] = df["amount"] / df["avg_amount_30d"].replace(0, np.nan)
    df["distance_log"] = np.log1p(df["distance_from_home_km"])
    df["hour"] = df["timestamp"].dt.hour
    df["is_night"] = df["hour"].isin([23, 0, 1, 2, 3, 4]).astype(int)
    if "is_refund" not in df:
        df["is_refund"] = 0
    return df


def split_by_time(df, train_end="2024-05-01", val_end="2024-06-01"):
    """Learn from the past, tune on May, test on June: the way the model will really be used."""
    df = df.dropna(subset=["timestamp"]).sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    train = df[df["timestamp"] < train_end]
    val = df[(df["timestamp"] >= train_end) & (df["timestamp"] < val_end)]
    test = df[df["timestamp"] >= val_end]
    return train, val, test

