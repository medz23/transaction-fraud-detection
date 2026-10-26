
import numpy as np
import pandas as pd

HISTORY = ["n_prior", "mins_since_prev_log", "txn_prev_1h", "txn_prev_24h", "country_new", "category_new",
           "amount_vs_hist"]  # fmt: skip


def count_before(times, hours):
    """For one customer's sorted timestamps: how many earlier payments in the last `hours`."""
    t = times.to_numpy("datetime64[s]").astype(np.int64)
    start = np.searchsorted(t, t - hours * 3600, side="left")
    return np.arange(len(t)) - start


def add_history(df):
    df = df.dropna(subset=["timestamp"]).sort_values(["customer_id", "timestamp"]).reset_index(drop=True)
    by = df.groupby("customer_id")
    df["n_prior"] = by.cumcount()
    minutes = (df["timestamp"] - by["timestamp"].shift()).dt.total_seconds() / 60
    df["mins_since_prev_log"] = np.log1p(minutes)
    for hours in (1, 24):
        df[f"txn_prev_{hours}h"] = by["timestamp"].transform(lambda s, h=hours: count_before(s, h))
    df["country_new"] = (~df.duplicated(["customer_id", "country"])).astype(float)
    df["category_new"] = (~df.duplicated(["customer_id", "merchant_category"])).astype(float)
    df.loc[df["n_prior"] == 0, ["country_new", "category_new"]] = np.nan
    median_so_far = by["amount"].transform(lambda s: s.shift().expanding().median())
    df["amount_vs_hist"] = df["amount"] / median_so_far
    return df
