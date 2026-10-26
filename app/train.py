
import warnings

import joblib
import lightgbm as lgb
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")

from app.data import CATEGORICAL, FEATURES, NUMERIC, add_features, clean, load, split_by_time
from app.paths import MODEL_FILE


def preprocessor(scale=False, numeric=NUMERIC):
    """Fill missing numbers with the median (+ scale them for the LSTM), one-hot the categories."""
    numeric_steps = [("fill", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])  # fmt: skip


def train_forest(train):
    """SMOTE sits INSIDE the pipeline, so it only ever sees training rows."""
    model = Pipeline([
        ("prep", preprocessor()),
        ("smote", SMOTE(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)),
    ])  # fmt: skip
    return model.fit(train[FEATURES], train["is_fraud"])


def train_lightgbm(train, val, numeric=NUMERIC):
    """Boosting: small trees added one by one, each fixing the previous ones' mistakes.

    Early stopping: keep adding trees while validation PR-AUC improves; stop 100 trees after the best one.
    """
    features = numeric + CATEGORICAL
    prep = preprocessor(numeric=numeric).fit(train[features])
    frauds = int(train["is_fraud"].sum())
    model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=15,  # small trees
        min_child_samples=50 if frauds > 200 else 20,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(
        prep.transform(train[features]), train["is_fraud"],
        eval_set=[(prep.transform(val[features]), val["is_fraud"])], eval_metric="average_precision",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    print(f"LightGBM stopped at {model.best_iteration_} trees")
    return Pipeline([("prep", prep), ("lgbm", model)])


def best_threshold(y, proba):
    """The cut-off with the best F1 on VALIDATION data. Never pick it on test."""
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(thresholds[np.argmax(f1[:-1])])


def report(name, y, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"{name:<19} PR-AUC {average_precision_score(y, proba):.3f}   "
          f"precision {tp / max(tp + fp, 1):.2f}   recall {tp / max(tp + fn, 1):.2f}   "
          f"caught {tp}  missed {fn}  false alarms {fp}")  # fmt: skip


def windows(frame, x, k=5):
    """For each transaction: it and the customer's previous k-1 transactions, padded with zeros."""
    out = np.zeros((len(frame), k, x.shape[1]), dtype=np.float32)
    customer = frame["customer_id"].to_numpy()
    for i in range(len(frame)):
        for step in range(k):
            j = i - step
            if j < 0 or customer[j] != customer[i]:
                break
            out[i, k - 1 - step] = x[j]
    return out


def train_lstm(train, val, test, epochs=30, patience=5):
    import torch
    from torch import nn

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prep = preprocessor(scale=True).fit(train[FEATURES])
    x_tr, x_va, x_te = (torch.tensor(windows(f, prep.transform(f[FEATURES]))).to(device) for f in (train, val, test))
    y_tr = torch.tensor(train["is_fraud"].to_numpy(), dtype=torch.float32).to(device)

    class FraudLSTM(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.lstm = nn.LSTM(n_features, 64, batch_first=True)
            self.out = nn.Linear(64, 1)

        def forward(self, x):
            _, (h, _) = self.lstm(x)
            return self.out(h[-1]).squeeze(-1)

    model = FraudLSTM(x_tr.shape[2]).to(device)
    pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def predict(x):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(model(x)).cpu().numpy()

    best, best_state, bad_epochs = -1, None, 0
    for epoch in range(epochs):
        model.train()
        for idx in torch.randperm(len(x_tr), device=device).split(512):
            optimizer.zero_grad()
            loss = loss_fn(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        score = average_precision_score(val["is_fraud"], predict(x_va))
        if score > best:
            best, best_state, bad_epochs = score, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs == patience:
                break
    model.load_state_dict(best_state)
    print(f"LSTM stopped after {epoch + 1} epochs (best on validation: epoch {epoch + 1 - bad_epochs})")
    return predict(x_va), predict(x_te)


def main():
    df = add_features(clean(load()))
    train, val, test = split_by_time(df)
    print(f"train {len(train)} rows ({train.is_fraud.sum()} frauds) | validation {len(val)} | test {len(test)}")

    models = {"Random Forest": train_forest(train), "LightGBM": train_lightgbm(train, val)}
    scores = {name: (m.predict_proba(val[FEATURES])[:, 1], m.predict_proba(test[FEATURES])[:, 1])
              for name, m in models.items()}  # fmt: skip
    scores["LSTM"] = train_lstm(train, val, test)

    from app.history import HISTORY, add_history

    h_train, h_val, h_test = split_by_time(add_history(df))
    h_model = train_lightgbm(h_train, h_val, numeric=NUMERIC + HISTORY)
    h_features = NUMERIC + HISTORY + CATEGORICAL
    scores["LightGBM + history"] = (h_model.predict_proba(h_val[h_features])[:, 1],
                                    h_model.predict_proba(h_test[h_features])[:, 1])

    print("\nResults on the June test set (each threshold chosen on May):")
    thresholds = {}
    for name, (p_val, p_test) in scores.items():
        thresholds[name] = best_threshold(val["is_fraud"], p_val)
        report(name, test["is_fraud"], p_test, thresholds[name])

    best = max(models, key=lambda name: average_precision_score(val["is_fraud"], scores[name][0]))
    joblib.dump({"model": models[best], "threshold": thresholds[best], "name": best}, MODEL_FILE)
    print(f"\nSaved {best} to {MODEL_FILE} (best on validation among models that score one transaction alone)")


if __name__ == "__main__":
    main()
