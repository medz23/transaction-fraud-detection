from app.data import FEATURES, add_features, clean, load, split_by_time


def cleaned():
    return clean(load())


def test_label_is_only_0_or_1():
    df = cleaned()
    assert set(df["is_fraud"].unique()) == {0, 1}


def test_each_transaction_appears_once():
    assert cleaned()["transaction_id"].is_unique


def test_leakage_column_is_gone():
    assert "fraud_review_outcome" not in cleaned().columns


def test_amount_is_a_positive_number():
    amount = cleaned()["amount"].dropna()
    assert amount.dtype.kind == "f" and (amount > 0).all()


def test_ages_are_realistic():
    assert cleaned()["customer_age"].dropna().between(18, 120).all()


def test_split_is_by_time():
    train, val, test = split_by_time(add_features(cleaned()))
    assert train["timestamp"].max() < val["timestamp"].min() <= val["timestamp"].max() < test["timestamp"].min()


def test_features_exist():
    df = add_features(cleaned())
    assert set(FEATURES) <= set(df.columns)
