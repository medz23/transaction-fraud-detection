import os

from strands import Agent, tool
from strands.session.file_session_manager import FileSessionManager

from app.model import score
from app.paths import SESSIONS_DIR

SYSTEM_PROMPT = (
    "You are a fraud analyst assistant. When the user describes a card transaction, call predict_fraud. "
    "Report the probability and the decision in plain words. You remember earlier transactions in this "
    "conversation, so you can compare them. Keep answers short."
)


@tool
def predict_fraud(
    amount: float,
    channel: str,
    merchant_category: str,
    country: str,
    card_present: bool,
    account_age_days: int,
    num_txn_24h: int,
    avg_amount_30d: float | None = None,
    distance_from_home_km: float | None = None,
    hour: int | None = None,
) -> dict:
    """Score one card transaction for fraud with the trained model.

    Args:
        amount: amount in USD
        channel: "online", "pos" or "atm"
        merchant_category: e.g. grocery, electronics, travel
        country: two-letter country code, e.g. US
        card_present: True if the physical card was used
        account_age_days: how old the customer's account is, in days
        num_txn_24h: how many transactions the customer made in the last 24 hours
        avg_amount_30d: the customer's usual amount, if known
        distance_from_home_km: distance from the customer's home, if known
        hour: hour of the day 0-23, if known
    """
    import pandas as pd

    when = pd.Timestamp("2024-06-15") + pd.Timedelta(hours=hour if hour is not None else 12)
    row = {
        "amount": amount, "channel": channel, "merchant_category": merchant_category.lower(),
        "country": country.upper(), "card_present": int(card_present), "account_age_days": account_age_days,
        "num_txn_24h": num_txn_24h, "avg_amount_30d": avg_amount_30d, "distance_from_home_km": distance_from_home_km,
        "customer_age": None, "device_missing": 0, "timestamp": when,
    }
    return score(row)


def make_model():
    """Which LLM drives the agent. Default: Amazon Nova on Bedrock (needs AWS credentials)."""
    if os.getenv("MODEL_PROVIDER") == "ollama":
        from strands.models.ollama import OllamaModel

        return OllamaModel(host="http://localhost:11434", model_id=os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    from strands.models import BedrockModel

    return BedrockModel(model_id=os.getenv("BEDROCK_MODEL", "us.amazon.nova-2-lite-v1:0"), region_name="us-east-1")


def ask(session_id: str, message: str) -> str:
    agent = Agent(
        model=make_model(),
        tools=[predict_fraud],
        system_prompt=SYSTEM_PROMPT,
        session_manager=FileSessionManager(session_id=session_id, storage_dir=str(SESSIONS_DIR)),
        callback_handler=None,  # don't print the answer token by token in the server log
    )
    return str(agent(message)).strip()
