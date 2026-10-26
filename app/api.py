from datetime import datetime
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.handlers import add_handlers
from app.model import load_model, score

app = FastAPI(title="Fraud service")
add_handlers(app)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


class Transaction(BaseModel):
    amount: float = Field(gt=0, le=50_000, description="amount in USD")
    avg_amount_30d: float | None = Field(None, gt=0, description="the customer's usual amount")
    channel: Literal["online", "pos", "atm"]
    merchant_category: str = Field(min_length=1, max_length=40)
    country: str = Field(min_length=2, max_length=2, description="ISO code, e.g. US")
    card_present: bool
    account_age_days: int = Field(ge=0)
    num_txn_24h: int = Field(ge=1, le=500)
    distance_from_home_km: float | None = Field(None, ge=0)
    customer_age: int | None = Field(None, ge=18, le=120)
    device_known: bool = True
    timestamp: datetime | None = None


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    message: str = Field(min_length=1, max_length=2000)


def to_row(tx: Transaction) -> dict:
    """Turn the request into the same columns clean() produces."""
    row = tx.model_dump()
    row["timestamp"] = pd.Timestamp(row["timestamp"] or datetime.now())
    row["card_present"] = int(row["card_present"])
    row["device_missing"] = int(not row.pop("device_known"))
    row["country"] = row["country"].upper()
    row["merchant_category"] = row["merchant_category"].lower()
    return row


@app.get("/health")
def health():
    try:
        load_model()
        return {"status": "ok", "model_loaded": True}
    except FileNotFoundError:
        return {"status": "ok", "model_loaded": False}


@app.post("/predict")
@limiter.limit("30/minute")
def predict(request: Request, tx: Transaction):
    try:
        return score(to_row(tx))
    except FileNotFoundError:
        return JSONResponse(status_code=503, content={"error": "model_not_loaded", "detail": "run python -m app.train"})


@app.post("/chat")
@limiter.limit("10/minute")
def chat(request: Request, body: ChatRequest):
    from app.agent import ask

    return {"session_id": body.session_id, "reply": ask(body.session_id, body.message)}
