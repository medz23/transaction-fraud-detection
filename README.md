# Fraud service

Build a small fraud-detection service from a dirty transactions file.

1. **Clean the data** (`app/data.py`): find and fix the data quality problems in `data/transactions.csv`.
2. **Train and compare models** (`app/train.py`): a Random Forest with SMOTE (bagging), LightGBM (boosting) and an LSTM.
   Choose on validation; save the best model that can score one transaction on its own.
3. **Serve it** (`app/api.py`): FastAPI with `/health` and `/predict`, input validation, rate limiting.
   Error handlers live in `app/handlers.py`: every error has one shape, `{"error": ..., "detail": ...}`.
4. **Add an agent** (`app/agent.py`): a Strands agent with a tool that calls the model, exposed at `/chat`.
   The API is stateless, but the agent must remember the conversation for the same `session_id`.

File locations (the CSV, `model.joblib`, the `sessions/` folder) are set in one place, `app/paths.py`, and are
built from the project folder, not from the folder you happen to run a command in.

## Commands

```
python -m venv venv
.\venv\Scripts\Activate.ps1        # Windows PowerShell   (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
python -m pytest                    # run the tests
python -m app.train                 # train and save model.joblib
uvicorn app.api:app --reload        # start the API, then open http://localhost:8000/docs
```
