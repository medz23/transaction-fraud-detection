from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SESSIONS_DIR = ROOT / "sessions"
MODEL_FILE = ROOT / "model.joblib"

DATA_DIR = ROOT / "data"
TRANSACTIONS_CSV = DATA_DIR / "transactions.csv"
