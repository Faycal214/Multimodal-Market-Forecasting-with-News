from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_TRAIN_PRICES = RAW_DIR / "train_prices.csv"
RAW_TRAIN_NEWS = RAW_DIR / "train_news.csv"
RAW_TEST_PRICES = RAW_DIR / "test_prices.csv"
RAW_TEST_NEWS = RAW_DIR / "test_news.csv"

PROCESSED_TRAIN_FILE = PROCESSED_DIR / "train_features.csv"
PROCESSED_TEST_FILE = PROCESSED_DIR / "test_features.csv"

MODEL_BUNDLE_FILE = MODELS_DIR / "multimodal_forecaster.pt"
LAG_SCALER_FILE = MODELS_DIR / "lag_scaler.pkl"
EXTRA_SCALER_FILE = MODELS_DIR / "extra_scaler.pkl"
ARTIFACTS_FILE = MODELS_DIR / "artifacts.json"

TARGET_COLS = ["price1", "price2", "price3"]
DATE_COL = "Date"
TEXT_COL = "TaggedNews"

PRICE_COLS = TARGET_COLS
LAG_DAYS = 7
MAX_LEN = 64

TEXT_MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 8
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-4
VAL_SIZE = 0.2
RANDOM_SEED = 42
GRAD_CLIP_NORM = 1.0

TARGET_SCALER_FILE = MODELS_DIR / "target_scaler.pkl"
EARLY_STOPPING_PATIENCE = 2
MIN_DELTA = 1e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"