from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
ASSETS_DIR = PROJECT_ROOT / "assets"
SYNTHETIC_SAMPLE_PATH = DATA_DIR / "SYNTHETIC_SAMPLE_NOT_CICIDS2017.csv"
MODEL_PATH = MODELS_DIR / "cyberguard_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
CORRELATION_THRESHOLD = 0.95
MAX_ROWS_DEFAULT = 200000
RANDOM_STATE = 42
TARGET_COLUMN = "Label"
IDENTIFIER_COLUMNS = {
    "flow id",
    "source ip",
    "destination ip",
    "source port",
    "timestamp",
}
