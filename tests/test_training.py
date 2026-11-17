from pathlib import Path

import pandas as pd

from config import DATA_DIR, METADATA_PATH, MODEL_PATH
from src.train_model import train_model


def test_training_creates_model_and_metadata():
    csv_path = DATA_DIR / 'SYNTHETIC_SAMPLE_NOT_CICIDS2017.csv'
    assert csv_path.exists(), 'Synthetic dataset was not generated.'
    result = train_model(DATA_DIR, synthetic=True)
    assert MODEL_PATH.exists()
    assert METADATA_PATH.exists()
    assert isinstance(result['metadata']['feature_names'], list)
    assert result['metadata']['training_count'] > 0
    assert result['metadata']['testing_count'] > 0
