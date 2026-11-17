import pandas as pd
import pytest

from config import DATA_DIR, METADATA_PATH, MODEL_PATH
from src.predict import load_model_and_metadata, run_prediction_on_csv


def test_prediction_aligns_columns_and_raises_on_missing():
    model, metadata = load_model_and_metadata(MODEL_PATH, METADATA_PATH)
    feature_names = metadata['feature_names']
    frame = pd.read_csv(DATA_DIR / 'SYNTHETIC_SAMPLE_NOT_CICIDS2017.csv')
    first = frame.iloc[:10].copy()
    first = first[feature_names[:5] + ['Label']]
    assert 'predicted_label' in run_prediction_on_csv(DATA_DIR / 'SYNTHETIC_SAMPLE_NOT_CICIDS2017.csv').columns
    missing = first.drop(columns=[feature_names[0]])
    with pytest.raises(ValueError):
        from src.predict import align_features
        align_features(feature_names, missing)
