import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from config import DATA_DIR, MODEL_PATH, METADATA_PATH, TARGET_COLUMN
from src.data_loader import detect_label_column, safe_read_csv
from src.preprocessing import clean_dataset, drop_identifier_columns, normalise_label_values

logger = logging.getLogger(__name__)


def _read_csv_input(value: Any) -> pd.DataFrame:
    if isinstance(value, (str, Path)):
        return safe_read_csv(Path(value), max_rows=None)
    if hasattr(value, 'read'):
        return pd.read_csv(value)
    raise TypeError('Prediction input must be a path or a file-like object.')


def align_features(train_features: List[str], new_frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = new_frame.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    missing = [col for col in train_features if col not in cleaned.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    extra = [col for col in cleaned.columns if col not in train_features]
    if extra:
        logger.warning('Extra columns ignored in prediction input: %s', extra)
    return cleaned[train_features].copy()


def predict_from_frame(model, frame: pd.DataFrame, feature_names: List[str], label_column: str | None = None) -> Tuple[pd.DataFrame, pd.Series]:
    cleaned = frame.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    cleaned = cleaned.replace([float('inf'), float('-inf')], float('nan'))
    cleaned = drop_identifier_columns(cleaned)
    if label_column is not None and label_column in cleaned.columns:
        actual = normalise_label_values(cleaned[label_column])
    else:
        actual = None

    aligned = align_features(feature_names, cleaned)
    for column in aligned.columns:
        aligned[column] = pd.to_numeric(aligned[column], errors='coerce')
    probabilities = model.predict_proba(aligned)
    top_prob = probabilities.max(axis=1)
    prediction = model.predict(aligned)
    results = pd.DataFrame({
        'predicted_label': prediction,
        'confidence': top_prob,
    })
    if actual is not None:
        results['actual_label'] = actual.reset_index(drop=True)
    return results, pd.Series(prediction)


def load_model_and_metadata(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    import joblib

    model = joblib.load(model_path)
    with open(metadata_path, 'r', encoding='utf-8') as handle:
        metadata = json.load(handle)
    return model, metadata


def run_prediction_on_csv(csv_path: str | Path | Any, model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    model, metadata = load_model_and_metadata(model_path, metadata_path)
    df = _read_csv_input(csv_path)
    feature_names = metadata.get('feature_names', [])
    if not feature_names:
        raise ValueError('No feature names saved in model metadata.')
    label_col = None
    if 'Label' in [str(col).strip() for col in df.columns]:
        label_col = detect_label_column(df)
    predictions, _ = predict_from_frame(model, df, feature_names, label_column=label_col)
    prediction_output = df.copy()
    prediction_output = prediction_output.reset_index(drop=True)
    prediction_output['predicted_label'] = predictions['predicted_label'].reset_index(drop=True)
    prediction_output['confidence'] = predictions['confidence'].reset_index(drop=True)
    if label_col is not None:
        prediction_output['actual_label'] = df[label_col].reset_index(drop=True)
    return prediction_output
