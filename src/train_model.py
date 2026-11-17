import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd

from config import DATA_DIR, METADATA_PATH, MODEL_PATH, RANDOM_STATE
from src.data_loader import load_dataset
from src.evaluation import compute_metrics
from src.preprocessing import build_pipeline, clean_dataset, prepare_feature_matrix, safe_stratified_split

logger = logging.getLogger(__name__)


def train_model(data_dir: Path = DATA_DIR, synthetic: bool = False, max_rows_per_file: Optional[int] = None) -> Dict[str, object]:
    dataset, source_files = load_dataset(data_dir, max_rows_per_file=max_rows_per_file)
    if dataset.empty:
        raise ValueError('No CSV files were found in the dataset directory.')

    cleaned, label_column, summary = clean_dataset(dataset)
    X, y = prepare_feature_matrix(cleaned, label_column)

    split_result, warning_message = safe_stratified_split(X, y)
    X_train, X_test, y_train, y_test = split_result

    pipeline = build_pipeline()
    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - start_time

    predictions = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, pd.Series(predictions), labels=sorted(set(y.unique())))
    feature_names = X_train.columns.tolist()
    metadata = {
        'feature_names': feature_names,
        'class_labels': sorted(set(y.unique())),
        'training_count': int(len(X_train)),
        'testing_count': int(len(X_test)),
        'training_time_seconds': round(elapsed, 4),
        'dataset_source_files': source_files,
        'synthetic': synthetic,
        'timestamp_utc': int(time.time()),
        'metrics': metrics,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2)

    return {
        'model': pipeline,
        'metadata': metadata,
        'warning': warning_message,
        'y_true': y_test,
        'y_pred': pd.Series(predictions),
        'summary': summary,
    }


def load_trained_model(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    model = joblib.load(model_path)
    with open(metadata_path, 'r', encoding='utf-8') as handle:
        metadata = json.load(handle)

    if not metadata.get('feature_names'):
        raise ValueError('Metadata is missing feature names.')
    if not hasattr(model, 'predict'):
        raise ValueError('Model file is invalid or incompatible.')
    return model, metadata


def verify_model_metadata(model_path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    model, metadata = load_trained_model(model_path, metadata_path)
    feature_names = metadata.get('feature_names', [])
    if not isinstance(feature_names, list):
        raise ValueError('Metadata feature names must be a list.')
    if not model:
        raise ValueError('The saved model could not be loaded.')
    return model, metadata
