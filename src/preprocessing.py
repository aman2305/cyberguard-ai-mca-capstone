import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold

from config import CORRELATION_THRESHOLD, IDENTIFIER_COLUMNS, RANDOM_STATE, TARGET_COLUMN

logger = logging.getLogger(__name__)


class DropCorrelatedFeatures(BaseEstimator, TransformerMixin):
    """Remove highly correlated features using the training data only."""

    def __init__(self, threshold: float = CORRELATION_THRESHOLD):
        self.threshold = threshold
        self.kept_features_ = []

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X)

        corr_matrix = frame.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []
        for column in upper.columns:
            if (upper[column] > self.threshold).any():
                to_drop.append(column)
        self.kept_features_ = [col for col in frame.columns if col not in set(to_drop)]
        self.feature_names_in_ = list(frame.columns)
        return self

    def transform(self, X, y=None):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names_in_)
        if not getattr(self, 'kept_features_', None):
            return frame
        return frame[self.kept_features_]


def normalise_label_values(series: pd.Series) -> pd.Series:
    cleaned = series.copy()
    cleaned = cleaned.astype(str)
    cleaned = cleaned.str.replace('\ufeff', '', regex=False)
    cleaned = cleaned.str.strip()
    cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
    cleaned = cleaned.replace({'nan': 'UNKNOWN', 'None': 'UNKNOWN'})
    return cleaned


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    dropped = []
    for column in list(cleaned.columns):
        name = str(column).strip().lower()
        if name in IDENTIFIER_COLUMNS or any(token in name for token in ('flow id', 'source ip', 'destination ip', 'source port', 'timestamp')):
            dropped.append(column)
    if dropped:
        cleaned = cleaned.drop(columns=dropped)
    return cleaned


def clean_dataset(df: pd.DataFrame, label_column: Optional[str] = None) -> Tuple[pd.DataFrame, str, Dict[str, object]]:
    working = df.copy()
    working.columns = [str(col).strip() for col in working.columns]
    working = working.loc[:, ~working.columns.duplicated(keep='first')]
    working = working.replace([np.inf, -np.inf], np.nan)

    if label_column is None:
        label_column = next((col for col in working.columns if str(col).strip().lower() == 'label'), None)
    if label_column is None:
        raise ValueError('A Label column is required for training.')

    working = drop_identifier_columns(working)
    feature_columns = [col for col in working.columns if col != label_column]
    for column in feature_columns:
        working[column] = pd.to_numeric(working[column], errors='coerce')

    labels = normalise_label_values(working[label_column])
    working = working.drop(columns=[label_column])
    working['Label'] = labels
    feature_columns = [col for col in working.columns if col != 'Label']
    valid_dataframe = working.dropna(axis=0, how='all')
    valid_dataframe = valid_dataframe.drop_duplicates()

    summary = {
        'rows_before': int(len(df)),
        'rows_after': int(len(valid_dataframe)),
        'columns_before': int(len(df.columns)),
        'columns_after': int(len(valid_dataframe.columns)),
        'duplicates_removed': int(len(df) - len(valid_dataframe)),
    }
    return valid_dataframe, 'Label', summary


def prepare_feature_matrix(dataframe: pd.DataFrame, label_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    features = dataframe.drop(columns=[label_column], errors='ignore')
    labels = dataframe[label_column].copy()
    feature_frame = features.copy()
    for column in feature_frame.columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors='coerce')
    return feature_frame, labels


def build_pipeline() -> object:
    pipeline = __import__('sklearn.pipeline', fromlist=['Pipeline']).Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('variance_filter', VarianceThreshold()),
        ('corr_filter', DropCorrelatedFeatures()),
        ('model', __import__('sklearn.ensemble', fromlist=['RandomForestClassifier']).RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
            class_weight='balanced',
            n_jobs=-1,
        )),
    ])
    return pipeline


def safe_stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    counts = y.value_counts()
    rare_classes = counts[counts < 2].index.tolist()
    if rare_classes:
        message = f"The following classes have fewer than 2 samples and are excluded from stratified splitting: {rare_classes}."
        filtered_mask = ~y.isin(rare_classes)
        X_valid = X.loc[filtered_mask].copy()
        y_valid = y.loc[filtered_mask].copy()
        if X_valid.empty or y_valid.empty:
            raise ValueError('No valid rows remain after excluding rare classes.')
        return train_test_split(X_valid, y_valid, test_size=test_size, random_state=RANDOM_STATE, stratify=y_valid), message
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y)
    return (X_train, X_test, y_train, y_test), None
