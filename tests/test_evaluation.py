import numpy as np
import pandas as pd
from sklearn import metrics

from src.evaluation import compute_metrics


def test_metrics_match_sklearn_independently():
    y_true = pd.Series(['BENIGN', 'BENIGN', 'DoS Hulk', 'DoS Hulk'])
    y_pred = pd.Series(['BENIGN', 'DoS Hulk', 'BENIGN', 'DoS Hulk'])
    report = compute_metrics(y_true, y_pred, labels=['BENIGN', 'DoS Hulk'])
    assert np.isclose(report['accuracy'], metrics.accuracy_score(y_true, y_pred))
    assert np.isclose(report['f1_macro'], metrics.f1_score(y_true, y_pred, average='macro', zero_division=0, labels=['BENIGN', 'DoS Hulk']))
    assert np.isclose(report['precision_weighted'], metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=['BENIGN', 'DoS Hulk']))
