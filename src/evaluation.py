import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import metrics

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, labels: Optional[List[str]] = None) -> Dict[str, float]:
    labels = labels or sorted(set(y_true).union(set(y_pred)))
    report = {
        'accuracy': float(metrics.accuracy_score(y_true, y_pred)),
        'precision_macro': float(metrics.precision_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'recall_macro': float(metrics.recall_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'f1_macro': float(metrics.f1_score(y_true, y_pred, average='macro', zero_division=0, labels=labels)),
        'precision_weighted': float(metrics.precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        'recall_weighted': float(metrics.recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        'f1_weighted': float(metrics.f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=labels)),
        'classification_report': metrics.classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True),
    }
    return report


def confusion_matrix_df(y_true: pd.Series, y_pred: pd.Series, labels: List[str]) -> pd.DataFrame:
    matrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_metrics_csv(metrics_dict: Dict[str, object], output_path: Path) -> None:
    rows = []
    for key, value in metrics_dict.items():
        if isinstance(value, dict):
            continue
        rows.append({'metric': key, 'value': value})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_classification_report_csv(y_true: pd.Series, y_pred: pd.Series, labels: List[str], output_path: Path) -> None:
    report = metrics.classification_report(y_true, y_pred, labels=labels, zero_division=0, output_dict=True)
    rows = []
    for class_name, values in report.items():
        if isinstance(values, dict):
            rows.append({'class': class_name, **values})
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_confusion_matrix_png(y_true: pd.Series, y_pred: pd.Series, labels: List[str], output_path: Path) -> None:
    matrix = confusion_matrix_df(y_true, y_pred, labels)
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.8), max(4, len(labels) * 0.8)))
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('Actual label')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_evaluation_outputs(y_true: pd.Series, y_pred: pd.Series, labels: List[str], output_dir: Path) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_dict = compute_metrics(y_true, y_pred, labels=labels)
    metrics_csv = output_dir / 'evaluation_metrics.csv'
    save_metrics_csv(metrics_dict, metrics_csv)
    metrics_json = output_dir / 'evaluation_metrics.json'
    metrics_json.write_text(json.dumps(metrics_dict, indent=2), encoding='utf-8')
    cm_png = output_dir / 'confusion_matrix.png'
    save_confusion_matrix_png(y_true, y_pred, labels, cm_png)
    report_csv = output_dir / 'classification_report.csv'
    save_classification_report_csv(y_true, y_pred, labels, report_csv)
    return {
        'metrics_csv': str(metrics_csv),
        'metrics_json': str(metrics_json),
        'confusion_matrix_png': str(cm_png),
        'classification_report_csv': str(report_csv),
    }
