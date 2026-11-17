import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from config import DATA_DIR, MAX_ROWS_DEFAULT

logger = logging.getLogger(__name__)


def normalise_column_name(column: object) -> str:
    if pd.isna(column):
        return ""
    return str(column).strip().replace("\ufeff", "").strip()


def remove_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [normalise_column_name(col) for col in cleaned.columns]
    cleaned = cleaned.loc[:, ~cleaned.columns.duplicated(keep='first')]
    return cleaned


def safe_read_csv(path: Path, max_rows: Optional[int] = None) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                low_memory=False,
                nrows=max_rows,
                na_values=["", "NaN", "nan", "NULL", "null"],
                keep_default_na=True,
            )
            df = remove_duplicate_columns(df)
            df = df.replace([np.inf, -np.inf], np.nan)
            return df
        except UnicodeDecodeError:
            continue
        except Exception:
            logger.exception("Failed to read %s with %s", path, encoding)
            raise
    raise ValueError(f"Unable to read dataset file: {path}")


def detect_label_column(df: pd.DataFrame) -> str:
    column_names = [normalise_column_name(col).lower() for col in df.columns]
    for idx, name in enumerate(column_names):
        if name == 'label':
            return df.columns[idx]
    raise ValueError("Label column not found in dataset. Expected a column named 'Label'.")


def get_csv_files(data_dir: Path = DATA_DIR) -> List[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.glob('*.csv') if path.is_file())


def load_dataset(data_dir: Path = DATA_DIR, max_rows_per_file: Optional[int] = MAX_ROWS_DEFAULT) -> Tuple[pd.DataFrame, List[str]]:
    files = get_csv_files(data_dir)
    if not files:
        return pd.DataFrame(), []

    frames = []
    for file_path in files:
        logger.info("Loading dataset from %s", file_path)
        frame = safe_read_csv(file_path, max_rows=max_rows_per_file)
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        return pd.DataFrame(), [str(path.name) for path in files]

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined, [str(path.name) for path in files]


def validate_dataset(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("The loaded dataset is empty.")
    if 'Label' not in [normalise_column_name(col) for col in df.columns]:
        raise ValueError("Dataset does not contain a Label column.")
    feature_columns = [col for col in df.columns if normalise_column_name(col).lower() != 'label']
    if not feature_columns:
        raise ValueError("No feature columns were found after cleaning.")
    numeric_count = 0
    for column in feature_columns:
        values = pd.to_numeric(df[column], errors='coerce')
        if values.notna().any():
            numeric_count += 1
    if numeric_count == 0:
        raise ValueError("No numeric feature columns were found.")
