import numpy as np
import pandas as pd

from src.preprocessing import clean_dataset


def test_clean_dataset_handles_malformed_columns_and_infs():
    df_one = pd.DataFrame({
        ' Flow ID ': ['f1', 'f2', 'f3'],
        ' Source IP ': ['10.0.0.1', '10.0.0.2', '10.0.0.3'],
        'Timestamp': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'Fwd Header Length': [100, 200, 100],
    })
    df_two = pd.DataFrame({
        'Fwd Header Length': [100, 200, 100],
        'Flow Bytes/s': [12.5, np.inf, np.nan],
        'Flow Packets/s': [4.0, -np.inf, 6.0],
        'Label': ['BENIGN', 'DoS Hulk', 'BENIGN'],
    })
    df = pd.concat([df_one, df_two], axis=1)
    cleaned, label_name, summary = clean_dataset(df)
    assert label_name == 'Label'
    assert 'Flow ID' not in cleaned.columns
    assert 'Source IP' not in cleaned.columns
    assert 'Timestamp' not in cleaned.columns
    assert cleaned['Label'].str.strip().nunique() >= 2
    assert cleaned['Flow Bytes/s'].isna().sum() >= 1
    assert cleaned['Flow Packets/s'].isna().sum() >= 1
    assert summary['duplicates_removed'] >= 0
