import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parents[1] / 'data' / 'SYNTHETIC_SAMPLE_NOT_CICIDS2017.csv'

np.random.seed(42)

rows = []
records = 600
classes = ['BENIGN', 'DoS Hulk', 'PortScan', 'Web Attack Brute Force', 'Heartbleed', 'Infiltration']
weights = [0.52, 0.23, 0.12, 0.08, 0.03, 0.02]

for idx in range(records):
    label = np.random.choice(classes, p=weights)
    row = {
        'Flow ID': f'flow-{idx}',
        'Source IP': f'10.0.0.{(idx % 255) + 1}',
        'Destination IP': f'192.168.1.{(idx % 255) + 1}',
        'Source Port': int(np.random.randint(1000, 50000)),
        'Destination Port': int(np.random.randint(1, 65535)),
        'Protocol': int(np.random.randint(1, 7)),
        'Timestamp': f'2024-05-01 {idx % 24:02d}:00:00',
        'Flow Duration': int(np.random.randint(100, 1000000)),
        'Total Fwd Packets': int(np.random.randint(1, 200)),
        'Total Backward Packets': int(np.random.randint(0, 100)),
        'Fwd Header Length': int(np.random.randint(20, 200)),
        'Bwd Header Length': int(np.random.randint(20, 200)),
        'Flow Bytes/s': float(np.random.uniform(0, 500000)),
        'Flow Packets/s': float(np.random.uniform(0, 10000)),
        'Label': label,
    }
    if idx % 17 == 0:
        row['Flow Bytes/s'] = np.inf
    if idx % 23 == 0:
        row['Flow Packets/s'] = -np.inf
    if idx % 13 == 0:
        row['Flow ID'] = 'flow-dup'
    if idx % 31 == 0:
        row['Label'] = 'BENIGN'
    rows.append(row)

frame = pd.DataFrame(rows)
frame = pd.concat([frame, frame.iloc[[0, 3, 6]].copy()], ignore_index=True)
frame = frame.rename(columns={'Fwd Header Length': 'Fwd Header Length', 'Label': 'Label'})
frame.insert(1, 'Fwd Header Length', frame['Fwd Header Length'])
frame.iloc[0, frame.columns.get_loc('Flow Bytes/s')] = np.nan
frame.iloc[1, frame.columns.get_loc('Flow Packets/s')] = np.nan
frame.to_csv(OUTPUT_PATH, index=False)
print(f'Synthetic dataset generated at {OUTPUT_PATH}')
