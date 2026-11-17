# CICIDS2017 data placement

This project is designed to work with the Canadian Institute for Cybersecurity's CIC-IDS2017 dataset (MachineLearningCSV files).

## Download
Use the official source:
- Canadian Institute for Cybersecurity, University of New Brunswick
- "Intrusion Detection Evaluation Dataset (CIC-IDS2017)"
- MachineLearningCSV files

## Place files here
Copy the CSV files into this folder before training:

`cyber-threat-detection/data/`

Examples:
- `data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv`
- `data/Tuesday-WorkingHours.pcap_ISCX.csv`

## Synthetic sample
A synthetic test sample can be generated for interface testing and validation:

`data/SYNTHETIC_SAMPLE_NOT_CICIDS2017.csv`

From the project root, run `python scripts/generate_synthetic_sample.py`.

This sample includes duplicate rows, NaN/inf values, rare classes, and malformed columns to exercise the cleaning pipeline. It is clearly labelled as synthetic and must not be used as research evidence.

## Notes
- The application reads all CSV files in `data/` and concatenates them.
- The repository does not bundle datasets. If the real dataset is absent, the app will run without a dataset; generate the synthetic sample above for UI and pipeline testing.
- Real CICIDS2017 model evaluation is pending until actual dataset files are downloaded and trained.
