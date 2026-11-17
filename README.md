# CYBERGUARD AI

AI-Based Cyber Threat Detection System

## Objective
This project is an offline, dataset-based academic prototype for network anomaly detection and threat classification. It loads labelled network-flow CSV files, cleans and preprocesses the data, trains a Random Forest classifier, evaluates it on a held-out test set, and exposes the workflow through a Streamlit dashboard.

## Setup (Windows PowerShell)
```powershell
cd cyber-threat-detection
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks script activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

If your shell is not PowerShell, adapt the commands to your environment.

### Windows ARM64 troubleshooting
Do not install x64 wheels into an ARM64 Python environment. This workspace has an x64 Python interpreter at `C:\Python312-x64\python.exe`; use it to create a matching environment:
```powershell
C:\Python312-x64\python.exe -m venv .venv-x64
.\.venv-x64\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-x64\Scripts\python.exe -m streamlit run app.py
```

Confirm the environment reports `win-amd64` with ` .\.venv-x64\Scripts\python.exe -c "import sysconfig; print(sysconfig.get_platform())"`. If an existing Streamlit server is using the mismatched `.venv`, stop that server before replacing the environment.

## Dataset setup
If the real CICIDS2017 CSVs are not in `data/`, place them there as `.csv` files and follow the instructions in `data/README.md`.

The repository does not bundle datasets. For interface testing only, generate a clearly labelled synthetic sample with:
```powershell
python scripts/generate_synthetic_sample.py
```
This synthetic data must not be treated as a valid experimental result.

## Security note
Only load model files you trained yourself. Joblib and pickle files can execute code when loaded, so treat them as executable artifacts.

## Project structure
- `app.py` – Streamlit interface
- `src/data_loader.py` – dataset loading and validation
- `src/preprocessing.py` – cleaning and feature preparation
- `src/train_model.py` – model training and metadata persistence
- `src/evaluation.py` – metrics and export logic
- `src/predict.py` – inference on uploaded data
- `src/visualization.py` – Plotly and matplotlib charts
- `scripts/generate_synthetic_sample.py` – synthetic test dataset generator

## Limitations
- This is offline-only and does not perform live packet capture.
- CICIDS2017 is a 2017 benchmark dataset with known artefacts and labelling limitations.
- Rare classes may have unreliable metrics if they do not have enough examples.
- Results may not generalise beyond the lab environment.
