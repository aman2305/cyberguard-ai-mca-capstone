import logging

import pandas as pd
import streamlit as st

from config import ASSETS_DIR, DATA_DIR, OUTPUTS_DIR
from src.data_loader import safe_read_csv, validate_dataset
from src.predict import run_prediction_on_csv
from src.train_model import train_model
from src.visualization import plot_dataset_overview

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="CYBERGUARD AI | MCA Capstone",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    'Dashboard',
    'Dataset Upload',
    'Data Preprocessing',
    'Model Training',
    'Threat Detection',
    'Model Evaluation',
    'Reports',
    'About Project',
]
SYNTHETIC_WARNING = "SYNTHETIC SAMPLE – NOT CICIDS2017 – NOT VALID EXPERIMENTAL RESULTS"


def load_css() -> None:
    css_path = ASSETS_DIR / 'style.css'
    if css_path.exists():
        with open(css_path, 'r', encoding='utf-8') as handle:
            st.markdown(f'<style>{handle.read()}</style>', unsafe_allow_html=True)


def sidebar_navigation() -> str:
    st.sidebar.markdown(
        '''
        <div class="sidebar-brand">
            <div class="sidebar-mark">CG</div>
            <div>
                <div class="sidebar-title">CYBERGUARD <span>AI</span></div>
                <div class="sidebar-subtitle">NETWORK THREAT ANALYSIS</div>
            </div>
        </div>
        <div class="sidebar-label">PROJECT WORKSPACE</div>
        ''',
        unsafe_allow_html=True,
    )
    return st.sidebar.radio(
        'Navigation',
        PAGES,
        index=0,
        label_visibility='collapsed',
        key='navigation',
    )


def ensure_session_state():
    for key in [
        'dataset',
        'cleaned_dataset',
        'model',
        'metadata',
        'prediction_frame',
        'outputs',
        'dataset_sources',
        'dataset_is_synthetic',
    ]:
        if key not in st.session_state:
            st.session_state[key] = [] if key == 'dataset_sources' else False if key == 'dataset_is_synthetic' else None


def render_project_header() -> None:
    st.markdown(
        '''
        <section class="project-hero">
            <div class="hero-topline">
                <span class="hero-kicker">MCA SEMESTER III &nbsp;·&nbsp; CAPSTONE PROJECT</span>
                <span class="offline-pill"><span class="offline-dot"></span> OFFLINE · DATASET-BASED</span>
            </div>
            <div class="hero-main">
                <div class="hero-emblem">CG</div>
                <div>
                    <h1 class="hero-title">CYBERGUARD <span>AI</span></h1>
                    <p class="hero-subtitle">Machine Learning-Based Network Anomaly Detection and Threat Classification</p>
                    <p class="hero-tagline">AI-Powered Network Threat Detection &amp; Analysis</p>
                </div>
            </div>
            <div class="student-strip">
                <div class="student-item"><span>PREPARED BY</span><strong>Aman Lakhera</strong></div>
                <div class="student-item"><span>PROGRAMME</span><strong>MCA</strong></div>
                <div class="student-item"><span>INSTITUTION</span><strong>Dnyaan Prasad Global University · School of Management &amp; Research, Pune</strong></div>
                <div class="student-item"><span>ACADEMIC YEAR</span><strong>2026–2027</strong></div>
            </div>
        </section>
        ''',
        unsafe_allow_html=True,
    )


def render_section_title(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="section-heading"><h2>{title}</h2><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def show_synthetic_notice() -> None:
    st.warning(SYNTHETIC_WARNING, icon="⚠️")


ensure_session_state()
load_css()
render_project_header()

page = sidebar_navigation()

if page == 'Dashboard':
    render_section_title('Research dashboard', 'A transparent view of the currently loaded dataset and saved model evaluation.')
    dataset = st.session_state.get('dataset')
    if dataset is None or dataset.empty:
        st.markdown(
            '''
            <div class="welcome-card">
                <div class="welcome-copy">
                    <span class="eyebrow">PROJECT OVERVIEW</span>
                    <h3>From network-flow records to explainable evaluation.</h3>
                    <p>Load labelled CSV data to inspect traffic classes, prepare features, train a Random Forest, and review its held-out results.</p>
                    <div class="welcome-note">No dataset loaded · No model metrics are shown until they are computed.</div>
                </div>
                <div class="welcome-art" aria-hidden="true">
                    <div class="orbit orbit-one"></div><div class="orbit orbit-two"></div>
                    <div class="core-shield">CG</div>
                    <span class="orbit-node node-one"></span><span class="orbit-node node-two"></span><span class="orbit-node node-three"></span>
                </div>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="flow-label">ANALYSIS PIPELINE</div>', unsafe_allow_html=True)
        pipeline_cols = st.columns(5)
        for column, (number, title, detail) in zip(
            pipeline_cols,
            [
                ('01', 'Dataset', 'Labelled CSV flows'),
                ('02', 'Prepare', 'Clean & select features'),
                ('03', 'Train', 'Stratified Random Forest'),
                ('04', 'Evaluate', 'Held-out test metrics'),
                ('05', 'Review', 'Predictions & reports'),
            ],
        ):
            with column:
                st.markdown(
                    f'<div class="pipeline-card"><span>{number}</span><strong>{title}</strong><small>{detail}</small></div>',
                    unsafe_allow_html=True,
                )
    else:
        if st.session_state.get('dataset_is_synthetic'):
            show_synthetic_notice()
        label_column = next((column for column in dataset.columns if str(column).strip().lower() == 'label'), None)
        if label_column is None:
            st.error("The loaded dataset has no Label column. Use Dataset Upload to select a labelled CSV.")
        else:
            total_records = len(dataset)
            labels = dataset[label_column].astype(str).str.strip()
            benign_count = int(labels.str.casefold().eq('benign').sum())
            attack_count = total_records - benign_count
            col1, col2, col3 = st.columns(3)
            col1.metric('Records analysed', f'{total_records:,}', help='Rows in the currently loaded dataset.')
            col2.metric('BENIGN records', f'{benign_count:,}')
            col3.metric('Attack-labelled records', f'{attack_count:,}')

            chart_col, model_col = st.columns([1.5, 1], gap='large')
            with chart_col:
                st.markdown('<div class="panel-heading"><span>01</span><div><h3>Traffic distribution</h3><p>Observed label counts in the loaded dataset</p></div></div>', unsafe_allow_html=True)
                st.plotly_chart(plot_dataset_overview(dataset.rename(columns={label_column: 'Label'})), use_container_width=True)
            with model_col:
                st.markdown('<div class="panel-heading"><span>02</span><div><h3>Model status</h3><p>Evaluation summary from this session</p></div></div>', unsafe_allow_html=True)
                if st.session_state.metadata:
                    metrics = st.session_state.metadata.get('metrics', {})
                    if st.session_state.metadata.get('synthetic'):
                        show_synthetic_notice()
                    macro_f1 = metrics.get('f1_macro')
                    if macro_f1 is not None:
                        st.metric('Held-out macro F1', f'{macro_f1:.3f}')
                    st.caption(f"Training rows: {st.session_state.metadata.get('training_count', '—')} · Test rows: {st.session_state.metadata.get('testing_count', '—')}")
                else:
                    st.markdown('<div class="empty-panel"><span class="status-chip muted">NOT TRAINED</span><h4>Model not trained yet</h4><p>Train a model to see computed test-set results here.</p></div>', unsafe_allow_html=True)

            recent_files = sorted(path for path in OUTPUTS_DIR.glob('*') if path.is_file() and path.name != '.gitkeep') if OUTPUTS_DIR.exists() else []
            st.markdown('<div class="panel-heading"><span>03</span><div><h3>Recent analysis outputs</h3><p>Files currently present in the outputs folder</p></div></div>', unsafe_allow_html=True)
            if recent_files:
                st.write(' · '.join(path.name for path in recent_files))
            else:
                st.caption('No analysis output files yet.')

elif page == 'Dataset Upload':
    render_section_title('Dataset upload', 'Import labelled CSV flow records or select files already placed in the project data folder.')
    uploaded = st.file_uploader('Upload CSV(s)', type=['csv'], accept_multiple_files=True)
    if uploaded:
        try:
            frames = []
            for item in uploaded:
                try:
                    frame = pd.read_csv(item, encoding='utf-8', low_memory=False)
                except UnicodeDecodeError:
                    item.seek(0)
                    frame = pd.read_csv(item, encoding='latin-1', low_memory=False)
                frames.append(frame)
            dataset = pd.concat(frames, ignore_index=True, sort=False)
            label_column = next((column for column in dataset.columns if str(column).strip().lower() == 'label'), None)
            if label_column is None:
                raise ValueError("No Label column found. Include a labelled target column in the CSV.")
            if label_column != 'Label':
                dataset = dataset.rename(columns={label_column: 'Label'})
            validate_dataset(dataset)
            st.session_state.dataset = dataset
            st.session_state.dataset_sources = [item.name for item in uploaded]
            st.session_state.dataset_is_synthetic = any('SYNTHETIC_SAMPLE_NOT_CICIDS2017' in item.name for item in uploaded)
            st.success(f'Loaded {len(dataset):,} rows from {len(uploaded)} file(s).')
        except (pd.errors.ParserError, ValueError, UnicodeDecodeError, OSError) as exc:
            logging.exception('Dataset upload failed')
            st.error(f'Could not load the uploaded dataset: {exc}')
    available_files = [path.name for path in DATA_DIR.glob('*.csv') if path.is_file()]
    if available_files:
        st.markdown('<div class="upload-subhead">OR SELECT A PROJECT DATA FILE</div>', unsafe_allow_html=True)
        selected = st.selectbox('CSV file in data/', [''] + available_files, label_visibility='collapsed')
        if selected:
            try:
                dataset = safe_read_csv(DATA_DIR / selected, max_rows=None)
                label_column = next((column for column in dataset.columns if str(column).strip().lower() == 'label'), None)
                if label_column is None:
                    raise ValueError("No Label column found. Include a labelled target column in the CSV.")
                if label_column != 'Label':
                    dataset = dataset.rename(columns={label_column: 'Label'})
                validate_dataset(dataset)
                st.session_state.dataset = dataset
                st.session_state.dataset_sources = [selected]
                st.session_state.dataset_is_synthetic = 'SYNTHETIC_SAMPLE_NOT_CICIDS2017' in selected
                st.success(f'Loaded {selected} from data/.')
            except (pd.errors.ParserError, ValueError, UnicodeDecodeError, OSError) as exc:
                logging.exception('Dataset file selection failed')
                st.error(f'Could not load {selected}: {exc}')
    dataset = st.session_state.get('dataset')
    if dataset is not None and not dataset.empty:
        if st.session_state.get('dataset_is_synthetic'):
            show_synthetic_notice()
        metrics_cols = st.columns(4)
        missing_count = int(dataset.isna().sum().sum())
        duplicate_count = int(dataset.duplicated().sum())
        metrics_cols[0].metric('Rows', f'{dataset.shape[0]:,}')
        metrics_cols[1].metric('Columns', f'{dataset.shape[1]:,}')
        metrics_cols[2].metric('Missing values', f'{missing_count:,}')
        metrics_cols[3].metric('Exact duplicates', f'{duplicate_count:,}')
        preview_col, distribution_col = st.columns([1.2, 1], gap='large')
        with preview_col:
            st.markdown('<div class="panel-heading"><span>01</span><div><h3>Data preview</h3><p>First five rows from the loaded file(s)</p></div></div>', unsafe_allow_html=True)
            st.dataframe(dataset.head(), use_container_width=True, hide_index=True)
        with distribution_col:
            st.markdown('<div class="panel-heading"><span>02</span><div><h3>Label distribution</h3><p>Class counts observed in this dataset</p></div></div>', unsafe_allow_html=True)
            st.dataframe(dataset['Label'].astype(str).value_counts().rename_axis('Label').reset_index(name='Records'), use_container_width=True, hide_index=True)
            st.plotly_chart(plot_dataset_overview(dataset), use_container_width=True)

elif page == 'Data Preprocessing':
    render_section_title('Data preprocessing', 'Review the shared cleaning steps used before model training and threat prediction.')
    dataset = st.session_state.get('dataset')
    if dataset is None or dataset.empty:
        st.info('No dataset loaded. Open Dataset Upload to begin.')
    else:
        from src.preprocessing import clean_dataset
        cleaned, label_column, summary = clean_dataset(dataset)
        st.session_state.cleaned_dataset = cleaned
        if st.session_state.get('dataset_is_synthetic'):
            show_synthetic_notice()
        metric_cols = st.columns(4)
        metric_cols[0].metric('Rows before', f'{summary["rows_before"]:,}')
        metric_cols[1].metric('Rows after', f'{summary["rows_after"]:,}')
        metric_cols[2].metric('Exact duplicates removed', f'{summary.get("duplicates_removed", 0):,}')
        metric_cols[3].metric('Columns after cleaning', f'{summary["columns_after"]:,}')
        st.markdown('<div class="process-note"><strong>Pipeline note</strong><span>Infinite values are converted to missing values; median imputation, zero-variance filtering, and correlation filtering are fitted within the training pipeline.</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel-heading"><span>01</span><div><h3>Cleaned data preview</h3><p>Features and normalized target labels</p></div></div>', unsafe_allow_html=True)
        st.dataframe(cleaned.head(), use_container_width=True, hide_index=True)

elif page == 'Model Training':
    render_section_title('Model training', 'Fit the Random Forest pipeline and evaluate it on a held-out test split.')
    dataset = st.session_state.get('dataset')
    if dataset is None or dataset.empty:
        st.info('No dataset loaded. Upload a labelled CSV before training.')
    else:
        if st.session_state.get('dataset_is_synthetic'):
            show_synthetic_notice()
        st.markdown('<div class="process-note"><strong>Leakage-safe workflow</strong><span>Stratified split → fit preprocessing on training rows → Random Forest → test-set evaluation → save model metadata.</span></div>', unsafe_allow_html=True)
        result = st.button('Train model')
        if result:
            try:
                with st.spinner('Fitting pipeline and evaluating the held-out test set...'):
                    training_result = train_model(
                        DATA_DIR,
                        synthetic=bool(st.session_state.get('dataset_is_synthetic')),
                        dataset=dataset,
                        dataset_sources=st.session_state.get('dataset_sources'),
                    )
                st.session_state.model = training_result['model']
                st.session_state.metadata = training_result['metadata']
                st.session_state.outputs = training_result
                st.success('Training and evaluation completed.')
                if training_result.get('warning'):
                    st.warning(training_result['warning'])
                if st.session_state.metadata.get('synthetic'):
                    show_synthetic_notice()
                st.metric('Held-out macro F1', f"{st.session_state.metadata['metrics']['f1_macro']:.3f}")
                st.caption(f"Training rows: {st.session_state.metadata['training_count']:,} · Test rows: {st.session_state.metadata['testing_count']:,} · Fit time: {st.session_state.metadata['training_time_seconds']:.2f} s")
            except (ValueError, OSError) as exc:
                logging.exception('Model training failed')
                st.error(f'Training could not be completed: {exc}')

elif page == 'Threat Detection':
    render_section_title('Threat detection', 'Classify uploaded flow records with the saved model. Predictions are model outputs, not confirmed attacks.')
    if st.session_state.get('model') is None:
        st.markdown('<div class="empty-panel"><span class="status-chip muted">MODEL REQUIRED</span><h4>Model not trained yet</h4><p>Train the model first, then upload a labelled or unlabelled CSV for prediction.</p></div>', unsafe_allow_html=True)
    else:
        uploaded = st.file_uploader('Upload CSV for prediction', type=['csv'])
        if uploaded is not None:
            if 'SYNTHETIC_SAMPLE_NOT_CICIDS2017' in uploaded.name:
                show_synthetic_notice()
            try:
                result_frame = run_prediction_on_csv(uploaded)
                st.session_state.prediction_frame = result_frame
                st.caption('Labels below are predicted by the model; they are not confirmed attacks.')
                category = st.selectbox('Filter predicted category', ['All'] + sorted(result_frame['predicted_label'].astype(str).unique().tolist()))
                visible = result_frame if category == 'All' else result_frame[result_frame['predicted_label'].astype(str) == category]
                st.dataframe(visible, use_container_width=True, hide_index=True)
                st.download_button('Download prediction results', result_frame.to_csv(index=False).encode('utf-8'), file_name='cyberguard_predictions.csv', mime='text/csv')
                st.markdown('<div class="panel-heading"><span>01</span><div><h3>Predicted category counts</h3><p>Counts computed from the uploaded file</p></div></div>', unsafe_allow_html=True)
                counts = result_frame['predicted_label'].value_counts().rename_axis('Predicted label').reset_index(name='Records')
                st.dataframe(counts, use_container_width=True, hide_index=True)
                if 'actual_label' in result_frame.columns:
                    st.warning('This file includes labels and may overlap with training data. Comparison results are only indicative.')
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                logging.exception('Threat prediction failed')
                st.error(f'Prediction could not be completed: {exc}')

elif page == 'Model Evaluation':
    render_section_title('Model evaluation', 'Held-out test-set metrics, with attention to class imbalance and per-class recall.')
    metadata = st.session_state.get('metadata')
    if metadata is None:
        st.info('Model not trained yet. Evaluation metrics will appear here after training.')
    else:
        if metadata.get('synthetic'):
            show_synthetic_notice()
        metrics = metadata.get('metrics', {})
        st.warning('Accuracy alone can be misleading on imbalanced data. Review macro-F1 and per-class recall as well.')
        metric_cols = st.columns(4)
        for column, label, key in [
            (metric_cols[0], 'Accuracy', 'accuracy'),
            (metric_cols[1], 'Macro precision', 'precision_macro'),
            (metric_cols[2], 'Macro recall', 'recall_macro'),
            (metric_cols[3], 'Macro F1', 'f1_macro'),
        ]:
            value = metrics.get(key)
            column.metric(label, f'{value:.3f}' if isinstance(value, (int, float)) else '—')
        report = metrics.get('classification_report', {})
        class_rows = [
            {'Class': name, **values}
            for name, values in report.items()
            if isinstance(values, dict) and 'f1-score' in values
        ]
        if class_rows:
            st.markdown('<div class="panel-heading"><span>01</span><div><h3>Class-wise report</h3><p>Precision, recall, and F1 computed on the held-out test set</p></div></div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame(class_rows), use_container_width=True, hide_index=True)
        else:
            st.info('No per-class report is present in the saved metadata.')

elif page == 'Reports':
    render_section_title('Reports', 'Download evaluation and prediction artifacts created by this project.')
    if not OUTPUTS_DIR.exists():
        st.info('No report files exist yet. Train the model or run a prediction first.')
    else:
        files = sorted(path for path in OUTPUTS_DIR.glob('*') if path.is_file() and path.name != '.gitkeep')
        if not files:
            st.info('No report files exist yet. Train the model or run a prediction first.')
        else:
            for file in files:
                file_col, button_col = st.columns([4, 1])
                file_col.markdown(f'<div class="file-row"><strong>{file.name}</strong><span>{file.stat().st_size:,} bytes</span></div>', unsafe_allow_html=True)
                button_col.download_button('Download', file.read_bytes(), file_name=file.name, key=f'download_{file.name}')

elif page == 'About Project':
    render_section_title('About the project', 'Purpose, system architecture, algorithm, technology, and research limitations.')
    about_col, algorithm_col = st.columns([1.25, 1], gap='large')
    with about_col:
        st.markdown('<div class="content-panel"><span class="eyebrow">OBJECTIVE</span><h3>Study data-driven network threat classification.</h3><p>CYBERGUARD AI is an offline academic prototype that analyses labelled network-flow records and predicts BENIGN or attack categories using a Random Forest classifier.</p><span class="eyebrow">DATASET</span><p>Designed for CICIDS2017 MachineLearningCSV flow records. Dataset labels and preprocessing assumptions affect the results.</p></div>', unsafe_allow_html=True)
    with algorithm_col:
        st.markdown('<div class="content-panel"><span class="eyebrow">ALGORITHM</span><h3>Random Forest</h3><p>An ensemble of decision trees learns patterns from the training split. Class balancing and held-out evaluation help expose performance beyond overall accuracy.</p><span class="eyebrow">TECHNOLOGY</span><p>Python · pandas · NumPy · scikit-learn · joblib · Streamlit · Plotly · matplotlib · seaborn</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="flow-label">SYSTEM ARCHITECTURE</div>', unsafe_allow_html=True)
    architecture = st.columns(5)
    for column, (number, title, detail) in zip(
        architecture,
        [
            ('01', 'Dataset', 'CICIDS2017 CSV'),
            ('02', 'Clean', 'Normalize & deduplicate'),
            ('03', 'Prepare', 'Impute & select features'),
            ('04', 'Classify', 'Random Forest'),
            ('05', 'Evaluate', 'Held-out metrics'),
        ],
    ):
        with column:
            st.markdown(f'<div class="pipeline-card"><span>{number}</span><strong>{title}</strong><small>{detail}</small></div>', unsafe_allow_html=True)
    st.markdown('<div class="limitations-panel"><span class="eyebrow">LIMITATIONS</span><ul><li>Offline dataset analysis only; no live packet capture or automated blocking.</li><li>CICIDS2017 is from 2017 and has known labelling and lab-environment artefacts.</li><li>High scores may reflect dataset characteristics and may not generalize to real networks.</li><li>Rare classes can have unreliable per-class metrics.</li></ul></div>', unsafe_allow_html=True)

st.markdown('<div class="app-footer"><span>CYBERGUARD AI · MCA CAPSTONE · 2026–2027</span><span>OFFLINE DATASET ANALYSIS</span></div>', unsafe_allow_html=True)
