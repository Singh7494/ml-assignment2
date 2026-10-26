from pathlib import Path
import csv
import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from evaluation import compute_metrics, validate_data

ROOT = Path(__file__).resolve().parent
MODELS = {
    'Logistic Regression': 'logistic_regression.pkl',
    'Decision Tree': 'decision_tree.pkl',
    'KNN': 'knn.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Support Vector Machine': 'svm.pkl',
}

st.set_page_config(page_title='Breast Cancer Classification Dashboard',
                   page_icon='🧬', layout='wide')
# Hide Streamlit's top-right developer toolbar in local and lab environments.
st.set_option('client.toolbarMode', 'minimal')
st.markdown(
    """
    <style>
    [data-testid="stToolbar"],
    [data-testid="stToolbarActions"],
    .stAppToolbar,
    [data-testid="stAppDeployButton"],
    .stAppDeployButton,
    .stDeployButton,
    [data-testid="stDeployButton"],
    #MainMenu {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title('🧬 Breast Cancer Classification Dashboard')
st.caption('BITS Pilani | M.Tech AIML | Machine Learning Assignment 2')
st.write('Compare five classifiers on held-out test data: 30 features, '
         '0 = malignant, 1 = benign. Precision, recall and F1 below refer to benign (1).')


@st.cache_resource
def load_model(filename):
    return joblib.load(ROOT / 'model' / filename)


selected = st.sidebar.selectbox(
    'Select model', list(MODELS),
    help='Show detailed evaluation for this model. The comparison below includes all five models.',
)
st.sidebar.write('Upload test data with the 30 original, unscaled feature columns '
                 'and a target column. Preprocessing is included in each saved model.')
sample = ROOT / 'test_data.csv'
if sample.exists():
    st.sidebar.download_button('Download held-out test CSV', sample.read_bytes(),
                               file_name='test_data.csv', mime='text/csv')
upload = st.sidebar.file_uploader('Upload test CSV', type=['csv'])
if upload is None:
    st.info('Download the test CSV, upload it, and run evaluation for all five models.')
    st.stop()

try:
    content = upload.getvalue().decode('utf-8-sig')
    header = next(csv.reader(io.StringIO(content)), [])
    if len(header) != len(set(header)):
        raise ValueError('Duplicate column names are not supported.')
    df = pd.read_csv(io.StringIO(content))
    reference_model = load_model(next(iter(MODELS.values())))
    X, y = validate_data(df, reference_model.feature_names_in_)
except Exception as exc:
    st.error(f'Cannot load test data or model: {exc}')
    st.stop()

with st.expander('Preview uploaded test data'):
    st.dataframe(df.head(), width='stretch')
    st.write(f'{len(df)} rows; {X.shape[1]} input features')
    st.write(y.value_counts().reindex([0, 1], fill_value=0).rename(
        index={0: 'Malignant', 1: 'Benign'}))

# Retain results across Streamlit reruns, but invalidate them when input changes.
run_key = upload.getvalue()
if st.session_state.get('evaluation_key') != run_key:
    st.session_state.pop('evaluation_results', None)
if st.button('Run Evaluation', type='primary'):
    results, predictions, errors = {}, {}, {}
    with st.spinner('Evaluating saved models…'):
        for name, filename in MODELS.items():
            try:
                model = load_model(filename)
                model_X, model_y = validate_data(df, model.feature_names_in_)
                results[name], predictions[name] = compute_metrics(model, model_X, model_y)
            except Exception as exc:
                errors[name] = str(exc)
    st.session_state['evaluation_key'] = run_key
    st.session_state['evaluation_results'] = (results, predictions, errors)

if 'evaluation_results' not in st.session_state:
    st.stop()
results, predictions, errors = st.session_state['evaluation_results']
for name, message in errors.items():
    st.error(f'{name}: {message}')
if y.nunique() < 2:
    st.warning('AUC is unavailable because the uploaded test data contains only one class.')
# Show detailed evaluation only for the selected model.
if selected in results:
    model_name = selected
    st.divider()
    st.subheader(f'Evaluation: {model_name}')
    for col, (name, value) in zip(st.columns(6), results[model_name].items()):
        col.metric(name, 'N/A' if pd.isna(value) else f'{value:.4f}')
    left, right = st.columns(2)
    with left:
        st.subheader('Confusion matrix')
        matrix = confusion_matrix(y, predictions[model_name], labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Malignant', 'Benign'], yticklabels=['Malignant', 'Benign'])
        ax.set(xlabel='Predicted class', ylabel='Actual class', title=model_name)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.subheader('Classification report')
        report = classification_report(y, predictions[model_name], labels=[0, 1],
                    target_names=['Malignant (0)', 'Benign (1)'],
                    output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(4), width='stretch')

if results:
    st.subheader('Compare all five models')
    comparison = pd.DataFrame.from_dict(results, orient='index')
    st.dataframe(comparison.round(4),
                 width='stretch')
    fig, ax = plt.subplots(figsize=(11, 5))
    comparison.plot.bar(ax=ax)
    ax.set_ylim(-1 if comparison.min().min() < 0 else 0, 1.1)
    ax.set_ylabel('Score')
    ax.set_xticklabels(comparison.index, rotation=15, ha='right')
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    for metric in ['Accuracy', 'F1', 'AUC']:
        valid = comparison[metric].dropna()
        if not valid.empty:
            leaders = valid.index[(valid - valid.max()).abs() < 1e-12]
            st.write(f'Best {metric}: {", ".join(leaders)} ({valid.max():.4f})')
    st.download_button('Download evaluation results', comparison.to_csv().encode(),
                       file_name='evaluation_results.csv', mime='text/csv')
