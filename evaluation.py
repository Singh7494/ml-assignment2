"""Input validation and evaluation shared by the Streamlit application and checks."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)


def validate_data(df, feature_names):
    if df.empty:
        raise ValueError('The CSV must contain at least one test row.')
    if df.columns.duplicated().any():
        raise ValueError('Duplicate column names are not supported.')
    expected = list(feature_names) + ['target']
    missing = sorted(set(expected) - set(df.columns))
    extra = sorted(set(df.columns) - set(expected))
    if missing or extra:
        raise ValueError(f'CSV columns do not match. Missing: {missing}; unexpected: {extra}.')
    try:
        numeric = df.loc[:, expected].apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise ValueError('All feature values and target labels must be numeric.') from exc
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('Missing values and infinite values are not supported.')
    if not numeric['target'].isin([0, 1]).all():
        raise ValueError('Target labels must be 0 (malignant) or 1 (benign).')
    return numeric.loc[:, list(feature_names)], numeric['target'].astype(int)


def compute_metrics(model, X, y):
    pred = model.predict(X)
    classes = list(model.classes_)
    if classes != [0, 1]:
        raise ValueError('The saved model must use classes 0 and 1.')
    prob = model.predict_proba(X)[:, classes.index(1)]
    return {
        'Accuracy': accuracy_score(y, pred),
        'AUC': roc_auc_score(y, prob) if y.nunique() == 2 else np.nan,
        'Precision': precision_score(y, pred, pos_label=1, zero_division=0),
        'Recall': recall_score(y, pred, pos_label=1, zero_division=0),
        'F1': f1_score(y, pred, pos_label=1, zero_division=0),
        'MCC': matthews_corrcoef(y, pred),
    }, pred
