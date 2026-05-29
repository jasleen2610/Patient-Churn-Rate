"""
Generates SHAP explanations for individual patient predictions
and global feature importance analysis.
"""
import shap
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.calibration import CalibratedClassifierCV


def _unwrap_classifier(classifier):
    """
    Return the underlying base estimator from a CalibratedClassifierCV wrapper,
    or the classifier itself if it is not wrapped.
    """
    if isinstance(classifier, CalibratedClassifierCV):
        return classifier.estimator
    return classifier


def get_shap_explainer(pipeline, X_train_sample: pd.DataFrame):
    """
    Create and return a SHAP TreeExplainer.
    Uses a sample of training data as background for speed.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    base_model = _unwrap_classifier(pipeline.named_steps["classifier"])
    
    X_bg_processed = preprocessor.transform(X_train_sample)
    background = shap.sample(X_bg_processed, 100, random_state=42)
    explainer = shap.TreeExplainer(base_model, background)
    return explainer


def get_patient_shap(
    explainer, pipeline, patient_row: pd.DataFrame
) -> Tuple[np.ndarray, float, pd.DataFrame]:
    """
    Calculate SHAP values for a single patient row.

    Returns:
        shap_values: array of per-feature contributions
        base_value: the model's baseline expected output
        row_processed: the pre-processed patient row to map features to
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    row_processed = preprocessor.transform(patient_row)
    
    shap_values = explainer.shap_values(row_processed)
    # For binary classification, shap_values is a list [neg, pos]
    # We want the positive class (churn = 1)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]
    
    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[1]
        
    return sv, float(base_value), row_processed


def get_top_risk_factors(
    shap_values: np.ndarray,
    feature_names: list,
    row_processed: pd.DataFrame,
    top_n: int = 5,
) -> list[dict]:
    """
    Return the top N features that INCREASED churn risk for this patient.

    Each item has: feature_name, shap_value, patient_value, direction.
    """
    factors = []
    # If row_processed is a pandas dataframe, it makes extraction easy.
    # The ColumnTransformer is configured to return pandas dataframes.
    for i, (name, sv) in enumerate(zip(feature_names, shap_values)):
        factors.append({
            "feature": name,
            "shap_value": float(sv),
            "patient_value": float(row_processed.iloc[0, i]),
        })

    # Sort by absolute impact, descending
    factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    top = factors[:top_n]

    for f in top:
        f["direction"] = "Increases Risk" if f["shap_value"] > 0 else "Reduces Risk"

    return top


def get_global_feature_importance(pipeline, feature_names: list) -> pd.DataFrame:
    """
    Return a DataFrame of global feature importances from the trained model.
    """
    base_model = _unwrap_classifier(pipeline.named_steps["classifier"])
    importances = base_model.feature_importances_
    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)
    return df
