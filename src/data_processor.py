import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from src.logger import logger
from src.config import settings


def _categorize_icd9(code: str) -> str:
    """Map ICD-9 diagnosis code to a high-level disease category."""
    try:
        code = str(code).replace("E", "").replace("V", "")
        code_num = float(code)
        if 390 <= code_num <= 459 or code_num == 785: return "Circulatory"
        elif 460 <= code_num <= 519 or code_num == 786: return "Respiratory"
        elif 520 <= code_num <= 579 or code_num == 787: return "Digestive"
        elif code_num == 250: return "Diabetes"
        elif 800 <= code_num <= 999: return "Injury"
        elif 710 <= code_num <= 739: return "Musculoskeletal"
        elif 580 <= code_num <= 629: return "Genitourinary"
        elif 140 <= code_num <= 239: return "Neoplasms"
        else: return "Other"
    except Exception:
        return "Other"

class FeatureDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop=None):
        self.cols_to_drop = cols_to_drop or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X.drop(columns=[c for c in self.cols_to_drop if c in X.columns])

class MedicationChangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, med_cols=None):
        self.med_cols = med_cols or []
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        cols = [c for c in self.med_cols if c in X.columns]
        if cols:
            X["num_meds_changed"] = X[cols].apply(
                lambda row: (row == "Up").sum() + (row == "Down").sum(), axis=1
            )
        return X.drop(columns=cols)

class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        if "age" in X.columns:
            age_map = {"[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
                       "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
                       "[80-90)": 85, "[90-100)": 95}
            X["age"] = X["age"].map(age_map).astype(float)
        
        diag_cols = ["diag_1", "diag_2", "diag_3"]
        for col in diag_cols:
            if col in X.columns:
                X[col] = X[col].apply(_categorize_icd9)
                
        return X

def build_data_pipeline():
    """Builds a scikit-learn Pipeline for robust, leak-proof feature engineering."""
    drop_cols = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty", "examide", "citoglipton"]
    med_cols = ["metformin", "repaglinide", "nateglinide", "chlorpropamide",
                "glimepiride", "acetohexamide", "glipizide", "glyburide",
                "tolbutamide", "pioglitazone", "rosiglitazone", "acarbose",
                "miglitol", "troglitazone", "tolazamide", "insulin",
                "glyburide-metformin", "glipizide-metformin",
                "glimepiride-pioglitazone", "metformin-rosiglitazone",
                "metformin-pioglitazone"]
                
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, make_column_selector(dtype_include=['int64', 'float64'])),
            ("cat", categorical_transformer, make_column_selector(dtype_include=['object', 'category']))
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    ).set_output(transform="pandas")  # Keeps it as a DataFrame for SHAP later

    pipeline = Pipeline(steps=[
        ("dropper", FeatureDropper(cols_to_drop=drop_cols)),
        ("med_change", MedicationChangeTransformer(med_cols=med_cols)),
        ("clinical_eng", ClinicalFeatureEngineer()),
        ("preprocessor", preprocessor)
    ])
    return pipeline

def load_and_split_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Loads raw data, performs row-level cleaning, and returns a train/test split."""
    logger.info(f"Loading dataset from {settings.RAW_DATA_PATH}...")
    df = pd.read_csv(settings.RAW_DATA_PATH, na_values=["?"], low_memory=False)
    
    # Downcast memory
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype('float32')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype('int32')

    # Target creation
    df["churn"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])
    
    # Drop rows with excessive missing values
    df = df.dropna(thresh=len(df.columns) * 0.6)
    
    X = df.drop(columns=["churn"])
    y = df["churn"]
    
    logger.info(f"Initial raw dataset split completed. Shape: X={X.shape}, y={y.shape}")
    return train_test_split(X, y, test_size=settings.TEST_SIZE, random_state=settings.RANDOM_STATE, stratify=y)
