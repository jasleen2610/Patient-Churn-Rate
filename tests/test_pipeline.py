import pytest
import pandas as pd
from src.data_processor import _categorize_icd9, FeatureDropper, MedicationChangeTransformer

def test_icd9_categorization():
    assert _categorize_icd9("428") == "Circulatory"
    assert _categorize_icd9("466") == "Respiratory"
    assert _categorize_icd9("530") == "Digestive"
    assert _categorize_icd9("250") == "Diabetes"
    assert _categorize_icd9("invalid") == "Other"

def test_feature_dropper():
    df = pd.DataFrame({"col1": [1], "col2": [2]})
    dropper = FeatureDropper(cols_to_drop=["col1"])
    transformed = dropper.transform(df)
    assert "col1" not in transformed.columns
    assert "col2" in transformed.columns

def test_medication_change_transformer():
    df = pd.DataFrame({
        "med1": ["Up", "No", "Down"],
        "med2": ["No", "Steady", "Up"]
    })
    transformer = MedicationChangeTransformer(med_cols=["med1", "med2"])
    transformed = transformer.transform(df)
    assert "num_meds_changed" in transformed.columns
    assert list(transformed["num_meds_changed"]) == [1, 0, 2]
    assert "med1" not in transformed.columns
    assert "med2" not in transformed.columns

if __name__ == "__main__":
    pytest.main([__file__])
