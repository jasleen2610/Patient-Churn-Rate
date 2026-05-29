import os
import pickle
import optuna
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from src.data_processor import load_and_split_data, build_data_pipeline
from src.logger import logger
from src.config import settings

optuna.logging.set_verbosity(optuna.logging.WARNING)


def train_model(save: bool = True) -> dict:
    """
    Load data, tune XGBoost via Optuna, calibrate probabilities, and save the full pipeline.
    """
    try:
        logger.info("Loading and splitting raw dataset...")
        X_train, X_test, y_train, y_test = load_and_split_data()

        logger.info("Building feature engineering pipeline...")
        preprocessing_pipeline = build_data_pipeline()
        
        logger.info("Pre-transforming data for hyperparameter tuning...")
        X_train_processed = preprocessing_pipeline.fit_transform(X_train)
        X_test_processed = preprocessing_pipeline.transform(X_test)
        
        feature_names = list(X_train_processed.columns)
        scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

        logger.info("Starting Optuna hyperparameter tuning (15 trials)...")
        
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.7, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
                "scale_pos_weight": scale_pos_weight,
                "eval_metric": "auc",
                "random_state": settings.RANDOM_STATE,
                "n_jobs": -1
            }
            model = XGBClassifier(**params)
            model.fit(X_train_processed, y_train, verbose=False)
            preds = model.predict_proba(X_test_processed)[:, 1]
            return roc_auc_score(y_test, preds)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=settings.N_TRIALS, show_progress_bar=False)
        
        logger.info(f"Best hyperparameters found: {study.best_params}")

        best_params = study.best_params
        best_params["scale_pos_weight"] = scale_pos_weight
        best_params["eval_metric"] = "auc"
        best_params["random_state"] = settings.RANDOM_STATE
        best_params["n_jobs"] = -1
        
        logger.info("Training final XGBoost model...")
        
        base_model = XGBClassifier(**best_params)
        base_model.fit(X_train_processed, y_train)

        full_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessing_pipeline),
            ("classifier", base_model)
        ])

        logger.info("Evaluating full pipeline on test set...")
        y_pred = full_pipeline.predict(X_test)
        y_prob = full_pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Model training completed. AUC-ROC: {auc:.4f}")

        results = {
            "model": full_pipeline,
            "feature_names": feature_names,
            "X_test": X_test,  
            "y_test": y_test,
            "y_prob": y_prob,
            "auc": auc,
            "report": report,
            "confusion_matrix": cm,
        }

        if save:
            os.makedirs(settings.MODEL_DIR, exist_ok=True)
            with open(settings.MODEL_PATH, "wb") as f:
                pickle.dump(results, f)
            logger.info(f"Model artifact persisted to {settings.MODEL_PATH}")

        return results
    except Exception as e:
        logger.exception("Error during model training pipeline.")
        raise
