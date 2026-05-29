import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Project Paths
    RAW_DATA_PATH: str = os.path.join("data", "diabetic_data.csv")
    MODEL_DIR: str = "models"
    MODEL_NAME: str = "xgb_churn_model.pkl"
    
    @property
    def MODEL_PATH(self) -> str:
        return os.path.join(self.MODEL_DIR, self.MODEL_NAME)

    # Model Hyperparameters (Defaults for tuning range)
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    N_TRIALS: int = 15
    
    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()
