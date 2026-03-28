from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Paths (Relative to project root)
    GRAPH_PATH: str = "data/graph/skingraph_v1.gpickle"
    MODEL_PATH: str = "models/artifacts/lgbm_risk_v1.pkl"
    INGREDIENTS_CSV: str = "data/processed/ingredients_master.csv"
    
    # API Keys (Loaded from .env automatically if present)
    GROQ_API_KEY: str = ""
    ENTREZ_EMAIL: str = "dummy@example.com"
    PUBCHEM_API_KEY: str = ""
    
    # Scoring Thresholds
    CONFLICT_HIGH_THRESHOLD: float = 0.75
    CONFLICT_MEDIUM_THRESHOLD: float = 0.45
    CONFIDENCE_HIGH_THRESHOLD: float = 0.75
    
    # Model Ensemble Weights
    WEIGHT_SKIN_ADJUSTED: float = 0.5
    WEIGHT_ML_SCORE: float = 0.3
    WEIGHT_BASE_SCORE: float = 0.2
    
    # Rate Limit Sleep Intervals (Seconds)
    PUBMED_SLEEP: float = 0.34
    PUBCHEM_SLEEP: float = 0.2
    SCRAPE_SLEEP_MIN: float = 2.0
    SCRAPE_SLEEP_MAX: float = 4.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Instantiate singleton
settings = Settings()
