import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Project Paths
    PROJECT_NAME: str = "Noise-Resilient Multimodal RAG"
    DATA_DIR: str = "data"
    MODELS_DIR: str = "models"
    LOGS_DIR: str = "logs"
    
    # Dataset
    KAGGLE_DATASET: str = "paramaggarwal/fashion-product-images-small"
    DATASET_VERSION: str = "v1.0.0"
    
    # Vector Store
    VECTOR_STORE_PATH: str = "vector_store/chroma_db"
    
    # Models
    TEXT_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    IMAGE_EMBEDDING_MODEL: str = "openai/clip-vit-base-patch32"
    LLM_MODEL: str = "gpt-3.5-turbo" # Default, can be changed
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Security
    API_KEY: str = os.getenv("API_KEY", "default_secret_key")

    class Config:
        env_file = ".env"

settings = Settings()
