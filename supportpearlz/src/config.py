from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr

class Settings(BaseSettings):
    openai_api_key: SecretStr
    log_level: str = "INFO"
    chunk_size: int = 800
    chunk_overlap: int = 150
    vector_store_path: str = "data/vector_store"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()