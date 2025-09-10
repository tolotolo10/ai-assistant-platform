"""
Configuration management for AI Assistant Platform.
"""
import json
from typing import Optional, List

from dotenv import load_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    # App
    app_name: str = "AI Assistant Platform"
    app_version: str = "0.1.0"
    debug: bool = False

    # Phase 2: RAG Settings
    vector_db: str = "pinecone"  
    pinecone_index: str = "ai-assistant-index"  
    chunk_size: int = 800  
    chunk_overlap: int = 120  

    # API
    api_v1_prefix: str = "/api/v1"
    allowed_hosts: List[str] = ["*"]

    @field_validator("allowed_hosts", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v):
        if v is None:
            return ["*"]
        if isinstance(v, list):
            cleaned = [str(h).strip() for h in v if str(h).strip()]
            return cleaned or ["*"]
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return ["*"]
            if s.startswith("[") and s.endswith("]"):
                try:
                    arr = json.loads(s)
                    if isinstance(arr, list):
                        cleaned = [str(h).strip() for h in arr if str(h).strip()]
                        return cleaned or ["*"]
                except json.JSONDecodeError:
                    pass
            cleaned = [h.strip() for h in s.split(",") if h.strip()]
            return cleaned or ["*"]
        return ["*"]

    # Security
    secret_key: str = "my-secret-key"
    access_token_expire_minutes: int = 30

    # AI keys
    openai_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    #pinecone_environment: str = "us-east1-gcp"


    pinecone_cloud: str = "aws"        # env: PINECONE_CLOUD
    pinecone_region: str = "us-east-1" # env: PINECONE_REGION  (pick a serverless region your plan supports)

    


    # Data stores
    database_url: str = "sqlite:///./ai_assistant.db"
    redis_url: str = "redis://localhost:6379"

    # Model config
    default_model: str = "gpt-4o"
    max_tokens: int = 2000
    temperature: float = 0.0
    memory_window_k: int = 10  # remember last 10 messages

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # ignore unexpected env vars instead of erroring
    )


# Global settings instance
settings = Settings()
