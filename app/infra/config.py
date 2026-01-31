from dotenv import load_dotenv
from pydantic import Field

load_dotenv()
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    app_name: str = Field(default="Simple Agent", alias="APP_NAME")
    app_version: str = Field(default="0.1.0", alias="APP_VERSION")
    open_ai_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENAI_BASE_URL")
    open_ai_api_key: str = Field(alias="OPENAI_API_KEY")

def get_settings() -> Settings:
    return Settings()