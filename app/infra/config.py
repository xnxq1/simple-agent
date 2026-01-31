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

def get_settings() -> Settings:
    return Settings()


settings = get_settings()
