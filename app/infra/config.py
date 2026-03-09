from dotenv import load_dotenv
from pydantic import Field, computed_field

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
    open_ai_base_url: str = Field(
        default="https://openrouter.ai/api/v1", alias="OPENAI_BASE_URL"
    )
    open_ai_api_key: str = Field(alias="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")

    qdrant_host: str = Field(default="qdrant", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    top_k_limit: int = Field(default=3, alias="TOP_K_LIMIT")


    postgres_db: str = Field(default="simple_agent", alias="POSTGRES_DB")
    postgres_user: str = Field(default="simple_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="simple_password", alias="POSTGRES_PASSWORD")
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")

    @computed_field
    @property
    def db_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


def get_settings() -> Settings:
    return Settings()
