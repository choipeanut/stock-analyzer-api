"""설정 모듈"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

# .env 로드
load_dotenv()

_BASE_DIR = Path(__file__).parent.parent


def _load_yaml(filename: str) -> dict:
    path = _BASE_DIR / "config" / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    # API 키
    alpha_vantage_api_key: str = Field(default="", env="ALPHA_VANTAGE_API_KEY")
    news_api_key: str = Field(default="", env="NEWS_API_KEY")
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    dart_api_key: str = Field(default="", env="DART_API_KEY")
    fred_api_key: str = Field(default="", env="FRED_API_KEY")
    bok_api_key: str = Field(default="", env="BOK_API_KEY")
    redis_url: str | None = Field(default=None, env="REDIS_URL")
    llm_provider: str = Field(default="anthropic", env="LLM_PROVIDER")
    llm_model: str = Field(default="claude-opus-4-6", env="LLM_MODEL")

    model_config = {"env_file": str(_BASE_DIR / ".env"), "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def app_settings(self) -> dict:
        return _load_yaml("settings.yaml")

    @property
    def weights(self) -> dict:
        return _load_yaml("weights.yaml")

    @property
    def base_dir(self) -> Path:
        return _BASE_DIR


@lru_cache
def get_settings() -> Settings:
    return Settings()
