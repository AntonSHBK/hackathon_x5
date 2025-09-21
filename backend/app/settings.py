import os
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field

from app.utils.logging import setup_logging

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Глобальные настройки приложения."""
    
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu", 
        description="Устройство для выполнения моделей: 'cpu' или 'cuda'"
    )
    # Директории
    DATA_DIR: Path = Field(default=BASE_DIR / "data")
    AUDIO_DIR: Path = Field(default=BASE_DIR / "data" / "audio")
    IMAGE_DIR: Path = Field(default=BASE_DIR / "data" / "images")
    CACHE_DIR: Path = Field(default=BASE_DIR / "data" / "cache_dir")
    LOG_DIR: Path = Field(default=BASE_DIR / "logs")

    # API ключи
    OPENAI_API_KEY: str = Field(
        ..., description="API ключ OpenAI"
    )    
    YANDEX_API_KEY: str = Field(
        ...,
        description="API ключ Яндекс Облака"
    )
    YANDEX_FOLDER_ID: str = Field(
        ...,
        description="ID папки в Яндекс Облаке для Yandex Foundation Models"
    )

    # Логирование
    LOG_LEVEL: str = Field(default="INFO")
    
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    @field_validator(
        "CACHE_DIR", 
        "LOG_DIR", 
        "DATA_DIR", 
        "AUDIO_DIR",
        "IMAGE_DIR",
        mode="before"
    )
    @classmethod
    def validate_paths(cls, value: str | Path) -> Path:
        """Автоматически создаёт директории при инициализации."""
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()

setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)