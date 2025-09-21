import os
from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field

from app.logging import setup_logging

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    
    DEVICE: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu", 
        description="Устройство для выполнения моделей: 'cpu' или 'cuda'"
    )
    # Директории
    DATA_DIR: Path = Field(default=BASE_DIR / "data")
    CACHE_DIR: Path = Field(default=BASE_DIR / "data" / "cache_dir")
    LOG_DIR: Path = Field(default=BASE_DIR / "logs")

    LOG_LEVEL: str = Field(default="INFO")
    
    @field_validator(
        "CACHE_DIR", 
        "LOG_DIR", 
        "DATA_DIR", 
        mode="before"
    )
    @classmethod
    def validate_paths(cls, value: str | Path) -> Path:
        path = Path(value)
        path.mkdir(parents=True, exist_ok=True)
        return path

settings = Settings()

setup_logging(log_dir=settings.LOG_DIR, log_level=settings.LOG_LEVEL)