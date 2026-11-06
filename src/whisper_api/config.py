"""Server settings, read from WHISPER_* environment variables or a .env file."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="WHISPER_",
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
    )

    host: str = "127.0.0.1"
    port: int = 8000
    api_key: str = ""

    model: str = "large-v3"
    model_dir: str | None = None
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str = "auto"
    cpu_threads: int = Field(default=0, ge=0)
    beam_size: int = Field(default=5, ge=1)
    hotwords: str = ""

    workers: int = Field(default=1, ge=1)
    max_queue: int = Field(default=16, ge=0)
    max_upload_mb: int = Field(default=500, ge=1)
