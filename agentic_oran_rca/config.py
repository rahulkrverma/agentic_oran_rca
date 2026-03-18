from __future__ import annotations

import os
from dataclasses import dataclass


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not str(value).strip():
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str

    chroma_host: str
    chroma_port: int

    ollama_base_url: str
    ollama_model: str
    ollama_embed_model: str

    log_level: str


def load_settings() -> Settings:
    return Settings(
        neo4j_uri=_require_env("NEO4J_URI"),
        neo4j_user=_require_env("NEO4J_USER"),
        neo4j_password=_require_env("NEO4J_PASSWORD"),
        chroma_host=_require_env("CHROMA_HOST"),
        chroma_port=int(_require_env("CHROMA_PORT")),
        ollama_base_url=_require_env("OLLAMA_BASE_URL"),
        ollama_model=_require_env("OLLAMA_MODEL"),
        ollama_embed_model=_require_env("OLLAMA_EMBED_MODEL"),
        log_level=_require_env("LOG_LEVEL"),
    )

