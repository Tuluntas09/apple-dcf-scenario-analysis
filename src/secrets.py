from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"


def _parse_simple_toml_value(line: str) -> tuple[str, str] | None:
    if "=" not in line or line.strip().startswith("#"):
        return None
    key, value = line.split("=", 1)
    key = key.strip().lstrip("\ufeff")
    value = value.strip().strip('"').strip("'")
    if not key:
        return None
    return key, value


def read_local_secret(name: str) -> str | None:
    if not LOCAL_SECRETS_PATH.exists():
        return None
    for line in LOCAL_SECRETS_PATH.read_text(encoding="utf-8").splitlines():
        parsed = _parse_simple_toml_value(line)
        if parsed and parsed[0] == name:
            return parsed[1]
    return None


def get_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    return read_local_secret(name)
