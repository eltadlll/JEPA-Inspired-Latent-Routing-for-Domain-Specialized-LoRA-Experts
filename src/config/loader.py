"""
Loads and validates pipeline configuration from YAML, with support for
`${ENV_VAR}` / `${ENV_VAR:default}` interpolation so secrets and
environment-specific paths never need to be hardcoded in the YAML files.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from pydantic import ValidationError

from src.config.schema import PipelineConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-?[^}]*)?\}")


class ConfigError(Exception):
    """Raised when configuration cannot be loaded or fails validation."""


def _interpolate_env_vars(raw_text: str) -> str:
    """Replace ${VAR} and ${VAR:default} occurrences with environment values."""

    def _replace(match: "re.Match[str]") -> str:
        var_name = match.group(1)
        default_clause = match.group(2)
        if var_name in os.environ:
            return os.environ[var_name]
        if default_clause is not None:
            # default_clause looks like ":-somevalue" or ":somevalue"
            return default_clause.lstrip(":-")
        raise ConfigError(
            f"Environment variable '{var_name}' is referenced in the config "
            f"but is not set and no default was provided."
        )

    return _ENV_VAR_PATTERN.sub(_replace, raw_text)


def _interpolate_value(value: Any) -> Any:
    """Recursively walk parsed YAML data and interpolate env vars in string
    leaves only. Doing this *after* parsing (rather than on raw text) means
    `${...}`-looking text inside YAML comments is never touched.
    """
    if isinstance(value, str):
        return _interpolate_env_vars(value)
    if isinstance(value, dict):
        return {k: _interpolate_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate_value(v) for v in value]
    return value


def _load_yaml_with_env(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw_text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML config at {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML content at {path} must be a mapping.")
    return _interpolate_value(data)


class ConfigLoader:
    """Loads a `PipelineConfig` from one or more YAML files.

    Supports layered configuration: a base config plus an optional
    override file (e.g. for local development or CI), merged shallowly
    at the top-level section.
    """

    def __init__(self, config_path: Path, override_path: Path | None = None) -> None:
        self.config_path = Path(config_path)
        self.override_path = Path(override_path) if override_path else None

    def load(self) -> PipelineConfig:
        merged = _load_yaml_with_env(self.config_path)

        if self.override_path is not None and self.override_path.exists():
            override_data = _load_yaml_with_env(self.override_path)
            merged = self._deep_merge(merged, override_data)

        try:
            config = PipelineConfig(**merged)
        except ValidationError as exc:
            raise ConfigError(f"Configuration validation failed:\n{exc}") from exc

        config.directories.ensure_exist()
        return config

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        result = dict(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


def load_config(config_path: str | Path, override_path: str | Path | None = None) -> PipelineConfig:
    """Convenience function used throughout the codebase and tests."""
    return ConfigLoader(Path(config_path), Path(override_path) if override_path else None).load()
