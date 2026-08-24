import os
from pathlib import Path

import pytest
import yaml

from src.config.loader import ConfigError, load_config
from src.config.schema import PipelineConfig


MINIMAL_CONFIG = {
    "project_name": "test-project",
    "directories": {
        "root": "data",
        "raw": "data/raw",
        "intermediate": "data/intermediate",
        "clean": "data/clean",
        "processed": "data/processed",
        "instruction": "data/instruction",
        "reports": "data/reports",
        "logs": "data/logs",
    },
    "sources": {
        "github_repos": [
            {"url": "https://github.com/example/repo", "category": "agent_engineering"}
        ]
    },
}


def _write_config(tmp_path: Path, data: dict) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(data), encoding="utf-8")
    return config_path


def test_load_valid_config(tmp_path):
    config_path = _write_config(tmp_path, MINIMAL_CONFIG)
    os.chdir(tmp_path)
    config = load_config(config_path)
    assert isinstance(config, PipelineConfig)
    assert config.project_name == "test-project"
    assert len(config.sources.github_repos) == 1


def test_config_without_sources_fails(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["sources"] = {}
    config_path = _write_config(tmp_path, data)
    os.chdir(tmp_path)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_config_with_bad_github_url_fails(tmp_path):
    data = dict(MINIMAL_CONFIG)
    data["sources"] = {"github_repos": [{"url": "not-a-url"}]}
    config_path = _write_config(tmp_path, data)
    os.chdir(tmp_path)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_env_var_interpolation(tmp_path, monkeypatch):
    monkeypatch.setenv("TEST_DATASET_NAME", "my-custom-dataset")
    data = dict(MINIMAL_CONFIG)
    data["export"] = {"dataset_name": "${TEST_DATASET_NAME}"}
    config_path = _write_config(tmp_path, data)
    os.chdir(tmp_path)
    config = load_config(config_path)
    assert config.export.dataset_name == "my-custom-dataset"


def test_env_var_default_used_when_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("UNSET_VAR_FOR_TEST", raising=False)
    data = dict(MINIMAL_CONFIG)
    data["export"] = {"dataset_name": "${UNSET_VAR_FOR_TEST:-fallback-name}"}
    config_path = _write_config(tmp_path, data)
    os.chdir(tmp_path)
    config = load_config(config_path)
    assert config.export.dataset_name == "fallback-name"


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "does_not_exist.yaml")
