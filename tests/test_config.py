import os
import pytest
from triton_nki_transpile.config import get_config

def test_config_loads_api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key-config")
    config = get_config()
    assert config.openai_api_key == "sk-test-key-config"

def test_config_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = get_config()
    assert config.openai_api_key is None

def test_config_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    config = get_config()
    assert config.llm_provider == "openai"
    assert config.model_name == "gpt-5-mini-2025-08-07"
    assert config.bedrock_region == "us-west-2"

def test_config_overrides(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "bedrock")
    monkeypatch.setenv("MODEL_NAME", "claude-3")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    config = get_config()
    assert config.llm_provider == "bedrock"
    assert config.model_name == "claude-3"
    assert config.bedrock_region == "us-east-1"
