import os
from dataclasses import dataclass, field
from typing import Optional

NUM_TESTS = 1
NUM_RUNS = 3
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"

@dataclass
class Config:
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_api_base: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
    bedrock_region: str = field(default_factory=lambda: os.getenv("AWS_DEFAULT_REGION", "us-west-2"))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai").lower())
    model_name: str = field(default_factory=lambda: os.getenv("MODEL_NAME", DEFAULT_MODEL))
    num_tests: int = field(default_factory=lambda: int(os.getenv("NUM_TESTS", NUM_TESTS)))
    num_runs: int = field(default_factory=lambda: int(os.getenv("NUM_RUNS", NUM_RUNS)))
    nki_addr: Optional[str] = field(default_factory=lambda: os.getenv("NKI_ADDR"))
    triton_addr: Optional[str] = field(default_factory=lambda: os.getenv("TRITON_ADDR"))

def get_config() -> Config:
    return Config()
