from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ORCHESTRATOR_MODEL_ID: str = Field(
        default="us.anthropic.claude-sonnet-4-6",
        description="Bedrock model ID for the main agent.",
    )
    ROUTER_MODEL_ID: str = Field(
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        description="Bedrock model ID for intent classification.",
    )

    AWS_REGION: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock and AgentCore.",
    )

    MEMORY_ID: str = Field(
        default="",
        description="AgentCore Memory ID (from CDK stack output).",
    )

    EMBEDDING_PROVIDER: str = Field(
        default="local",
        description="Embedding provider: 'local' (BGE) or 'bedrock' (Titan).",
    )

    TEST_OUTPUT_DIR: Path = Field(
        default=_REPO_ROOT / "test_output",
        description="Directory containing xbrl_facts.csv, extracted_facts.csv, and xbrl_metadata.json.",
    )


settings = Settings()
