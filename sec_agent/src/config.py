from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ORCHESTRATOR_PROVIDER: Literal["bedrock", "deepseek"] = Field(
        default="deepseek",
        description=(
            "Which LLM provider serves the orchestrator (the ReAct agent). "
            "Router and judge stay on Bedrock Haiku regardless."
        ),
    )
    ORCHESTRATOR_MODEL_ID: str = Field(
        default="us.anthropic.claude-sonnet-4-6",
        description="Bedrock model ID for the main agent (used when provider=bedrock).",
    )
    ROUTER_MODEL_ID: str = Field(
        default="us.anthropic.claude-haiku-4-5-20251001-v1:0",
        description="Bedrock model ID for intent classification.",
    )

    AWS_REGION: str = Field(
        default="us-east-1",
        description="AWS region for Bedrock and AgentCore.",
    )

    DEEPSEEK_API_KEY: str = Field(
        default="",
        description="API key for DeepSeek (OpenAI-compatible endpoint).",
    )
    DEEPSEEK_MODEL_ID: str = Field(
        default="deepseek-v4-flash",
        description="DeepSeek model ID (used when provider=deepseek).",
    )
    DEEPSEEK_BASE_URL: str = Field(
        default="https://api.deepseek.com/v1",
        description="DeepSeek OpenAI-compatible API base URL.",
    )

    MEMORY_ID: str = Field(
        default="",
        description="AgentCore Memory ID (from CDK stack output).",
    )

    COMPRESS_TOOL_OUTPUTS: bool = Field(
        default=False,
        description=(
            "When true, tool outputs are passed through a Haiku filter "
            "that keeps only content relevant to the current user question "
            "before being returned to the orchestrator LLM."
        ),
    )


settings = Settings()
