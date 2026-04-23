from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    COMPRESS_TOOL_OUTPUTS: bool = Field(
        default=False,
        description=(
            "When true, tool outputs are passed through a Haiku filter "
            "that keeps only content relevant to the current user question "
            "before being returned to the orchestrator LLM."
        ),
    )


settings = Settings()
