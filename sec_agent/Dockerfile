FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1

# Install sec_agent dependencies
COPY sec_agent/pyproject.toml pyproject.toml
RUN uv pip install -r pyproject.toml

# Install OpenTelemetry for AWS observability
RUN uv pip install aws-opentelemetry-distro>=0.12.0

# OpenTelemetry configuration
ENV AGENT_OBSERVABILITY_ENABLED=true \
    OTEL_PYTHON_DISTRO=aws_distro \
    OTEL_PYTHON_CONFIGURATOR=aws_configurator \
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
    OTEL_TRACES_EXPORTER=otlp \
    OTEL_SERVICE_NAME=rag-agent

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 8080

# Copy sec_agent source
COPY sec_agent/src/ src/
COPY sec_agent/agentcore.py agentcore.py

# NOTE: For deployment the agent needs access to structured facts
# (data/extracted_facts_and_tables/) and the narrative Chroma store
# (data/narrative_store/). These are not copied here — wire them in via
# volume mount, EFS, or explicit COPY when finalizing the deploy path.

CMD ["opentelemetry-instrument", "python", "-m", "src.main"]
