FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim
WORKDIR /app

ENV UV_SYSTEM_PYTHON=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_NO_PROGRESS=1 \
    PYTHONUNBUFFERED=1 \
    DOCKER_CONTAINER=1

# Install rag_app_new dependencies
COPY rag_app_new/pyproject.toml pyproject.toml
RUN uv pip install -r pyproject.toml

# Install CPU-only PyTorch first (avoids pulling ~2.4GB of NVIDIA CUDA libraries)
RUN uv pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install rag_app runtime dependencies only (no docling, deepeval, opencv, etc.)
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN uv pip install -r /tmp/requirements-runtime.txt && rm /tmp/requirements-runtime.txt

# Install OpenTelemetry for AWS observability
RUN uv pip install aws-opentelemetry-distro>=0.12.0

# OpenTelemetry configuration
ENV AGENT_OBSERVABILITY_ENABLED=true \
    OTEL_PYTHON_DISTRO=aws_distro \
    OTEL_PYTHON_CONFIGURATOR=aws_configurator \
    OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf \
    OTEL_TRACES_EXPORTER=otlp \
    OTEL_SERVICE_NAME=rag-agent

# Copy only the rag_app Python source files needed at runtime
COPY rag_app/__init__.py rag_app/__init__.py
COPY rag_app/retrieval_engine.py rag_app/retrieval_engine.py
COPY rag_app/indexing_engine.py rag_app/indexing_engine.py
COPY rag_app/model_loader.py rag_app/model_loader.py
COPY rag_app/answer_cache.py rag_app/answer_cache.py
COPY rag_app/embeddings.py rag_app/embeddings.py

# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore

# Copy pre-built vector store data and give ownership to non-root user
COPY --chown=bedrock_agentcore:bedrock_agentcore rag_app/data/vector_store/ rag_app/data/vector_store/

USER bedrock_agentcore

EXPOSE 8080

# Copy rag_app_new source
COPY rag_app_new/src/ src/
COPY rag_app_new/agentcore.py agentcore.py

CMD ["opentelemetry-instrument", "python", "-m", "src.main"]
