"""Container entry point for the RAG agent."""

import os
from src.config import settings

print(f"[startup] EMBEDDING_PROVIDER env={os.environ.get('EMBEDDING_PROVIDER')!r} settings={settings.EMBEDDING_PROVIDER!r}", flush=True)

# Force rag_app to see the resolved provider value before any rag_app import
os.environ["EMBEDDING_PROVIDER"] = settings.EMBEDDING_PROVIDER

from src.infrastructure.api import app

if __name__ == "__main__":
    app.run()
