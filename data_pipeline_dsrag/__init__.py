"""dsRAG-based pipeline — builds a KnowledgeBase using dsRAG's semantic
sectioning + AutoContext + RSE, with our existing Bedrock Titan v2
embeddings and a local FlashRank reranker. Parallel to `data_pipeline/`
(our custom parser) so we can A/B the two retrieval approaches without
disturbing the existing pipeline."""
