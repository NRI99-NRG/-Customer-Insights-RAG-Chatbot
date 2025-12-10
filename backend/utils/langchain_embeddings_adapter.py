# backend/utils/langchain_embeddings_adapter.py
from typing import List, Sequence, Any

class LangchainEmbeddingsAdapter:
    """
    Adapter that wraps your existing EmbeddingsProvider instance and
    exposes the methods Faiss/LangChain expects:
      - embed_documents(list[str]) -> list[list[float]]
      - embed_query(str) -> list[float]
    """

    def __init__(self, provider):
        # provider: instance of your EmbeddingsProvider
        self.provider = provider

    def embed_documents(self, documents: Sequence[str]) -> List[List[float]]:
        """
        Called by vectorstores when embedding multiple documents.
        Adapt to whichever method your provider exposes.
        """
        # Try common method names in order of likelihood
        if hasattr(self.provider, "embed_documents"):
            return self.provider.embed_documents(list(documents))
        if hasattr(self.provider, "embed_batch"):
            return self.provider.embed_batch(list(documents))
        if hasattr(self.provider, "embed"):
            # provider.embed(text) -> vector
            return [self.provider.embed(d) for d in documents]
        # last fallback: try calling the provider if it implements a __call__
        if callable(self.provider):
            return [self.provider(d) for d in documents]
        raise RuntimeError(
            "Wrapped embeddings provider does not expose embed_documents/embed_batch/embed or __call__"
        )

    def embed_query(self, query: str) -> List[float]:
        """
        Called by vectorstores when embedding a query.
        """
        if hasattr(self.provider, "embed_query"):
            return self.provider.embed_query(query)
        if hasattr(self.provider, "embed"):
            return self.provider.embed(query)
        if callable(self.provider):
            return self.provider(query)
        raise RuntimeError(
            "Wrapped embeddings provider does not expose embed_query/embed or __call__"
        )

    # Some versions expect the object to be callable for single queries:
    def __call__(self, text: str) -> Any:
        return self.embed_query(text)
