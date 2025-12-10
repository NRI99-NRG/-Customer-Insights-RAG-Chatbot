# backend/utils/embeddings_provider.py
import os
from typing import List, Optional
from sentence_transformers import SentenceTransformer

# Read from .env (but we only support "local")
ENV_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()


class EmbeddingsProvider:
    """
    Local-only embeddings provider using SentenceTransformer.
    This keeps the interface identical to the original version,
    but removes all Gemini logic to keep it simple.
    """

    def __init__(self,
                 provider: Optional[str] = None,
                 local_model: str = "all-MiniLM-L6-v2"):
        # We keep the variable to avoid breaking other files
        self.provider = (provider or ENV_PROVIDER or "local").lower()

        if self.provider != "local":
            print(f"⚠️ EMBEDDINGS_PROVIDER={self.provider} not supported. Using local embeddings instead.")
            self.provider = "local"

        self.local_model_name = local_model
        self._local = None

    def _init_local(self):
        if self._local is None:
            self._local = SentenceTransformer(self.local_model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts. Returns list[list[float]].
        Local SBERT only.
        """
        self._init_local()
        arr = self._local.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in arr]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        """
        self._init_local()
        v = self._local.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        return v.tolist()
