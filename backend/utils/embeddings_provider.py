# backend/utils/embeddings_provider.py
import os
from typing import List, Optional

# Try to import Gemini (google genai) integration
_gemini_available = True
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    _gemini_available = False

# Try to import local SBERT
_local_available = True
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    _local_available = False

# Default provider: read from env, otherwise local
ENV_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()

# Allow GEMINI_API_KEY alias -> GOOGLE_API_KEY
if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")


class EmbeddingsProvider:
    """
    Embeddings provider with two backends:
      - 'local' : sentence-transformers (default)
      - 'gemini': Google Generative AI embeddings (requires GOOGLE_API_KEY & quota)
    """

    def __init__(self,
                 provider: Optional[str] = None,
                 gemini_model: str = "models/gemini-embedding-001",
                 local_model: str = "all-MiniLM-L6-v2"):
        self.provider = (provider or ENV_PROVIDER or "local").lower()
        self.gemini_model = gemini_model
        self.local_model_name = local_model
        self._gemini = None
        self._local = None

        if self.provider == "gemini" and not _gemini_available:
            if _local_available:
                print("⚠️ Gemini library not available — falling back to local embeddings.")
                self.provider = "local"
            else:
                raise ImportError("Gemini embeddings requested but langchain_google_genai not installed.")
        if self.provider == "local" and not _local_available:
            raise ImportError("Local embeddings requested but sentence-transformers is not installed. pip install -U sentence-transformers")

    # lazy inits
    def _init_gemini(self):
        if self._gemini is None:
            if not _gemini_available:
                raise RuntimeError("langchain_google_genai not installed.")
            if not os.getenv("GOOGLE_API_KEY"):
                raise EnvironmentError("GOOGLE_API_KEY (or GEMINI_API_KEY) required for Gemini embeddings.")
            self._gemini = GoogleGenerativeAIEmbeddings(model=self.gemini_model)

    def _init_local(self):
        if self._local is None:
            if not _local_available:
                raise RuntimeError("sentence-transformers not installed.")
            self._local = SentenceTransformer(self.local_model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts. Returns list[list[float]].
        """
        if self.provider == "gemini":
            try:
                self._init_gemini()
                return self._gemini.embed_documents(texts)
            except Exception as e:
                # fallback to local
                print("⚠️ Gemini embeddings failed — falling back to local. Error:", e)
                if _local_available:
                    self.provider = "local"
                    self._init_local()
                    arr = self._local.encode(texts, show_progress_bar=False, convert_to_numpy=True)
                    return [v.tolist() for v in arr]
                raise

        # local branch
        self._init_local()
        arr = self._local.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return [v.tolist() for v in arr]

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query string.
        """
        if self.provider == "gemini":
            try:
                self._init_gemini()
                return self._gemini.embed_query(text)
            except Exception as e:
                print("⚠️ Gemini query embed failed — falling back to local. Error:", e)
                if _local_available:
                    self.provider = "local"
                    self._init_local()
                    v = self._local.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
                    return v.tolist()
                raise

        self._init_local()
        v = self._local.encode([text], show_progress_bar=False, convert_to_numpy=True)[0]
        return v.tolist()
