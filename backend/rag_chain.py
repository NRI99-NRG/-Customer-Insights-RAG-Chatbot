# backend/rag_chain.py
import os
import re
from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Any, Tuple, List, Optional

# Vectorstore and embeddings adapter
from langchain_community.vectorstores import FAISS
from backend.utils.embeddings_provider import EmbeddingsProvider
from backend.utils.langchain_embeddings_adapter import LangchainEmbeddingsAdapter

# Ollama client (local LLM)
from ollama import Client, ResponseError

SYSTEM_PROMPT = (
    "You are a helpful analytics assistant. Use ONLY the context provided "
    "to answer the user's question. If the answer is not in the context, say 'I don't know.'"
)

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "backend/store")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")


def load_vectorstore(path: str = VECTORSTORE_DIR) -> FAISS:
    """
    Load FAISS index from disk. We wrap the raw EmbeddingsProvider with the
    LangchainEmbeddingsAdapter so FAISS has an embeddings object to use for
    query-time embeddings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorstore not found at '{path}' — create it first.")

    # create raw provider (keeps reading EMBEDDINGS_PROVIDER env for compatibility)
    raw_provider = EmbeddingsProvider(provider=os.getenv("EMBEDDINGS_PROVIDER", "local"))
    lc_embeddings = LangchainEmbeddingsAdapter(raw_provider)

    # FAISS.load_local expects an embeddings object (and may require allow_dangerous_deserialization)
    return FAISS.load_local(
        path,
        embeddings=lc_embeddings,
        allow_dangerous_deserialization=True,
    )


def build_retrieval_qa_chain(k: int = 6) -> Tuple[Optional[object], object]:
    """
    Return a tuple (chain, retriever). We return None for chain because
    the actual LLM invocation is performed in run_query() (Ollama).
    """
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return None, retriever


def _build_context_from_docs(docs: List[object], max_chars: int = 20000) -> str:
    """
    Joins retrieved documents into a single context string.
    Trims to the first max_chars characters if too long.
    """
    parts = []
    for i, d in enumerate(docs, start=1):
        meta = getattr(d, "metadata", {}) or {}
        header = f"[DOC {i} - metadata: {meta}]"
        content = getattr(d, "page_content", getattr(d, "content", "")) or ""
        parts.append(header + "\n" + content)
    joined = "\n\n".join(parts)
    return joined if len(joined) <= max_chars else joined[:max_chars]


# numeric extraction helpers (used for deterministic ranking questions)
_NUMERIC_FIELDS = [
    "Total_Spending", "Customer_Lifetime_Value", "TotalSpending", "CLV", "Total_Spend",
    "Total_Spending_USD", "Total Spent", "Total Spending"
]
_NUMERIC_RE = re.compile(r"([A-Za-z_ ]+)\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _extract_preferred_numeric(doc_text: str):
    """
    Try to extract a numeric value using a priority list of field names.
    Returns (field_name, float_value) or (None, None)
    """
    if not doc_text:
        return None, None

    # prefer explicit keyed fields first using direct regex search
    for fname in _NUMERIC_FIELDS:
        m = re.search(rf"{re.escape(fname)}\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)", doc_text, flags=re.IGNORECASE)
        if m:
            try:
                return fname, float(m.group(1))
            except Exception:
                continue

    # fallback: attempt to find any label:number pair and return the largest numeric we find
    matches = re.findall(_NUMERIC_RE, doc_text)
    numeric_candidates = []
    for label, num in matches:
        try:
            numeric_candidates.append((label.strip(), float(num)))
        except Exception:
            pass

    if numeric_candidates:
        # choose the largest numeric candidate (heuristic)
        label, val = max(numeric_candidates, key=lambda t: t[1])
        return label, val

    return None, None


def _is_ranking_question(question: str) -> bool:
    q = (question or "").lower()
    triggers = [
        "which users", "top users", "most revenue", "contribute the most",
        "highest revenue", "highest spending", "top spenders", "top contributors",
        "who contributes", "which customers contribute", "top customers", "top clv",
    ]
    return any(t in q for t in triggers)


def run_query(chain, question: str, retriever, k: int = 6) -> Dict[str, Any]:
    """
    Fetch docs, optionally answer deterministic ranking questions using
    numeric extraction, otherwise call Ollama with context + question.
    Returns {"answer": str, "sources": [ {metadata, page_content}, ... ]}.
    """
    # adjust retriever k if supported
    try:
        if isinstance(k, int) and k > 0:
            retriever.search_kwargs["k"] = k
    except Exception:
        pass

    # 1) Fetch docs robustly (support multiple retriever APIs)
    docs = []
    try:
        # Preferred: LangChain retriever API
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        # Older versions / LCEL: retriever.invoke(...)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)
        # Fallback: underlying vectorstore similarity_search
        else:
            vs = getattr(retriever, "vectorstore", None)
            if vs is not None:
                docs = vs.similarity_search(question, k=k)
            else:
                raise RuntimeError("Retriever object has no known fetch method.")
    except Exception as e:
        # try fallback on underlying vectorstore if present
        try:
            vs = getattr(retriever, "vectorstore", None)
            if vs is not None:
                docs = vs.similarity_search(question, k=k)
            else:
                raise RuntimeError(f"Failed to fetch documents with retriever: {e}") from e
        except Exception as e2:
            raise RuntimeError(f"Failed to fetch documents with retriever: {e2}") from e2

    # Normalize sources for returning to UI
    sources = [
        {
            "metadata": getattr(d, "metadata", {}) or {},
            "page_content": getattr(d, "page_content", getattr(d, "content", "")) or ""
        }
        for d in docs
    ]

    # 2) If ranking question, compute deterministic top-k from numeric fields
    if _is_ranking_question(question):
        ranked = []
        for d in docs:
            txt = getattr(d, "page_content", getattr(d, "content", "")) or ""
            field, value = _extract_preferred_numeric(txt)

            meta = getattr(d, "metadata", {}) or {}
            user_id = meta.get("user_id") or meta.get("User_ID") or None

            # try to find "User_ID: #123" or "User ID: 123" in content if missing
            if not user_id:
                m = re.search(r"User[_ ]?ID\s*[:\-]\s*#?(\d+)", txt, flags=re.IGNORECASE)
                if m:
                    user_id = f"#{m.group(1)}"

            ranked.append({"doc": d, "user_id": user_id or "unknown", "field": field, "value": value})

        numeric_only = [r for r in ranked if r["value"] is not None]
        if not numeric_only:
            answer_text = "No numeric spending/CLV values found in retrieved documents to rank users."
            return {"answer": answer_text, "sources": sources}

        numeric_only.sort(key=lambda x: x["value"], reverse=True)
        topk = numeric_only[:k]

        lines = ["Top users by detected numeric field from retrieved docs:"]
        for i, t in enumerate(topk, start=1):
            # format value with comma separators if possible
            try:
                vstr = f"{t['value']:,}"
            except Exception:
                vstr = str(t["value"])
            lines.append(f"{i}. User: {t['user_id']} — {t['field'] or 'value'} = {vstr}")

        answer_text = "\n".join(lines)

        top_sources = []
        for t in topk:
            d = t["doc"]
            top_sources.append({
                "metadata": getattr(d, "metadata", {}) or {},
                "page_content": getattr(d, "page_content", getattr(d, "content", "")) or ""
            })

        return {"answer": answer_text, "sources": top_sources}

    # 3) Otherwise, call Ollama with constructed context
    context = _build_context_from_docs(docs)
    if not context.strip():
        return {"answer": "No relevant documents found.", "sources": sources}

    system_message = {"role": "system", "content": SYSTEM_PROMPT}
    user_content = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    user_message = {"role": "user", "content": user_content}

    client = Client()
    try:
        response = client.chat(model=OLLAMA_MODEL, messages=[system_message, user_message])
    except ResponseError as e:
        err_msg = getattr(e, "error", str(e))
        raise RuntimeError(f"Ollama responded with an error: {err_msg}") from e
    except Exception as e:
        raise RuntimeError("Failed to call local Ollama server. Is `ollama server` running and reachable?") from e

    try:
        answer = response.message.content
    except Exception:
        answer = str(response)

    return {"answer": answer, "sources": sources}
