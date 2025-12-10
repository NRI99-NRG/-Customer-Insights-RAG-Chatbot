# backend/rag_chain.py
import os
import re
from dotenv import load_dotenv
load_dotenv()

from typing import Dict, Any, Tuple, List, Optional
from langchain_community.vectorstores import FAISS
from backend.utils.embeddings_provider import EmbeddingsProvider
from backend.utils.langchain_embeddings_adapter import LangchainEmbeddingsAdapter
from ollama import Client, ResponseError

VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "backend/store")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

SYSTEM_PROMPT = (
    "You are a helpful analytics assistant. Use ONLY the context provided to answer the user's question. "
    "If the answer is not in the context, say 'I don't know.'"
)

_NUMERIC_FIELDS = ["Total_Spending", "Customer_Lifetime_Value", "CLV", "TotalSpending", "Total_Spend"]
_NUMERIC_RE = re.compile(r"([A-Za-z_ ]+)\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_RANK_TRIGGERS = [
    "which users", "top users", "most revenue", "contribute the most",
    "highest revenue", "highest spending", "top spenders", "top contributors",
    "who contributes", "which customers contribute", "top customers", "top clv"
]


def load_vectorstore(path: str = VECTORSTORE_DIR) -> FAISS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorstore not found at '{path}'. Run ingestion first.")
    provider = EmbeddingsProvider(provider=os.getenv("EMBEDDINGS_PROVIDER", "local"))
    lc_emb = LangchainEmbeddingsAdapter(provider)
    return FAISS.load_local(path, embeddings=lc_emb, allow_dangerous_deserialization=True)


def build_retrieval_qa_chain(k: int = 6) -> Tuple[Optional[object], object]:
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return None, retriever


def _join_docs(docs: List[object], max_chars: int = 20000) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        content = getattr(d, "page_content", getattr(d, "content", "")) or ""
        parts.append(f"[DOC {i} - metadata: {meta}]\n{content}")
    joined = "\n\n".join(parts)
    return joined if len(joined) <= max_chars else joined[:max_chars]


def _extract_numeric(txt: str):
    if not txt:
        return None, None
    for f in _NUMERIC_FIELDS:
        m = re.search(rf"{re.escape(f)}\s*[:\-]\s*([0-9]+(?:\.[0-9]+)?)", txt, flags=re.IGNORECASE)
        if m:
            try:
                return f, float(m.group(1))
            except Exception:
                pass
    matches = re.findall(_NUMERIC_RE, txt)
    candidates = []
    for label, num in matches:
        try:
            candidates.append((label.strip(), float(num)))
        except Exception:
            pass
    return max(candidates, key=lambda t: t[1]) if candidates else (None, None)


def _is_ranking_question(q: str) -> bool:
    if not q:
        return False
    q = q.lower()
    return any(t in q for t in _RANK_TRIGGERS)


def run_query(chain, question: str, retriever, k: int = 6) -> Dict[str, Any]:
    # set k if possible
    try:
        if isinstance(k, int) and k > 0:
            retriever.search_kwargs["k"] = k
    except Exception:
        pass

    # fetch docs (robust)
    docs = []
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(question)
        elif hasattr(retriever, "invoke"):
            docs = retriever.invoke(question)
        else:
            vs = getattr(retriever, "vectorstore", None)
            if vs is not None:
                docs = vs.similarity_search(question, k=k)
            else:
                raise RuntimeError("Retriever has no known fetch method.")
    except Exception:
        # fallback: try vectorstore similarity_search
        vs = getattr(retriever, "vectorstore", None)
        if vs is not None:
            docs = vs.similarity_search(question, k=k)
        else:
            raise

    sources = [
        {"metadata": getattr(d, "metadata", {}) or {}, "page_content": getattr(d, "page_content", getattr(d, "content", "")) or ""}
        for d in docs
    ]

    # deterministic ranking for numeric questions
    if _is_ranking_question(question):
        ranked = []
        for d in docs:
            txt = getattr(d, "page_content", getattr(d, "content", "")) or ""
            field, value = _extract_numeric(txt)
            meta = getattr(d, "metadata", {}) or {}
            user_id = meta.get("user_id") or meta.get("User_ID") or None
            if not user_id:
                m = re.search(r"User[_ ]?ID\s*[:\-]\s*#?(\d+)", txt, flags=re.IGNORECASE)
                if m:
                    user_id = f"#{m.group(1)}"
            ranked.append({"doc": d, "user_id": user_id or "unknown", "field": field, "value": value})
        numeric_only = [r for r in ranked if r["value"] is not None]
        if not numeric_only:
            return {"answer": "No numeric spending/CLV values found in retrieved documents to rank users.", "sources": sources}
        numeric_only.sort(key=lambda x: x["value"], reverse=True)
        topk = numeric_only[:k]
        lines = [f"Top users by detected numeric field from retrieved docs:"]
        for i, t in enumerate(topk, 1):
            try:
                vstr = f"{t['value']:,}"
            except Exception:
                vstr = str(t["value"])
            lines.append(f"{i}. User: {t['user_id']} — {t['field'] or 'value'} = {vstr}")
        top_sources = [{"metadata": getattr(t["doc"], "metadata", {}) or {}, "page_content": getattr(t["doc"], "page_content", getattr(t["doc"], "content", "")) or ""} for t in topk]
        return {"answer": "\n".join(lines), "sources": top_sources}

    # otherwise call Ollama with context
    context = _join_docs(docs)
    if not context.strip():
        return {"answer": "No relevant documents found.", "sources": sources}

    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    user_msg = {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"}
    client = Client()
    try:
        resp = client.chat(model=OLLAMA_MODEL, messages=[system_msg, user_msg])
    except ResponseError as e:
        raise RuntimeError(f"Ollama error: {getattr(e,'error',str(e))}") from e
    except Exception:
        raise RuntimeError("Failed to call local Ollama server. Is it running?")

    try:
        answer = resp.message.content
    except Exception:
        answer = str(resp)

    return {"answer": answer, "sources": sources}
