# frontend/streamlit_app.py
# Complete Streamlit front-end for your local RAG pipeline (improved error handling).

import os
import sys
import traceback
from typing import Dict, Any

# Path to project root: Langchain/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load environment variables early
from dotenv import load_dotenv
load_dotenv()

import streamlit as st

# Try import backend functions (show friendly message in UI if fails)
try:
    from backend.rag_chain import build_retrieval_qa_chain, run_query
except Exception as e:
    build_retrieval_qa_chain = None
    run_query = None
    _import_error = e
else:
    _import_error = None

st.set_page_config(page_title="RAG Analytics Chatbot", layout="wide")
st.title("RAG Analytics Chatbot")

st.markdown(
    "Ask questions about the dataset (CLV, churn, revenue, etc.). "
    "This app calls your local RAG pipeline directly (no FastAPI). "
    "Make sure `backend/store` exists and the vectorstore has been built."
)

# ---------------- Streamlit compatibility shim for caching ----------------
if hasattr(st, "cache_resource"):
    cache_singleton = st.cache_resource
elif hasattr(st, "experimental_singleton"):
    cache_singleton = st.experimental_singleton
else:
    def cache_singleton(func):
        return func
# -------------------------------------------------------------------------

# read some helpful env hints to show to user
LLM_MODEL = os.getenv("LLM_MODEL") or os.getenv("OLLAMA_MODEL") or os.getenv("OPENAI_MODEL") or "not-set"
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "not-set")

# Initialize chain & retriever once per session (cached)
@cache_singleton
def init_rag(k: int = 6) -> Dict[str, Any]:
    if _import_error is not None:
        return {"ok": False, "error": f"Failed to import backend.rag_chain: {_import_error}", "trace": ""}
    try:
        chain, retriever = build_retrieval_qa_chain(k=k)
        return {"ok": True, "chain": chain, "retriever": retriever}
    except Exception as e:
        return {"ok": False, "error": str(e), "trace": traceback.format_exc()}

init = init_rag()

if not init.get("ok"):
    st.error("Failed to initialize RAG chain at startup.")
    st.write(init.get("error"))
    if init.get("trace"):
        st.code(init.get("trace"))
    st.stop()

chain = init["chain"]
retriever = init["retriever"]

# ---------------- UI ----------------
col_top_left, col_top_right = st.columns([3, 1])
with col_top_right:
    st.markdown("### Info")
    st.write(f"- Embeddings provider: **{EMBEDDINGS_PROVIDER}**")
    st.write(f"- LLM model: **{LLM_MODEL}**")
    st.write("- `backend/store` must exist (vectorstore created).")
    st.write("- To avoid quota errors, use a local LLM (Ollama) or turn on 'Docs only' to inspect retrieved docs.")
    if _import_error:
        st.warning("Backend import failed. See error above.")

with col_top_left:
    question = st.text_area(
        "Question",
        value="Which users contribute the most revenue?",
        height=160,
        placeholder="Type a question about the dataset (CLV, churn, revenue, etc.)"
    )

k = st.number_input(
    "Retriever k (number of documents to fetch)",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

docs_only = st.checkbox("Show retrieved documents only (do not call LLM)", value=False)

col1, col2 = st.columns([3, 1])
with col1:
    if st.button("Send"):
        if not question or not question.strip():
            st.warning("Please type a question before sending.")
        else:
            with st.spinner("Running retrieval (and LLM if enabled)..."):
                try:
                    # If user only wants docs, we still call run_query but instruct UI to ignore answer
                    resp = run_query(chain, question, retriever, k=int(k))

                    # sources may be list of dicts with page_content and metadata
                    sources = resp.get("sources", []) or []
                    # If user asked "docs only", skip LLM answer display and show sources only
                    if docs_only:
                        st.success("Retrieved documents (docs-only mode).")
                        if not sources:
                            st.info("No source chunks returned by retriever.")
                        for i, s in enumerate(sources, start=1):
                            meta = s.get("metadata", {})
                            page_content = s.get("page_content", s.get("content", ""))
                            st.markdown(f"**Source {i} — metadata:** `{meta}`")
                            display_text = page_content if len(page_content) <= 3000 else page_content[:3000] + "\n\n...[trimmed]"
                            st.code(display_text)
                    else:
                        # Normal full flow: show answer then sources
                        answer = resp.get("answer") or resp.get("result") or "<no answer>"
                        if hasattr(answer, "content"):
                            answer_text = answer.content
                        else:
                            answer_text = str(answer)

                        st.subheader("Answer")
                        st.write(answer_text)

                        st.subheader(f"Top {len(sources)} retrieved source chunks")
                        if not sources:
                            st.info("No source chunks returned by retriever.")
                        for i, s in enumerate(sources, start=1):
                            meta = s.get("metadata", {})
                            page_content = s.get("page_content", s.get("content", ""))
                            st.markdown(f"**Source {i} — metadata:** `{meta}`")
                            display_text = page_content if len(page_content) <= 3000 else page_content[:3000] + "\n\n...[trimmed]"
                            st.code(display_text)

                except Exception as e:
                    # Friendly handling for common errors
                    err_text = str(e)
                    st.error(f"Error during query: {err_text}")

                    # Common problem: embeddings provider object not callable
                    if "object is not callable" in err_text and "EmbeddingsProvider" in err_text:
                        st.warning(
                            "It looks like your embeddings provider object is not callable. "
                            "Check backend/utils/embeddings_provider.py: it should expose a LangChain Embeddings object "
                            "or implement embed_query/embed_documents methods and be passed to FAISS.load_local correctly."
                        )

                    # Common problem: Google/Gemini quota (ResourceExhausted)
                    if "Quota exceeded" in err_text or "ResourceExhausted" in err_text or "quota" in err_text.lower():
                        st.warning(
                            "Quota or rate-limit error detected. Options:\n"
                            "- Use a local LLM (Ollama) and set OLLAMA_MODEL in .env.\n"
                            "- Use a different API key / check cloud provider billing.\n"
                            "- Use Docs-only mode to inspect retrieved chunks without calling the LLM."
                        )

                    st.code(traceback.format_exc())

with col2:
    st.markdown("### Help")
    st.write("- If you see *quota* errors, switch to a local model or check billing.")
    st.write("- If the retrieved docs look wrong, re-create vectorstore with the same embeddings provider.")
    st.write("- For production, expose an API (FastAPI) and add concurrency controls.")

st.markdown("---")
st.caption("Streamlit UI — quick local demo (no FastAPI).")
