# backend/vector_store.py
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from backend.utils.embeddings_provider import EmbeddingsProvider

CSV_PATH_DEFAULT = "backend/data/enhanced_user_dataset_cleaned.csv"
VECTORSTORE_DIR = "backend/store"


def load_dataframe(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def row_to_document(row_dict: dict) -> str:
    lines = [f"User Record - {row_dict.get('user_id', 'unknown')}"]
    for k, v in row_dict.items():
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def dataframe_to_documents(df):
    docs = []
    for _, row in df.iterrows():
        text = row_to_document(row.to_dict())
        docs.append(Document(page_content=text, metadata={"user_id": row.get("user_id")}))
    return docs


def create_vectorstore(provider="local", csv_path=CSV_PATH_DEFAULT, chunk_size=800, chunk_overlap=120):
    print("📌 Loading CSV...")
    df = load_dataframe(csv_path)

    if df.empty:
        raise ValueError("Loaded dataframe is empty.")

    print("📌 Converting rows to documents...")
    docs = dataframe_to_documents(df)

    print("📌 Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"📌 Total Chunks: {len(chunks)}")

    print(f"📌 Initializing embeddings provider (provider={provider})...")
    emb = EmbeddingsProvider(provider=provider)

    class EmbAdapter:
        def embed_documents(self, texts):
            return emb.embed_texts(texts)
        def embed_query(self, text):
            return emb.embed_query(text)

    print("📌 Creating FAISS index (this may take a while)...")
    vectorstore = FAISS.from_documents(chunks, EmbAdapter())

    print("📌 Saving FAISS index to", VECTORSTORE_DIR)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    print("🎉 Vectorstore created successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default=os.getenv("EMBEDDINGS_PROVIDER", "local"))
    parser.add_argument("--csv", default=CSV_PATH_DEFAULT)
    parser.add_argument("--chunk_size", type=int, default=800)
    parser.add_argument("--chunk_overlap", type=int, default=120)
    args = parser.parse_args()

    try:
        create_vectorstore(provider=args.provider, csv_path=args.csv, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    except Exception as e:
        print("❌ Failed to build vectorstore:", e)
        raise
