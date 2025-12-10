# backend/utils/test_embeddings.py
from dotenv import load_dotenv
load_dotenv()

from backend.utils.embeddings_provider import EmbeddingsProvider

import os

if __name__ == "__main__":
    print("Testing embeddings provider...")
    print("ENV provider:", os.getenv("EMBEDDINGS_PROVIDER", "not set"))
    try:
        ep = EmbeddingsProvider()  # defaults from env (we default to local)
        texts = [
            "Customer with high CLV and repeat purchases",
            "Customer with low engagement and high churn risk"
        ]
        vectors = ep.embed_texts(texts)
        print("✅ Number of vectors:", len(vectors))
        print("✅ Dimension of vector:", len(vectors[0]) if vectors else 0)
        print("👉 First 8 values of first vector:", vectors[0][:8] if vectors else None)
        qvec = ep.embed_query("Which users are at high churn risk?")
        print("✅ Query vector length:", len(qvec))
        print("\n🎉 Embeddings provider working!")
    except Exception as e:
        print("\n❌ Error while testing embeddings:")
        print(e)
