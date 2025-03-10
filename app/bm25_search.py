import json
from rank_bm25 import BM25Okapi
from fastapi import HTTPException
import os

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
bm25_data_path = os.path.join(DATA_FOLDER, "bm25_corpus.json")

try:
    if not os.path.exists(bm25_data_path):
        raise FileNotFoundError(f"BM25 corpus file not found at {bm25_data_path}")
        
    with open(bm25_data_path, "r", encoding="utf-8") as f:
        content = f.read()
        if not content.strip():
            raise ValueError("BM25 corpus file is empty")
        bm25_corpus = json.loads(content)

    if isinstance(bm25_corpus, list) and all(isinstance(doc, dict) and "text" in doc for doc in bm25_corpus):
        tokenized_corpus = [doc["text"].lower().split() for doc in bm25_corpus if doc["text"].strip()]
    else:
        raise ValueError("Invalid BM25 corpus format. Expected list of dicts with 'text' keys.")

    if not tokenized_corpus:
        raise ValueError("BM25 corpus is empty after processing.")

    bm25 = BM25Okapi(tokenized_corpus)

except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
    print(f"Error loading BM25 corpus: {str(e)}")
    bm25_corpus = []
    tokenized_corpus = []
    bm25 = None

def bm25_search(query: str, top_k: int = 5):
    """Searches BM25 index for relevant chunks."""
    if not bm25 or not bm25_corpus:
        raise HTTPException(status_code=503, detail="Search index is not available")

    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    results = [
        {
            "text": bm25_corpus[i]["text"],
            "source": bm25_corpus[i].get("source", "Unknown"),
            "page": bm25_corpus[i].get("page", "Unknown")
        }
        for i in top_indices
    ]

    if not results:
        raise HTTPException(status_code=404, detail="No relevant documents found.")
    
    return results