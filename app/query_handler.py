import os
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import ollama
import urllib.parse
from app.bm25_search import bm25_search

DB_FOLDER = os.path.join(os.path.dirname(__file__), "vdb")
BACKEND_URL = "http://localhost:8000"
STATIC_IMAGE_DIR  = os.path.join(os.path.dirname(__file__), "static/images/")

model_path = os.path.join(os.path.dirname(__file__), "models")
embedding_model = HuggingFaceEmbeddings(model_name=model_path)
vector_db = Chroma(persist_directory=DB_FOLDER, embedding_function=embedding_model)

conversation_history = {}

def generate_response(question, context, history):
    """Generates a response using an LLM model based on retrieved context and conversation history."""
    context_texts = "\n\n".join([c["text"] for c in context if "text" in c])
    sources = {c["source"] for c in context}

    images = set()
    for c in context:
        image_paths = c.get("image_path", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        images.update(image_paths)

    images = list(images)

    citation_text = "Sources: " + ", ".join(sources) if sources else "No sources available."

    history_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in history[-3:]]) # last 3 messages.

    prompt = f"""
    You are OrchestrAI, an AI assistant helping users with a technical document.
    Answer ONLY using the provided context below.

    **Conversation History:**
    {history_text}

    **Retrieved Context:**
    {context_texts}

    **Related Images:**
    {', '.join(images) if images else 'No relevant images found.'}

    **User Question:**
    {question}

    **Response Guidelines:**
    - If the user greets (e.g., "hi", "hello", "hey"), respond warmly without referring to the document.
        - Example: "Hello! How can I assist you today?"
    - If the user says "thanks" or "thank you," acknowledge it politely.
        - Example: "You're welcome! Let me know if you need any further assistance."
    - If the question requires document context, answer based on the retrieved context.
    - If no relevant information is found, say:
        - "I couldn't find relevant information in the document."
    - **Only include citations if the response is based on document context.**

    Your response should be structured, bit elaborate and not one or two lines, clear, and factual.

    {citation_text if context_texts else ""}
    """

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])

    response_text = response.get("message", {}).get("content", "Error in response")
    if isinstance(response_text, list):
        response_text = " ".join(response_text)

    return {"answer": response_text + f"\n\n {citation_text}", "images": images}

from sentence_transformers import CrossEncoder
import numpy as np

local_model_path = os.path.join(os.path.dirname(__file__), "models/ms-marco-MiniLM-L-6-v2")

reranker = CrossEncoder(local_model_path)

def rerank_results(question: str, results: list[dict]) -> list[dict]:
    """Takes question and search results, rank results based on relevants and returns sorted results."""
    if not results:
        return []

    query_doc_pairs = [(question, res["text"]) for res in results]
    scores = reranker.predict(query_doc_pairs)

    scores = scores.tolist() if isinstance(scores, np.ndarray) else scores

    print("\nScores Before Sorting:")
    for res, score in zip(results, scores):
        print(f"Score: {score:.4f} | Text: {res['text']}")

    ranked_results = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)

    print("\nScores After Sorting:")
    for res, score in ranked_results:
        print(f"Score: {score:.4f} | Text: {res['text']}")

    return [res for res, _ in ranked_results]

def answer_question(question, user_id="default_user"):
    """Retrieves relevant information using hybrid search (BM25 + ChromaDB) and applies reranking."""
    
    history = conversation_history.get(user_id, [])
    
    bm25_results = bm25_search(question, top_k=5)
    vector_results = vector_db.similarity_search(question, k=5)

    combined_results = []
    seen_texts = set()

    for res in bm25_results:
        text = res["text"]
        if text not in seen_texts:
            seen_texts.add(text)
            combined_results.append(res)

    for doc in vector_results:
        text = doc.page_content
        if text not in seen_texts:
            seen_texts.add(text)
            combined_results.append({
                "text": text,
                "source": doc.metadata.get("source", "Unknown PDF"),
                "page": doc.metadata.get("page", "N/A"),
                "image_path": doc.metadata.get("image_path", [])
            })

    if not combined_results:
        return {
            "text": "I couldn't find any relevant information in the document.",
            "images": [],
            "source_info": []
        }

    ranked_results = rerank_results(question, combined_results)

    top_k = 3
    best_context = ranked_results[:top_k]

    all_images = set()
    source_info = []
    context_data = []

    for res in best_context:
        pdf_name = res["source"]
        page_number = res["page"]

        pdf_filename = urllib.parse.quote(pdf_name)
        pdf_url = f"{BACKEND_URL}/pdf/{pdf_filename}"

        image_paths = res.get("image_path", [])
        if isinstance(image_paths, str):
            image_paths = [image_paths]

        cleaned_images = []
        for img in image_paths:
            if not img.startswith("http"):
                img_filename = os.path.basename(img)
                img_filename = urllib.parse.quote(img_filename)
                full_url = f"{BACKEND_URL}/static/images/{img_filename}"
            else:
                full_url = img
            cleaned_images.append(full_url)

        all_images.update(cleaned_images)

        context_data.append({
            "text": res["text"],
            "source": pdf_name,
            "page": page_number,
            "image_path": cleaned_images
        })

        source_info.append({
            "pdf": pdf_name,
            "page": page_number,
            "link": pdf_url
        })

    response_data = generate_response(question, context_data, history)

    history.append({'role': 'user', 'content': question})
    history.append({'role': 'ai', 'content': response_data.get("answer", "Error in response")})
    conversation_history[user_id] = history

    return {
        "text": response_data.get("answer", "Error in response"),
        "images": list(all_images),
        "source_info": source_info
    }

if __name__ == "__main__":
    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = answer_question(user_question)
        print("\nAI Response:", answer["text"])
        print("Related Images:", answer["images"])