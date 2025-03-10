import os
import json
import pickle
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(BASE_DIR, "data")
DB_FOLDER = os.path.join(BASE_DIR, "vdb")
IMAGE_FOLDER = os.path.join(BASE_DIR, "static/images")
BM25_CORPUS_PATH = os.path.join(DATA_FOLDER, "bm25_corpus.json")
BM25_INDEX_PATH = os.path.join(DATA_FOLDER, "bm25_index.pkl")

os.makedirs(IMAGE_FOLDER, exist_ok=True)
os.makedirs(DATA_FOLDER, exist_ok=True)
os.makedirs(DB_FOLDER, exist_ok=True)

model_path = os.path.join(BASE_DIR, "models")
embedding_model = HuggingFaceEmbeddings(model_name=model_path)

def preprocess_text(text):
    """Tokenize text for BM25."""
    return text.lower().split()

def load_bm25_corpus():
    """Load BM25 corpus from JSON."""
    if os.path.exists(BM25_CORPUS_PATH) and os.path.getsize(BM25_CORPUS_PATH) > 0:
        with open(BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("BM25 corpus is corrupted. Resetting.")
                return []
    return []

def save_bm25_corpus(corpus):
    """Save BM25 corpus to JSON."""
    with open(BM25_CORPUS_PATH, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=4, ensure_ascii=False)

def index_bm25(corpus):
    """Creates a BM25 search Index and save."""
    tokenized_corpus = [preprocess_text(doc["text"]) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Indexed {len(corpus)} documents in BM25.")

def extract_images_from_pdf(pdf_path, pdf_name):
    """Converts PDF pages to images."""
    image_paths = []
    try:
        images = convert_from_path(pdf_path, dpi=300)
        for idx, img in enumerate(images):
            image_filename = f"{pdf_name}_page_{idx+1}.png"
            image_path = os.path.join(IMAGE_FOLDER, image_filename)
            img.save(image_path, "PNG")
            image_paths.append(image_path.replace("\\", "/"))
    except Exception as e:
        print(f"Error processing PDF images: {e}")
    return image_paths

def ingest_pdfs():
    """Extracts text, stores in vector DB, and indexes BM25."""
    documents = []
    bm25_corpus = load_bm25_corpus()

    for file_name in os.listdir(DATA_FOLDER):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(DATA_FOLDER, file_name)
            print(f"Processing: {pdf_path}")
            images = extract_images_from_pdf(pdf_path, file_name)
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            for idx, page in enumerate(pages):
                page.metadata["source"] = file_name
                page.metadata["page"] = idx + 1
                if idx < len(images):
                    page.metadata["image_path"] = images[idx]
            documents.extend(pages)

    if not documents:
        print("No documents found for ingestion.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    new_entries = [
        {
            "text": chunk.page_content,
            "source": chunk.metadata["source"],
            "page": chunk.metadata["page"],
            "image_path": chunk.metadata.get("image_path", "")
        }
        for chunk in chunks
    ]
    bm25_corpus.extend(new_entries)
    save_bm25_corpus(bm25_corpus)
    index_bm25(bm25_corpus)
    
    vector_db = Chroma.from_documents(chunks, embedding_model, persist_directory=DB_FOLDER)
    vector_db.persist()
    print(f"Indexed {len(chunks)} chunks from {len(os.listdir(DATA_FOLDER))} PDFs.")

def search_bm25(query, top_n=5):
    """Searches BM25 corpus for top relevant results."""
    if not os.path.exists(BM25_INDEX_PATH):
        print("BM25 index not found. Run ingestion first.")
        return []
    
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25 = pickle.load(f)
    
    tokenized_query = preprocess_text(query)
    scores = bm25.get_scores(tokenized_query)
    top_indexes = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
    
    with open(BM25_CORPUS_PATH, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    
    results = [corpus[i] for i in top_indexes]
    return results

if __name__ == "__main__":
    ingest_pdfs()
    test_query = "How to onboard users to Symphony?"
    print(f"BM25 Search Results for '{test_query}':")
    results = search_bm25(test_query)
    for idx, result in enumerate(results):
        print(f"{idx+1}. {result['text'][:200]}...")
