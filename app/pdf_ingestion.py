# import fitz  # PyMuPDF
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma
# from langchain.embeddings import HuggingFaceEmbeddings
# import os
# from langchain.vectorstores.utils import filter_complex_metadata
# from langchain.schema import Document

# model_path = os.path.join(os.path.dirname(__file__), "models")
# embedding_function = HuggingFaceEmbeddings(model_name=model_path)

# def extract_text_and_images(pdf_path):
#     """Extracts text and images from the given PDF file."""
#     doc = fitz.open(pdf_path)
#     texts = []
#     documents = []
#     image_data = {}

#     for page_num, page in enumerate(doc):
#         page_text = page.get_text()
#         texts.append(page_text)

#         image_list = page.get_images(full=True)
#         image_refs = []

#         for img in image_list:
#             xref = img[0]
#             base_image = doc.extract_image(xref)
#             image_bytes = base_image["image"]
#             image_ext = base_image["ext"]
            
#             image_filename = f"static/images/page_{page_num+1}_{xref}.{image_ext}"
#             with open(image_filename, "wb") as img_file:
#                 img_file.write(image_bytes)
            
#             image_refs.append(image_filename)

#         documents.append({
#             "text": page_text,
#             "metadata": {"images": image_refs, "page": page_num+1}
#         })

#     return documents

# def clean_metadata(metadata):
#     """ Convert list values to comma-separated strings. """
#     if not isinstance(metadata, dict):
#         return {}

#     cleaned_metadata = {}
#     for key, value in metadata.items():
#         if isinstance(value, list):
#             cleaned_metadata[key] = ", ".join(map(str, value))
#         elif isinstance(value, (str, int, float, bool)):
#             cleaned_metadata[key] = value
#         else:
#             cleaned_metadata[key] = str(value)

#     return cleaned_metadata

# def ingest_to_chroma(extracted_docs):
#     docs_to_store = []

#     for doc in extracted_docs:
#         if not isinstance(doc, dict):
#             print(f"Skipping unexpected type: {type(doc)}")
#             continue

#         metadata = doc.get("metadata", {})
#         cleaned_metadata = clean_metadata(metadata)

#         temp_doc = Document(page_content=doc["text"], metadata=cleaned_metadata)
#         docs_to_store.append(temp_doc)

#     vector_store = Chroma.from_documents(
#         docs_to_store,
#         HuggingFaceEmbeddings(model_name=model_path),
#         persist_directory="./chroma_db"
#     )

#     return vector_store

# if __name__ == "__main__":
#     pdf_path = "uploads/developer_guide.pdf"

#     extracted_docs = extract_text_and_images(pdf_path)

#     vector_db = ingest_to_chroma(extracted_docs)

#     sample_docs = vector_db.similarity_search("Sonarqube", k=5)
#     for doc in sample_docs:
#         print(doc.page_content, doc.metadata)

import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain.schema import Document

model_path = os.path.join(os.path.dirname(__file__), "models")
embedding_function = HuggingFaceEmbeddings(model_name=model_path)

def extract_text_and_images(pdf_path):
    """Extracts text and images from the given PDF file."""
    doc = fitz.open(pdf_path)
    documents = []

    for page_num, page in enumerate(doc):
        page_text = page.get_text()
        image_list = page.get_images(full=True)
        image_refs = []

        for img in image_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            
            image_filename = f"static/images/page_{page_num+1}_{xref}.{image_ext}"
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)
            
            image_refs.append(image_filename)

        documents.append({
            "text": page_text,
            "metadata": {"images": image_refs if image_refs else "None", "page": page_num+1}
        })

    return documents

def clean_metadata(metadata):
    """Convert list values to comma-separated strings or 'None'."""
    if not isinstance(metadata, dict):
        return {}

    cleaned_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            cleaned_metadata[key] = ", ".join(map(str, value)) if value else "None"
        elif isinstance(value, (str, int, float, bool)):
            cleaned_metadata[key] = value
        else:
            cleaned_metadata[key] = str(value)

    return cleaned_metadata

def ingest_to_chroma(extracted_docs):
    docs_to_store = []

    for doc in extracted_docs:
        if not isinstance(doc, dict):
            print(f"Skipping unexpected type: {type(doc)}")
            continue

        metadata = doc.get("metadata", {})
        cleaned_metadata = clean_metadata(metadata)

        temp_doc = Document(page_content=doc["text"], metadata=cleaned_metadata)
        docs_to_store.append(temp_doc)

    vector_store = Chroma.from_documents(
        docs_to_store,
        HuggingFaceEmbeddings(model_name=model_path),
        persist_directory="./chroma_db"
    )

    return vector_store

if __name__ == "__main__":
    pdf_path = "uploads/developer_guide.pdf"

    extracted_docs = extract_text_and_images(pdf_path)

    vector_db = ingest_to_chroma(extracted_docs)

    sample_docs = vector_db.similarity_search("Sonarqube", k=5)
    for doc in sample_docs:
        print(doc.page_content, doc.metadata)
