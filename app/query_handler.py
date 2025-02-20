import ollama
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import os
from langchain.memory import ConversationBufferMemory

model_path = os.path.join(os.path.dirname(__file__), "models")
embedding_function = HuggingFaceEmbeddings(model_name=model_path)

vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

BACKEND_URL = "http://localhost:8000"

def re_rank_documents(query, documents):
    """Re-ranks documents based on keyword presence."""
    query_tokens = set(query.lower().split())
    ranked_docs = sorted(documents, key=lambda doc: sum(1 for word in query_tokens if word in doc.page_content.lower()), reverse=True)
    return ranked_docs


# def answer_question(question):
#     """Retrieves relevant context and returns an answer along with images."""
#     retrieved_context = vector_db.max_marginal_relevance_search(question, k=3, lambda_mult=0.7)
#     # retrieved_context = re_rank_documents(question, retrieved_context)

#     if not retrieved_context:
#         return {"text": "I couldn't find any relevant information in the document.", "images": []}

#     print("\n🔍 Retrieved Contexts:")
#     for i, doc in enumerate(retrieved_context):
#         print(f"Context {i+1}: {doc.page_content[:200]}...")
#         print(f"Images Metadata: {doc.metadata.get('images', 'No images found')}\n")  

#     context_texts = [doc.page_content for doc in retrieved_context]
#     image_paths = []

#     image_paths = []
#     for doc in retrieved_context:
#         images = doc.metadata.get("images", [])
#         if isinstance(images, str):  
#             images = [img.strip() for img in images.split(",") if img.strip()]
#         if isinstance(images, list):
#             image_paths.extend([f"{BACKEND_URL}/{img}" for img in images if img])

#     image_paths = list(set(image_paths))

#     print("\Images:", image_paths)

#     formatted_context = "\n\n".join(context_texts)[:1000]

#     response_text = generate_response(question, formatted_context)

#     return {"text": response_text, "images": image_paths}
# def answer_question(question):
#     retrieved_context = vector_db.similarity_search(question, k=3)

#     print("\n🔍 Retrieved Contexts:")
#     if not retrieved_context:
#         print("❌ No relevant chunks retrieved!")
#     else:
#         for i, doc in enumerate(retrieved_context):
#             print(f"📄 Context {i+1}: {doc.page_content[:300]}...")  
#             print(f"📌 Metadata: {doc.metadata}\n")
#             print("🖼️ Image Data:", doc.metadata.get("image_path", "No Image"))

#     if not retrieved_context:
#         return {"text": "I couldn't find any relevant information in the document.", "images": []}

#     context_texts = [doc.page_content for doc in retrieved_context]

#     response_text = generate_response(question, context_texts)

#     return {"text": response_text, "images": []}

memory = ConversationBufferMemory(return_messages=True)

BACKEND_URL = "http://localhost:8000"

def generate_response(question, context):
    if not isinstance(context, list):
        context = []

    context_texts = "\n\n".join([c["text"] for c in context if "text" in c])

    images = [
        img.strip() if img.startswith("http") else f"{BACKEND_URL}/{img.strip()}"
        for c in context if "images" in c
        for img in (c["images"] if isinstance(c["images"], list) else c["images"].split(", "))
        if img.strip() and img.lower() != "none"
    ]

    print("\nUsing Context for Response:")
    print(context_texts if context_texts else "No relevant context found!")
    print("Images Found:", images if images else "No images found!")

    prompt = f""" 
    You are OrchestrAI, an AI assistant helping users with a technical document.  
    Answer ONLY using the provided context below.

    **Retrieved Context:**  
    {context_texts}  

    **Related Images:**  
    {', '.join(images) if images else 'No relevant images found.'}

    **User Question:**  
    {question}  

    If the answer is not in the context, say: "I couldn't find relevant information in the document." 
    Your response should be concise, factual, and structured.
    """ 

    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])

    return {"answer": response['message']['content'], "images": images}


def answer_question(question):
    retrieved_context = vector_db.similarity_search(question, k=3)

    print("\nRetrieved Contexts:")
    if not retrieved_context:
        print("No relevant chunks retrieved!")
    else:
        for i, doc in enumerate(retrieved_context):
            print(f"Context {i+1}: {doc.page_content[:300]}...")  
            print(f"Metadata: {doc.metadata}\n")
            print("Image Data:", doc.metadata.get("images", "No Image"))

    if not retrieved_context:
        return {"text": "I couldn't find any relevant information in the document.", "images": []}

    context_data = []
    for doc in retrieved_context:
        images = doc.metadata.get("images", [])

        if isinstance(images, str):
            images = images.split(", ")

        full_image_urls = [
            img.strip() if img.startswith("http") else f"{BACKEND_URL}/{img.strip()}"
            for img in images
            if img.strip() and img.lower() != "none"
        ]

        context_data.append({
            "text": doc.page_content,
            "images": full_image_urls
        })

    response_data = generate_response(question, context_data)

    return {"text": response_data["answer"], "images": response_data["images"]}

# def generate_response(question, context):
#     if not isinstance(context, list):
#         context = []

#     for msg in context:
#         if isinstance(msg, dict) and "text" in msg:
#             memory.save_context({"input": msg["text"]}, {"output": msg.get("response", "")})

#     history = memory.load_memory_variables({})["history"]

#     prompt = f""" 
#     You are OrchestrAI, an AI assistant helping users with a technical document.  
#     Answer ONLY using the provided context. 

#     Conversation history:
#     {history}

#     User Question: 
#     {question} 

#     If the answer is not in the context, say: "I couldn't find relevant information." 
#     Be concise, factual, and structured in your answer. 
#     """ 

#     response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    
#     return response['message']['content']

# def generate_response(question, context):
#     if not isinstance(context, list):
#         context = []

#     # Ensure we're passing only text chunks as context
#     context_texts = "\n\n".join(context)

#     # DEBUG: Print context before prompting
#     print("\n🔍 Using Context for Response:")
#     print(context_texts if context_texts else "❌ No relevant context found!")

#     prompt = f""" 
#     You are OrchestrAI, an AI assistant helping users with a technical document.  
#     Answer ONLY using the provided context below.

#     📄 **Retrieved Context:**  
#     {context_texts}  

#     📝 **User Question:**  
#     {question}  

#     If the answer is not in the context, say: "I couldn't find relevant information in the document." 
#     Your response should be concise, factual, and structured.
#     """ 

#     response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    
#     return response['message']['content']

# def generate_response(question, context):
#     if not isinstance(context, list):
#         context = []

#     context_texts = "\n\n".join([c["text"] for c in context if "text" in c])
#     images = [c["images"] for c in context if "images" in c and c["images"]]

#     print("\n🔍 Using Context for Response:")
#     print(context_texts if context_texts else "❌ No relevant context found!")
#     print("🖼️ Images Found:", images if images else "❌ No images found!")

#     prompt = f""" 
#     You are OrchestrAI, an AI assistant helping users with a technical document.  
#     Answer ONLY using the provided context below.

#     📄 **Retrieved Context:**  
#     {context_texts}  

#     🖼️ **Related Images:**  
#     {', '.join(images) if images else 'No relevant images found.'}

#     📝 **User Question:**  
#     {question}  

#     If the answer is not in the context, say: "I couldn't find relevant information in the document." 
#     Your response should be concise, factual, and structured.
#     """ 

#     response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])

#     return {"answer": response['message']['content'], "images": images}


if __name__ == "__main__":
    while True:
        user_question = input("\nAsk a question: ")
        if user_question.lower() in ["exit", "quit"]:
            break
        answer = answer_question(user_question)
        print("\nAI Response:", answer["text"])
        print("Related Images:", answer["images"])



# def generate_response(question, context):
#     # Format conversation history into a structured context
#     history = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in context])

#     prompt = f""" 
#      You are OrchestrAI, a helpful assistant in answering questions related to Symphony 
#     based on a developer guide. You are having a conversation with a user. You'll be
#     given the context and the user query along with history of past few messages.
#     If the answer is not in the context, say: "I couldn't find relevant information."
#     Be concise and structured in your answer.

#     Conversation so far:
#     {history}

#     User Question: 
#     {question} 

#     If the answer is not in the context, say: "I couldn't find relevant information." 
#     Be concise, factual, and structured in your answer. 
#     """ 

#     response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    
#     return response['message']['content']

# def generate_response(question, context):
#     history = "\n".join([f"{msg['role'].capitalize()}: {msg['text']}" for msg in context])

#     # prompt = f"""
#     # You are an AI assistant helping users with a technical document. 
#     # Answer ONLY using the provided context.

#     # Context:
#     # {context}

#     # User Question:
#     # {question}

#     # If the answer is not in the context, say: "I couldn't find relevant information."
#     # Be concise, factual, and structured in your answer.
#     # """
#     prompt = f"""
#     You are OrchestrAI, a helpful assistant in answering questions related to Symphony 
#     based on a developer guide. You are having a conversation with a user. You'll be
#     given the context and the user query along with history of past few messages.
#     If the answer is not in the context, say: "I couldn't find relevant information."
#     Be concise and structured in your answer.
#     Here is the conversation so far:

#     {history}

#     Now, answer the latest user query:
#     User: {question}
#     Assistant:
#     """
#     response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    
#     return response['message']['content']

