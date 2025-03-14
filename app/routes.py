from fastapi import APIRouter, UploadFile, File
from app.pdf_ingestion import ingest_pdfs
from app.query_handler import answer_question

router = APIRouter()

@router.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    pdf_path = f"uploads/{file.filename}"
    print("PDF Path - ", pdf_path)
    with open(pdf_path, "wb") as buffer:
        buffer.write(await file.read())

    response = ingest_pdfs(pdf_path)
    return response

@router.post("/query")
async def query(payload: dict):
    question = payload.get("question")
    return answer_question(question)
