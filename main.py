from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import os
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from app.bm25_search import bm25_search

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "app", "data")

@app.get("/pdf/{pdf_name}")
async def get_pdf(pdf_name: str):
    pdf_path = os.path.join(DATA_FOLDER, pdf_name)
    print(f"Trying to serve PDF: {pdf_path}")
    if not os.path.isfile(pdf_path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(pdf_path, media_type="application/pdf")

@app.get("/keyword_search/")
async def keyword_search(query: str, top_k: int = 5):
    return bm25_search(query, top_k)

if not os.path.exists("uploads"):
    os.makedirs("uploads")

if not os.path.exists("app/static/images"):
    os.makedirs("app/static/images")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
