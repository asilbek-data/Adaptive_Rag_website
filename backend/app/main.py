import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .graph import rag_app
from .ingest import run_ingest

app = FastAPI(title="Adaptive Agentic RAG API", version="1.0.0")

# Allow the deployed frontend (Vercel) and local dev to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten to your frontend's origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatIn(BaseModel):
    question: str


class ChatOut(BaseModel):
    answer: str
    steps: list[str]
    sources: list[dict]
    web_used: bool


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    if not body.question or not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    result = rag_app.invoke({"question": body.question})

    return ChatOut(
        answer=result.get("generation", "(no answer)"),
        steps=result.get("steps", []),
        sources=result.get("sources", []),
        web_used=result.get("web_used", False),
    )


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".pdf", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail="Only .pdf, .txt, .md are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        stats = run_ingest(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {"filename": file.filename, **stats}
