"""
Multimodal ingest: PDFs (text + images, images captioned by a vision model)
and plain text/markdown files -> chunks -> Qdrant.
Ported from Agentic_RAG_ready.ipynb (cells 8-10).
"""
import base64
from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import get_vision_llm, vectorstore

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
MIN_IMAGE_BYTES = 3000  # skip tiny icons/decorations


def caption_image(image_bytes: bytes, vision_llm) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe this image/diagram in 1-3 sentences, focusing on any "
                    "text, labels, numbers, or technical content visible in it. "
                    "Be factual and specific."
                ),
            },
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
        ]
    )
    return vision_llm.invoke([msg]).content


def load_pdf(path: Path, vision_llm) -> List[Document]:
    docs: List[Document] = []
    pdf = fitz.open(path)

    for page_num, page in enumerate(pdf, start=1):
        text = page.get_text().strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={"source": path.name, "page": page_num, "type": "text"},
            ))

        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = pdf.extract_image(xref)
            image_bytes = base_image["image"]
            if len(image_bytes) < MIN_IMAGE_BYTES:
                continue
            try:
                caption = caption_image(image_bytes, vision_llm)
            except Exception as e:  # noqa: BLE001
                print(f"  [!] Image captioning failed ({path.name} p{page_num}): {e}")
                continue
            docs.append(Document(
                page_content=f"[Image] {caption}",
                metadata={
                    "source": path.name,
                    "page": page_num,
                    "type": "image",
                    "image_index": img_index,
                },
            ))

    pdf.close()
    return docs


def load_text_file(path: Path) -> List[Document]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [Document(page_content=text, metadata={"source": path.name, "type": "text"})]


def load_path(path: Path, vision_llm) -> List[Document]:
    docs: List[Document] = []
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    for f in files:
        if not f.is_file():
            continue
        if f.suffix.lower() == ".pdf":
            docs.extend(load_pdf(f, vision_llm))
        elif f.suffix.lower() in {".txt", ".md"}:
            docs.extend(load_text_file(f))
    return docs


def run_ingest(source_path: str) -> dict:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {source_path}")

    vision_llm = get_vision_llm()
    raw_docs = load_path(path, vision_llm)
    if not raw_docs:
        return {"documents": 0, "chunks": 0}

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(raw_docs)

    vectorstore.add_documents(chunks)
    return {"documents": len(raw_docs), "chunks": len(chunks)}
