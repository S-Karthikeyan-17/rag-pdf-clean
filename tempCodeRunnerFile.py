from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Dict
from pathlib import Path
import uuid
import fitz  # PyMuPDF
import re
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings as ChromaSettings
from openai import OpenAI
import os

# ======================
# CONFIG
# ======================
class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    CHROMA_PERSIST_DIR: str = "./storage/chroma"
    UPLOAD_DIR: str = "./storage/uploads"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE_TOKENS: int = 350
    CHUNK_OVERLAP_TOKENS: int = 50
    TOP_K: int = 5
    class Config:
        env_file = ".env"

settings = Settings()
Path(settings.CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ======================
# MODELS
# ======================
class UploadResponse(BaseModel):
    doc_id: str
    pages: int
    chunks: int

class QueryRequest(BaseModel):
    doc_id: str = Field(..., description="Document ID returned by /documents")
    question: str
    top_k: int | None = None

class ContextChunk(BaseModel):
    page: int
    text: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    contexts: List[ContextChunk]

# ======================
# UTILS
# ======================
def new_id() -> str:
    return uuid.uuid4().hex

def ensure_pdf(file: UploadFile):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

def save_upload(file: UploadFile, doc_id: str) -> str:
    suffix = Path(file.filename).suffix or ".pdf"
    path = Path(settings.UPLOAD_DIR) / f"{doc_id}{suffix}"
    with open(path, "wb") as f:
        f.write(file.file.read())
    return str(path)

def approx_token_count(text: str) -> int:
    return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

def split_into_paras(text: str) -> List[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def window_chunks(paras: List[str], max_tokens: int, overlap: int) -> List[str]:
    chunks = []
    current = []
    while paras:
        p = paras.pop(0)
        cur = "\n\n".join(current + [p])
        if approx_token_count(cur) > max_tokens and current:
            chunks.append("\n\n".join(current))
            while current and approx_token_count("\n\n".join(current)) > overlap:
                current.pop(0)
        current.append(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks

def chunk_page_text(page_text: str, max_tokens: int, overlap: int) -> List[str]:
    paras = split_into_paras(page_text)
    if not paras:
        return []
    return window_chunks(paras, max_tokens, overlap)

# ======================
# PDF INGEST
# ======================
def extract_pdf(path: str) -> List[Dict]:
    out = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text") or ""
            out.append({"page": i, "text": text})
    return out

# ======================
# EMBEDDINGS
# ======================
_embedder = None
def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedder

def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_embedder().encode(texts, normalize_embeddings=True).tolist()

def embed_query(text: str) -> List[float]:
    return get_embedder().encode([text], normalize_embeddings=True)[0].tolist()

# ======================
# VECTOR DB
# ======================
client = chromadb.PersistentClient(
    path=settings.CHROMA_PERSIST_DIR,
    settings=ChromaSettings(anonymized_telemetry=False)
)
COLLECTION = "pdf_chunks"
collection = client.get_or_create_collection(
    name=COLLECTION, metadata={"hnsw:space": "cosine"}
)

def add_chunks(doc_id: str, chunks: List[Dict]):
    documents = [c["text"] for c in chunks]
    embeddings = embed_texts(documents)
    metadatas = [{"doc_id": doc_id, "page": c["page"]} for c in chunks]
    collection.add(
        ids=[c["id"] for c in chunks],
        metadatas=metadatas,
        documents=documents,
        embeddings=embeddings
    )

def query_db(doc_id: str, question: str, top_k: int) -> Dict:
    q_emb = embed_query(question)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k,
        where={"doc_id": doc_id},
        include=["documents", "distances", "metadatas"]
    )
    return res

def count_for_doc(doc_id: str) -> int:
    res = collection.get(where={"doc_id": doc_id}, include=[])
    return len(res["ids"])

# ======================
# RAG PIPELINE
# ======================
def build_prompt(question: str, contexts: List[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return (
        "You are a helpful assistant. Use ONLY the provided context to answer.\n"
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

def generate_answer(question: str, contexts: List[str]) -> str:
    if not contexts:
        return "I don't know. I couldn't find relevant context."
    if not settings.OPENAI_API_KEY:
        return f"{contexts[0]}\n\n(Generated from retrieved context; no LLM configured.)"
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
    client = OpenAI()
    prompt = build_prompt(question, contexts)
    completion = client.chat.completions.create(
        model=settings.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return completion.choices[0].message.content.strip()

def retrieve(doc_id: str, question: str, top_k: int) -> Dict:
    res = query_db(doc_id, question, top_k)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    contexts = []
    for text, meta, dist in zip(docs, metas, dists):
        contexts.append({
            "text": text,
            "page": meta.get("page", -1),
            "score": float(1 - dist) if dist is not None else 0.0
        })
    return {"contexts": contexts}

# ======================
# FASTAPI APP
# ======================
app = FastAPI(
    title="PDF RAG Chatbot",
    version="1.0.0",
    description="Upload a PDF, index it, and chat with it."
)

@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ensure_pdf(file)
    doc_id = new_id()
    path = save_upload(file, doc_id)

    pages = extract_pdf(path)
    if not pages:
        raise HTTPException(status_code=400, detail="No extractable text found.")

    chunks = []
    for p in pages:
        parts = chunk_page_text(p["text"], settings.CHUNK_SIZE_TOKENS, settings.CHUNK_OVERLAP_TOKENS)
        for j, text in enumerate(parts):
            chunks.append({"id": f"{doc_id}_{p['page']}_{j}", "page": p["page"], "text": text})

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks generated from PDF.")

    add_chunks(doc_id, chunks)
    return UploadResponse(doc_id=doc_id, pages=len(pages), chunks=len(chunks))

@app.post("/query", response_model=QueryResponse)
async def query_pdf(req: QueryRequest):
    top_k = req.top_k or settings.TOP_K
    if count_for_doc(req.doc_id) == 0:
        raise HTTPException(status_code=404, detail="Document not found or not indexed.")
    result = retrieve(req.doc_id, req.question, top_k)
    contexts = result["contexts"]
    answer = generate_answer(req.question, [c["text"] for c in contexts])
    return QueryResponse(
        answer=answer,
        contexts=[ContextChunk(page=c["page"], text=c["text"], score=c["score"]) for c in contexts]
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

# Mount static frontend folder
app.mount("/", StaticFiles(directory="static", html=True), name="static")
