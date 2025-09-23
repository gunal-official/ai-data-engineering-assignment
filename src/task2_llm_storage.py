#!/usr/bin/env python3
"""
Task 2: LLM Summarization and Database Storage
- LLM summarization (OpenAI, Claude, Gemini)
- SQL database storage for documents & summaries
- Vector database (Chroma) for embeddings & semantic search
- FastAPI endpoints for document processing
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

import openai
from anthropic import Anthropic
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
import chromadb

from config import Config

# ==========================
# Logging
# ==========================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ==========================
# SQLAlchemy Models
# ==========================
Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True)
    filename = Column(String(512), unique=True)
    document_type = Column(String(100))
    file_path = Column(String(1024))
    file_size = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Summary(Base):
    __tablename__ = "summaries"
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer)
    llm_model = Column(String(50))
    summary_text = Column(Text)
    processing_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

# ==========================
# Pydantic Schemas
# ==========================
class DocumentResponse(BaseModel):
    id: int
    filename: str
    document_type: str

class SummaryResponse(BaseModel):
    filename: str
    summary: str
    model: str
    processing_time: float

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    document_type: Optional[str] = None

# ==========================
# Database Manager
# ==========================
class DatabaseManager:
    def __init__(self, url: str = None):
        self.url = url or Config.SQL_DB_URL
        self.engine = create_engine(self.url, echo=False)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def add_document(self, filename: str, document_type: str, file_path: str, file_size: int) -> int:
        with self.SessionLocal() as session:
            existing = session.query(Document).filter_by(filename=filename).first()
            if existing:
                logger.warning(f"Duplicate file detected: {filename}, returning existing document ID {existing.id}")
                return existing.id

            doc = Document(filename=filename, document_type=document_type,
                           file_path=file_path, file_size=file_size)
            session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc.id

    def add_summary(self, document_id: int, llm_model: str, summary_text: str, processing_time: float) -> int:
        with self.SessionLocal() as session:
            summary = Summary(document_id=document_id, llm_model=llm_model,
                              summary_text=summary_text, processing_time=processing_time)
            session.add(summary)
            session.commit()
            session.refresh(summary)
            return summary.id

    def get_document_by_filename(self, filename: str):
        with self.SessionLocal() as session:
            return session.query(Document).filter(Document.filename == filename).first()

# ==========================
# LLM Summarizer
# ==========================
class LLMSummarizer:
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
        self.google_model = None

        if Config.OPENAI_API_KEY:
            openai.api_key = Config.OPENAI_API_KEY
            self.openai_client = openai

        if Config.ANTHROPIC_API_KEY:
            self.anthropic_client = Anthropic(api_key=Config.ANTHROPIC_API_KEY)

        if Config.GOOGLE_API_KEY:
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            self.google_model = genai.GenerativeModel("gemini-pro")

    async def summarize(self, text: str, model: str = "gpt-4") -> Dict:
        try:
            start = time.time()
            if model == "gpt-4" and self.openai_client:
                resp = await openai.ChatCompletion.acreate(
                    model="gpt-4",
                    messages=[{"role": "system", "content": "Summarize concisely."},
                              {"role": "user", "content": text}],
                    max_tokens=300
                )
                return {
                    "summary": resp.choices[0].message.content.strip(),
                    "model": "gpt-4",
                    "processing_time": time.time() - start
                }

            elif model == "claude" and self.anthropic_client:
                resp = self.anthropic_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    messages=[{"role": "user", "content": text}],
                    max_tokens=300
                )
                return {
                    "summary": resp.content[0].text.strip(),
                    "model": "claude",
                    "processing_time": time.time() - start
                }

            elif model == "gemini" and self.google_model:
                resp = self.google_model.generate_content(text)
                return {
                    "summary": resp.text.strip(),
                    "model": "gemini-pro",
                    "processing_time": time.time() - start
                }

            else:
                return {"summary": "", "model": model, "processing_time": 0,
                        "error": f"{model} not configured"}

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

# ==========================
# Vector DB
# ==========================
class VectorDB:
    def __init__(self, path="./chroma_db"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection_name = "documents"
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)

        try:
            self.collection = self.client.get_collection(name=self.collection_name)
        except:
            self.collection = self.client.create_collection(name=self.collection_name)

    def add_document(self, doc_id: str, content: str, metadata: Dict):
        embedding = self.encoder.encode(content).tolist()
        self.collection.add(embeddings=[embedding], documents=[content],
                            metadatas=[metadata], ids=[doc_id])

    def search(self, query: str, top_k=5, filter_metadata: Dict = None) -> List[Dict]:
        embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k, where=filter_metadata)
        return [{"document_id": id_, "content": doc, "metadata": meta}
                for id_, doc, meta in zip(results['ids'][0], results['documents'][0], results['metadatas'][0])]

# ==========================
# FastAPI App
# ==========================
app = FastAPI(title="Task 2: LLM Summarization & DB")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

db_manager = DatabaseManager()
vector_db = VectorDB()
llm_summarizer = LLMSummarizer()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        save_path = os.path.join(Config.DATA_DIR, file.filename)

        # Save file
        with open(save_path, "wb") as f:
            f.write(await file.read())

        doc_type = file.filename.split(".")[0]  # naive classification
        doc_id = db_manager.add_document(file.filename, doc_type, save_path, os.path.getsize(save_path))

        logger.info(f"Uploaded document: {file.filename} (ID: {doc_id})")
        return {"id": doc_id, "filename": file.filename, "document_type": doc_type}

    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/summarize")
async def summarize(filename: str, model: str = "gpt-4"):
    doc = db_manager.get_document_by_filename(filename)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        with open(doc.file_path, "r", encoding="utf-8") as f:
            content = f.read()

        summary_data = await llm_summarizer.summarize(content, model=model)
        db_manager.add_summary(doc.id, summary_data["model"],
                               summary_data["summary"], summary_data["processing_time"])
        return summary_data

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")

@app.post("/search")
async def search(req: SearchRequest):
    try:
        results = vector_db.search(req.query, top_k=req.top_k)
        return {"query": req.query, "results": results, "total_found": len(results)}
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.task2_llm_storage:app", host=Config.API_HOST, port=Config.API_PORT, reload=True)
