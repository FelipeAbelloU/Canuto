"""API REST FastAPI para integración con SIRIUS (Django).

Ejecutar:
    uvicorn src.api.app:app --reload --port 8000

Endpoints:
    GET  /        → estado del servicio
    GET  /health  → healthcheck para SIRIUS
    POST /query   → consulta al asistente normativo
    POST /reset   → reinicia historial de conversación
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.factory import create_pipeline
from src.chat.pipeline import ChatPipeline

app = FastAPI(
    title="Asistente Normativo Unillanos — API",
    description=(
        "API REST para consultas sobre normativa universitaria de Unillanos. "
        "Microservicio para integración con el sistema SIRIUS."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: ChatPipeline | None = None


def get_pipeline() -> ChatPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = create_pipeline()
    return _pipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Pregunta del usuario")
    session_id: str | None = Field(None, description="Reservado para uso futuro")


class QueryResponse(BaseModel):
    answer: str
    model_used: str


@app.get("/", tags=["Estado"])
def root():
    return {
        "service": "Asistente Normativo Unillanos",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health", tags=["Estado"])
def health():
    pipeline = get_pipeline()
    model_ready = pipeline.model is not None and pipeline.model.is_available()
    return {
        "status": "healthy" if model_ready else "degraded",
        "model_ready": model_ready,
        "model": pipeline.model_name,
    }


@app.post("/query", response_model=QueryResponse, tags=["Chat"])
def query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    pipeline = get_pipeline()
    result = pipeline.query(req.question)
    return QueryResponse(answer=result["answer"], model_used=result["model_used"])


@app.post("/reset", tags=["Chat"])
def reset_conversation():
    get_pipeline().reset_history()
    return {"status": "ok", "message": "Historial reiniciado."}
