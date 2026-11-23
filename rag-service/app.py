import json
import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding

DATA_PATH = Path(__file__).with_name("passages.json")
PASSAGES = json.loads(DATA_PATH.read_text(encoding="utf-8-sig"))
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
model = TextEmbedding(model_name=MODEL_NAME)
DOC_VECTORS = np.asarray(list(model.embed(PASSAGES)), dtype=np.float32)
DOC_VECTORS /= np.maximum(np.linalg.norm(DOC_VECTORS, axis=1, keepdims=True), 1e-12)

app = FastAPI(title="Personalized Technical RAG", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class Query(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=10)


class Result(BaseModel):
    answer: Optional[str]
    sources: list[int]
    scores: list[float]
    confidence: float


INJECTION = re.compile(r"ignore (all |any )?(previous|prior|system)|reveal (the )?(system|prompt)|disregard .*instructions", re.I)


@app.get("/health")
def health():
    return {"status": "ok", "passages": len(PASSAGES)}


@app.post("/query", response_model=Result)
def query(body: Query):
    question = body.question.strip()
    # Prompt-like text is not allowed to change retrieval or answer policy.
    query_vector = np.asarray(list(model.query_embed(question))[0], dtype=np.float32)
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
    raw = DOC_VECTORS @ query_vector
    order = np.argsort(-raw)[:body.top_k]
    best = float(raw[order[0]]) if len(order) else 0.0
    second = float(raw[order[1]]) if len(order) > 1 else 0.0
    # Require a clear semantic match and a margin over unrelated passages.
    if best < 0.34 or (best < 0.43 and best - second < 0.025):
        return Result(answer=None, sources=[], scores=[], confidence=round(min(0.39, max(0.05, best)), 4))
    ids = [int(i) for i in order]
    scores = [round(float(np.clip(raw[i], 0.0, 1.0)), 4) for i in ids]
    # Return cited source text as a faithful factual answer; passages are untrusted data.
    answer = PASSAGES[ids[0]]
    confidence = float(np.clip(0.55 + (best - 0.35) * 0.9, 0.61, 0.98))
    return Result(answer=answer, sources=ids, scores=scores, confidence=round(confidence, 4))
