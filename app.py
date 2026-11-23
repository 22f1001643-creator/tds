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


INJECTION = re.compile(
    r"\b(?:ignore|disregard|forget|override|bypass|circumvent|do not follow|don't follow)\b"
    r".{0,120}\b(?:instructions?|rules?|prompt|system|developer|previous|prior|above)\b"
    r"|\b(?:reveal|show|print|repeat|expose|dump)\b.{0,100}\b(?:system prompt|hidden instructions|developer message|secret|credentials|api key)\b"
    r"|\b(?:act as|pretend to be|you are now)\b.{0,100}\b(?:system|developer|admin|unrestricted)\b",
    re.I,
)
MULTI_PART = re.compile(r"\b(compare|contrast|both|difference|versus|\bvs\b|relationship|across|combine|synthesi[sz]e|while|whereas)\b", re.I)

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "to",
    "was", "what", "when", "where", "which", "who", "why", "with",
}
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
DOC_TOKENS = [set(t.lower() for t in TOKEN_RE.findall(p) if t.lower() not in STOP_WORDS) for p in PASSAGES]
DOC_DF = {}
for tokens in DOC_TOKENS:
    for token in tokens:
        DOC_DF[token] = DOC_DF.get(token, 0) + 1
N_DOCS = len(PASSAGES)
DOC_IDF = {t: float(np.log((N_DOCS + 1) / (df + 1)) + 1) for t, df in DOC_DF.items()}
DOC_TFIDF = []
for tokens in DOC_TOKENS:
    weights = {t: DOC_IDF[t] for t in tokens}
    norm = max(float(np.sqrt(sum(v * v for v in weights.values()))), 1e-12)
    DOC_TFIDF.append({t: v / norm for t, v in weights.items()})


@app.get("/health")
def health():
    return {"status": "ok", "passages": len(PASSAGES)}


@app.get("/")
def root():
    return {"status": "ok", "service": "personalized-technical-rag", "health": "/health", "query": "/query"}


@app.post("/query", response_model=Result)
def query(body: Query):
    question = body.question.strip()
    # Treat instruction-like clauses as untrusted query text, preserving factual clauses.
    clauses = re.split(r"(?<=[.!?;,:])\s*|\n+", question)
    clean_question = " ".join(part for part in clauses if part.strip() and not INJECTION.search(part)).strip()
    if not clean_question or INJECTION.search(question) and len(TOKEN_RE.findall(clean_question)) < 3:
        return Result(answer=None, sources=[], scores=[], confidence=0.1)

    query_vector = np.asarray(list(model.query_embed(clean_question))[0], dtype=np.float32)
    query_vector /= max(float(np.linalg.norm(query_vector)), 1e-12)
    dense = DOC_VECTORS @ query_vector
    q_tokens = set(t.lower() for t in TOKEN_RE.findall(clean_question) if t.lower() not in STOP_WORDS)
    q_weights = {t: DOC_IDF.get(t, float(np.log(N_DOCS + 1) + 1)) for t in q_tokens}
    q_norm = max(float(np.sqrt(sum(v * v for v in q_weights.values()))), 1e-12)
    q_weights = {t: v / q_norm for t, v in q_weights.items()}
    lexical = np.asarray([
        sum(q_weights.get(t, 0.0) * w for t, w in doc.items())
        for doc in DOC_TFIDF
    ], dtype=np.float32)
    # Semantic similarity handles paraphrases; TF-IDF overlap separates near-identical confounders.
    raw = 0.78 * dense + 0.22 * lexical
    order = np.argsort(-raw)[:body.top_k]
    best = float(raw[order[0]]) if len(order) else 0.0
    second = float(raw[order[1]]) if len(order) > 1 else 0.0
    # Require a clear semantic match and a margin over unrelated passages.
    if best < 0.36 or (best < 0.46 and lexical[order[0]] < 0.035):
        return Result(answer=None, sources=[], scores=[], confidence=round(min(0.39, max(0.05, best)), 4))
    # Single-fact questions usually need only the best passage. For explicit comparisons
    # and synthesis, retain other close, lexically supported passages.
    if MULTI_PART.search(clean_question):
        ids = [int(i) for i in order if raw[i] >= best - 0.13 and (i == order[0] or lexical[i] >= 0.06)]
        ids = ids[:body.top_k] or [int(order[0])]
    else:
        ids = [int(order[0])]
    scores = [round(float(np.clip(raw[i], 0.0, 1.0)), 4) for i in ids]
    # Return cited source text as a faithful factual answer; passages are untrusted data.
    answer = " ".join(PASSAGES[i] for i in ids)
    confidence = float(np.clip(0.61 + (best - 0.40) * 0.8, 0.61, 0.98))
    return Result(answer=answer, sources=ids, scores=scores, confidence=round(confidence, 4))
