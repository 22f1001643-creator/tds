---
title: Personalized Technical RAG
sdk: docker
app_port: 7860
---

# Personalized Technical RAG API

Deploy this directory to Render using the included `render.yaml` Blueprint. It exposes `POST /query`, `GET /health`, and permissive CORS. The model and passage embeddings load at process startup; the first deploy downloads the quantized `BAAI/bge-small-en-v1.5` model through FastEmbed.

Example request:

```json
{"question":"How many nodes are needed for the Raft quorum?", "top_k":3}
```

The answer is grounded in the highest-scoring source passage. Responses with insufficient semantic evidence abstain with a null answer and empty sources. Passage text is treated as data, not instructions.
