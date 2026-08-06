"""
predictor.py — HeatMind RAG query engine
─────────────────────────────────────────
Flow for every query:
  1. Embed the planner's natural-language query
  2. Apply metadata filters (steel type, route, status, buffer)
  3. Retrieve top-K most semantically similar order documents
  4. Build a structured prompt with system rules + retrieved orders + query
  5. Send to Ollama → stream back the structured recommendation
"""
import os
from pathlib import Path

# SET TELEMETRY VARS BEFORE ANY IMPORTS
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

import json
import chromadb
import requests
from sentence_transformers import SentenceTransformer

# ── CONFIG ────────────────────────────────────────────────────────────────────
DB_PATH           = "./steel_db"
COLLECTION        = "steel_orders"
TOP_K             = 15
EMBED_MODEL       = os.environ.get("EMBED_MODEL",       "all-MiniLM-L6-v2")
OLLAMA_HOST       = os.environ.get("OLLAMA_HOST",       "http://localhost:11434")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "llama3")

# ── LAZY SINGLETONS ───────────────────────────────────────────────────────────
_embed_model = None
_collection  = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        print("  Loading embedding model...")
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        db = chromadb.PersistentClient(path=DB_PATH)
        _collection = db.get_or_create_collection(
            COLLECTION,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


# ── RETRIEVE ──────────────────────────────────────────────────────────────────
def retrieve_orders(
    query: str,
    filters: dict | None = None,
    top_k: int = TOP_K,
) -> list[dict]:
    model      = get_embed_model()
    collection = get_collection()

    query_vector = model.encode(query).tolist()
    where_clause = _build_where(filters)

    count = collection.count()
    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        where=where_clause if where_clause else None,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    return [
        {
            "document": doc,
            "metadata": meta,
            "distance": round(dist, 4),
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]


def _build_where(filters: dict | None) -> dict | None:
    if not filters:
        return None

    key_map = {
        "steel_type": "steel_type",
        "route":      "route",
        "status":     "status",
        "buffer":     "buffer",
    }

    conditions = []
    for api_key, db_key in key_map.items():
        value = filters.get(api_key)
        if value and value != "":
            conditions.append({db_key: {"$eq": value}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


# ── PROMPT BUILDER ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert hot-rolling mill production planner at a steel plant.
You have deep knowledge of steelmaking metallurgy, rolling mill operations, and production scheduling.

HARD CONSTRAINTS you must always enforce:
1. A single heat must contain only ONE steel type (L, A, P, H, I, S, M — never mix).
2. Target heat mass: 200–300 MT. Hard minimum: 150 MT. Hard maximum: 350 MT.
3. Orders with very different widths (>150 mm spread) require roll changes — flag this.
4. Never mix incompatible chemistry families (e.g. pipeline grade P with low-carbon L).
5. Red buffer (R) and Black buffer (B) orders are URGENT — always prioritise them.
6. Firm orders (status=F) take priority over Open (status=O).

RESPONSE FORMAT — always structure your response exactly like this:

## PROPOSED HEAT COMPOSITION
List each recommended order with: Order ID | Grade | Width×Thickness | Mass MT | Buffer

## TOTAL HEAT MASS
State the total MT and whether it is within target range.

## RISK FLAGS
Bullet-point any risks: urgent buffers, grade mixing concerns, width spread, mass issues.

## ROLLING SEQUENCE
Suggest the order to roll these heats (widest to narrowest, thick to thin).

## EXCLUDED ORDERS
List retrieved orders NOT included and briefly explain why.

## REASONING
2–3 sentences explaining your overall recommendation logic.
"""


def build_prompt(query: str, retrieved: list[dict]) -> str:
    context_parts = []
    for i, item in enumerate(retrieved, 1):
        similarity_pct = round((1 - item["distance"]) * 100, 1)
        context_parts.append(
            f"[Order {i} — similarity {similarity_pct}%]\n{item['document']}"
        )
    context = "\n\n".join(context_parts)
    return (
        f"PLANNER QUERY:\n{query}\n\n"
        f"RETRIEVED ORDERS FROM DATABASE ({len(retrieved)} orders):\n\n"
        f"{context}"
    )


# ── NON-STREAMING PREDICT ─────────────────────────────────────────────────────
def predict(
    query: str,
    filters: dict | None = None,
    top_k: int = TOP_K,
) -> dict:
    retrieved = retrieve_orders(query, filters=filters, top_k=top_k)

    if not retrieved:
        return {
            "answer":    "No orders found matching the given filters. Try broadening your filter criteria.",
            "retrieved": [],
            "query":     query,
        }

    user_message = build_prompt(query, retrieved)
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    OLLAMA_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    answer = resp.json()["message"]["content"]

    return {
        "answer":    answer,
        "retrieved": retrieved,
        "query":     query,
    }


# ── STREAMING PREDICT ─────────────────────────────────────────────────────────
def predict_stream(
    query: str,
    filters: dict | None = None,
    top_k: int = TOP_K,
):
    yield ": ping\n\n"
    retrieved = retrieve_orders(query, filters=filters, top_k=top_k)
    yield f"data: {json.dumps({'type': 'retrieved', 'orders': retrieved})}\n\n"
    if not retrieved:
        yield f"data: {json.dumps({'type': 'answer_chunk', 'text': 'No orders found matching these filters.'})}\n\n"
        yield "data: [DONE]\n\n"
        return
    user_message = build_prompt(query, retrieved)
    resp = requests.post(
        f"{OLLAMA_HOST}/api/chat",
        json={
            "model":    OLLAMA_CHAT_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            chunk = json.loads(line)
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield f"data: {json.dumps({'type': 'answer_chunk', 'text': token})}\n\n"
            if chunk.get("done"):
                yield "data: [DONE]\n\n"
                break
