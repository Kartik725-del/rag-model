"""
Start the server:
    uvicorn main:app --reload --port 8000

Then open: http://localhost:8000
"""
import os
from pathlib import Path
os.environ["ANONYMIZED_TELEMETRY"] = "false"
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
import json
import io
import threading
from typing import Optional
from fastapi import FastAPI, Query, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from predictor import predict, predict_stream, retrieve_orders, get_collection
from ingest import build_index
# ── INGESTION STATE ───────────────────────────────────────────────────────────
_ingest_status: dict = {
    "state":    "idle",
    "message":  "",
    "indexed":  0,
    "filename": "",
}
_ingest_lock = threading.Lock()


def _run_ingest_background(csv_path: str, filename: str):
    global _ingest_status
    with _ingest_lock:
        _ingest_status = {"state": "running", "message": "Ingesting…", "indexed": 0, "filename": filename}
    try:
        build_index(csv_path=csv_path, db_path=str(DB_PATH), reset=True)
        import predictor as pred
        pred._collection = None
        count = get_collection().count()
        with _ingest_lock:
            _ingest_status = {
                "state":    "done",
                "message":  f"Ingested {count:,} orders from {filename}",
                "indexed":  count,
                "filename": filename,
            }
    except Exception as exc:
        with _ingest_lock:
            _ingest_status = {
                "state":    "error",
                "message":  str(exc),
                "indexed":  0,
                "filename": filename,
            }


# ── LIFESPAN ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    from predictor import get_embed_model, get_collection
    print("\n  HeatMind server starting up...")
    try:
        get_embed_model()
        count = get_collection().count()
        print(f"  Embedding model loaded. {count:,} orders in vector DB.")
        if count == 0:
            print("  WARNING: Vector DB is empty. Run 'python ingest.py' first.")
    except Exception as e:
        print(f"  WARNING: Could not pre-load models: {e}")
    yield


# ── APP ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       ="HeatMind — RAG Heat Plan Advisor",
    description ="AI-powered heat plan advisor",
    version     ="1.0.0",
    lifespan    =lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
load_dotenv()
import chromadb
HTML_PATH = Path(__file__).parent / "steel_heat_advisor.html"
DATA_PATH = Path(__file__).parent / "data"
DB_PATH   = Path(__file__).parent / "steel_db"
LOGO_PATH = Path(__file__).parent / "logo.png"


# ── MODELS ────────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:      str
    steel_type: Optional[str] = None
    route:      Optional[str] = None
    status:     Optional[str] = None
    buffer:     Optional[str] = None
    top_k:      int = 15


class FilterRequest(BaseModel):
    steel_type: Optional[str] = None
    route:      Optional[str] = None
    status:     Optional[str] = None
    buffer:     Optional[str] = None
    width_min:  Optional[float] = None
    width_max:  Optional[float] = None
    top_k:      int = 50


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    if not HTML_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Frontend not found at {HTML_PATH}.")
    return HTMLResponse(content=HTML_PATH.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    try:
        count = get_collection().count()
        return {
            "status":         "ok",
            "orders_indexed": count,
            "db_path":        str(DB_PATH),
            "ollama_host":    os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(e)})


@app.post("/api/orders")
async def get_orders(req: FilterRequest):
    filters = {
        "steel_type": req.steel_type,
        "route":      req.route,
        "status":     req.status,
        "buffer":     req.buffer,
    }
    results = retrieve_orders(query="steel order for hot rolling", filters=filters, top_k=req.top_k)

    if req.width_min is not None or req.width_max is not None:
        results = [
            r for r in results
            if (req.width_min is None or float(r["metadata"].get("width", 0)) >= req.width_min)
            and (req.width_max is None or float(r["metadata"].get("width", 0)) <= req.width_max)
        ]
    return {"orders": results, "count": len(results)}


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest):
    filters = {
        "steel_type": req.steel_type,
        "route":      req.route,
        "status":     req.status,
        "buffer":     req.buffer,
    }

    def event_generator():
        try:
            yield from predict_stream(query=req.query, filters=filters, top_k=req.top_k)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'text': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",
            "X-Content-Type-Options":      "nosniff",
            "Connection":                  "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )
@app.post("/api/query")
async def query_sync(req: QueryRequest):
    filters = {
        "steel_type": req.steel_type,
        "route":      req.route,
        "status":     req.status,
        "buffer":     req.buffer,
    }
    try:
        return predict(query=req.query, filters=filters, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG pipeline error: {str(e)}")
    
@app.post("/api/ingest")
async def ingest_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    with _ingest_lock:
        if _ingest_status["state"] == "running":
            raise HTTPException(status_code=409, detail="An ingestion is already in progress. Please wait.")

    DATA_PATH.mkdir(exist_ok=True)
    save_path = DATA_PATH / file.filename
    save_path.write_bytes(await file.read())

    threading.Thread(
        target=_run_ingest_background,
        args=(str(save_path), file.filename),
        daemon=True,
    ).start()

    return {"status": "started", "filename": file.filename}

@app.get("/api/ingest/status")
async def ingest_status():
    with _ingest_lock:
        return dict(_ingest_status)

@app.get("/api/export")
async def export_heat(heat_id: str = Query(default="HEAT-EXPORT")):
    rows = [
        "heat_id,order_id,grade,width_mm,thick_mm,mass_mt,buffer,status,route",
        f"{heat_id},EXAMPLE-001,T01576,1460,2.5,258,R,O,HR_DOM",
    ]
    return StreamingResponse(
        io.StringIO("\n".join(rows)),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={heat_id}.csv"},
    )

@app.get("/api/stats")
async def db_stats():
    try:
        collection = get_collection()
        total      = collection.count()
        sample     = collection.get(limit=min(total, 5000), include=["metadatas"])
        metas      = sample["metadatas"]

        buffer_counts, status_counts, route_counts = {}, {}, {}
        for m in metas:
            buf = m.get("buffer", "?");  buffer_counts[buf] = buffer_counts.get(buf, 0) + 1
            sta = m.get("status", "?");  status_counts[sta] = status_counts.get(sta, 0) + 1
            rt  = m.get("route",  "?");  route_counts[rt]   = route_counts.get(rt,  0) + 1

        return {
            "total_orders":  total,
            "buffer_counts": buffer_counts,
            "status_counts": status_counts,
            "top_routes":    dict(sorted(route_counts.items(), key=lambda x: -x[1])[:10]),
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"error": str(e)})

@app.get("/logo.png")
async def serve_logo():
    return FileResponse(LOGO_PATH)