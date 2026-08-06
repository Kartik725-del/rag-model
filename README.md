<p align="center">
  <img src="logo.png" alt="HeatMind logo" width="420"/>
</p>

<h1 align="center">HeatMind</h1>
<p align="center"><b>RAG-powered Heat Plan Advisor for Hot-Rolling Mill Production Planning</b></p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="FastAPI" src="https://img.shields.io/badge/backend-FastAPI-009688">
  <img alt="ChromaDB" src="https://img.shields.io/badge/vector%20store-ChromaDB-orange">
  <img alt="Ollama" src="https://img.shields.io/badge/LLM-Ollama-black">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

## Overview

**HeatMind** is a Retrieval-Augmented Generation (RAG) assistant built for steel plant production planners. It ingests raw hot-rolling order data (CSV exports from a mill's order system), indexes it as vector embeddings, and lets planners ask natural-language questions — *"Which orders can I combine for a 250 MT heat of grade T01576?"* — to get a structured, metallurgically-aware heat plan recommendation, generated locally by an LLM via Ollama.

Everything runs **locally**: embeddings are generated with `sentence-transformers`, orders are stored in a local `ChromaDB` instance, and the recommendation itself is produced by a locally-hosted Ollama model — no order data ever leaves the machine.

## How It Works

```
CSV of mill orders ─► ingest.py cleans & embeds rows ─► stored in ChromaDB
                                                              │
Planner asks a question in the UI ─► predictor.py embeds + filters + retrieves top-K orders
                                                              │
                              Structured prompt (rules + retrieved orders + query)
                                                              │
                         Ollama LLM ─► streamed recommendation ─► FastAPI (SSE) ─► Web UI
```

The system enforces hard steelmaking constraints in its prompt (single steel type per heat, target heat mass, chemistry compatibility, buffer/urgency prioritization, width-spread roll-change flags) so the LLM's recommendations stay within real operational limits.

## Features

- 🔍 **Semantic order search** — natural-language queries over thousands of mill orders using local embeddings.
- 🎛 **Metadata filtering** — filter retrieval by steel type, production route, order status, and buffer indicator.
- 🤖 **RAG-based heat plan generation** — structured recommendations covering proposed composition, total mass, risk flags, rolling sequence, and excluded orders.
- ⚡ **Streaming responses** — recommendations stream token-by-token to the UI via Server-Sent Events.
- 📤 **CSV drag-and-drop ingestion** — upload a new orders CSV directly from the browser; indexing runs in the background with live progress polling.
- 📊 **Live dashboard KPIs** — order counts, red/urgent buffer counts, and top production routes.
- 📁 **Heat export** — export a proposed heat plan as CSV.
- 🔒 **Fully local** — embeddings, vector store, and LLM inference all run on your own infrastructure; no external API calls.

## Tech Stack

| Layer            | Technology                                   |
|-------------------|-----------------------------------------------|
| Backend API       | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn |
| Vector store      | [ChromaDB](https://www.trychroma.com/) (persistent, local) |
| Embeddings        | [sentence-transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2` by default) |
| LLM inference     | [Ollama](https://ollama.com/) (local model, e.g. `llama3`) |
| Data processing   | pandas, numpy |
| Frontend          | Single-page HTML/CSS/vanilla JS UI (no build step) |

## Project Structure

```
.
├── main.py                    # FastAPI app — routes, ingestion endpoint, streaming query endpoint
├── ingest.py                  # CSV → cleaned text documents → embeddings → ChromaDB
├── predictor.py                # Retrieval + prompt construction + Ollama query (sync & streaming)
├── steel_heat_advisor.html     # Frontend dashboard served at "/"
├── logo.png                    # Project logo
├── requirements.txt             # Python dependencies
├── data/                       # (created at runtime) uploaded/ingested CSV files
├── steel_db/                    # (created at runtime) persistent ChromaDB storage
└── .gitignore
```

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally, with a chat model pulled (e.g. `ollama pull llama3`)
- A CSV of mill order data (see [Data Format](#data-format) below)

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/heatmind.git
cd heatmind
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
EMBED_MODEL=all-MiniLM-L6-v2
OLLAMA_HOST=http://localhost:11434
OLLAMA_CHAT_MODEL=llama3
```

### 4. Add your orders data

Place your orders CSV under `data/`, e.g.:

```
data/SummerTrainee_SampleDataset_052026.csv
```

(Alternatively, skip this step and use the drag-and-drop CSV upload in the web UI once the server is running.)

### 5. Build the vector index

```bash
python ingest.py --csv data/SummerTrainee_SampleDataset_052026.csv --db ./steel_db --reset
```

### 6. Start the server

```bash
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000** in your browser.

## Data Format

`ingest.py` expects a CSV with (at minimum) the following columns:

| Column | Description |
|---|---|
| `THO_ID_ORDER`, `THO_ID_ORDER_ITEM` | Order and order-item identifiers |
| `THO_STL_TYPE` | Steel type family (L, A, P, H, I, S, M) |
| `THO_HR_QLTY`, `THO_QLTY_CD` | Grade name and quality code |
| `THO_HR_WIDTH`, `THO_HR_THICK` | Hot-roll width / thickness (mm) |
| `THO_ORDER_MASS`, `THO_MS_PLANNED` | Ordered mass and planned dispatch mass (MT) |
| `THO_TDC_ROUTE_DESC`, `THO_FPATH` | Production route and flow path |
| `THO_ORDER_STATUS` | Order status (e.g. F = Firm, O = Open) |
| `THO_TOC_BUFFER_IND` | Buffer/urgency indicator (R, B, Y, I, G) |
| `THO_UTR`, `THO_HRWK`, `THO_YR_WK_DIS` | Additional scheduling fields |
| `THO_MIN_DATE_SC`, `THO_PLANENDDATE` | Scheduling dates |
| `THO_FP_BAL_TOPLN`, `THO_CD_PROCESS`, `THO_PROD_CD` | Balance, process, and product codes |

Rows are de-duplicated on `(THO_ID_ORDER, THO_ID_ORDER_ITEM)`, and missing values are backfilled with sensible defaults during cleaning.

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `GET` | `/health` | Health check + indexed order count |
| `GET` | `/api/stats` | Buffer/status/route breakdown for the dashboard |
| `POST` | `/api/orders` | Retrieve orders by metadata filters (steel type, route, status, buffer, width range) |
| `POST` | `/api/query` | Non-streaming RAG query → full heat plan recommendation |
| `POST` | `/api/query/stream` | Streaming RAG query (Server-Sent Events) |
| `POST` | `/api/ingest` | Upload a new CSV and trigger background re-indexing |
| `GET` | `/api/ingest/status` | Poll status of a background ingestion job |
| `GET` | `/api/export?heat_id=...` | Export a heat plan as CSV |

### Example query

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
        "query": "Combine orders for a 250 MT heat of grade T01576",
        "steel_type": "L",
        "top_k": 15
      }'
```

## Heat Planning Rules

The advisor enforces the following constraints when composing a recommendation:

1. A single heat contains only **one** steel type family (L, A, P, H, I, S, M — never mixed).
2. Target heat mass: **200–300 MT** (hard limits: 150–350 MT).
3. Orders with a width spread greater than **150 mm** are flagged (roll change required).
4. Incompatible chemistry families are never mixed (e.g. pipeline grade `P` with low-carbon `L`).
5. **Red** and **Black** buffer orders are treated as urgent and prioritized.
6. **Firm** orders take priority over **Open** orders.

## Roadmap Ideas

- [ ] Authentication / role-based access for planners
- [ ] Multi-model support (swap Ollama models per query)
- [ ] Historical recommendation logging and audit trail
- [ ] Automated heat-plan approval workflow

## Contributing

Contributions are welcome. Please open an issue to discuss significant changes before submitting a pull request.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes
4. Open a pull request

---

<p align="center"><sub>Built for hot-rolling mill production planning · Plant 738</sub></p>
