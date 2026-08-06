import argparse
import os
import sys
import time
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
CSV_PATH   = "data/SummerTrainee_SampleDataset_052026.csv"
DB_PATH    = "./steel_db"
COLLECTION = "steel_orders"
BATCH_SIZE = 32
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
NULL_COLS   = ["THO_FP_BALCAST", "THO_DUE_DT", "THO_FL_I2FP_CLS"]


# ── ROW → TEXT ────────────────────────────────────────────────────────────────
def order_to_text(row: pd.Series) -> str:
    return (
        f"Order {row['THO_ID_ORDER']}-{row['THO_ID_ORDER_ITEM']} "
        f"for steel type {row['THO_STL_TYPE']}, "
        f"grade {row['THO_HR_QLTY']} (code {row['THO_QLTY_CD']}), "
        f"hot-roll width {row['THO_HR_WIDTH']} mm, "
        f"thickness {row['THO_HR_THICK']} mm. "
        f"Ordered mass {row['THO_ORDER_MASS']} MT, "
        f"planned dispatch mass {row['THO_MS_PLANNED']} MT. "
        f"Production route: {row['THO_TDC_ROUTE_DESC']}. "
        f"Flow path: {row['THO_FPATH']}. "
        f"Order status: {row['THO_ORDER_STATUS']}. "
        f"Buffer indicator: {row['THO_TOC_BUFFER_IND']}. "
        f"UTR flag: {row['THO_UTR']}. "
        f"Production week: {row['THO_HRWK']}, "
        f"dispatch week: {row['THO_YR_WK_DIS']}. "
        f"Earliest SC date: {row['THO_MIN_DATE_SC']}. "
        f"Plan end date: {row['THO_PLANENDDATE']}. "
        f"Balance on top-line order: {row['THO_FP_BAL_TOPLN']} MT. "
        f"Process code: {row['THO_CD_PROCESS']}. "
        f"Product code: {row['THO_PROD_CD']}."
    ).strip()


# ── CLEAN CSV ─────────────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    existing_null_cols = [c for c in NULL_COLS if c in df.columns]
    df = df.drop(columns=existing_null_cols)

    str_defaults = {
        "THO_STL_TYPE":       "UNKNOWN",
        "THO_TDC_ROUTE_DESC": "UNKNOWN",
        "THO_FPATH":          "UNKNOWN",
        "THO_QLTY_CD":        "UNASSIGNED",
        "THO_TDC_NO":         "UNASSIGNED",
        "THO_TOC_BUFFER_IND": "UNKNOWN",
        "THO_UTR":            "N",
        "THO_MIN_DATE_SC":    "TBD",
        "THO_PLANENDDATE":    "TBD",
        "THO_YR_WK_DIS":      "TBD",
        "THO_CD_PROCESS":     "UNKNOWN",
        "THO_PROD_CD":        "UNKNOWN",
        "THO_HR_QLTY":        "UNKNOWN",
    }
    num_defaults = {
        "THO_ORDER_MASS":   0.0,
        "THO_MS_PLANNED":   0.0,
        "THO_HR_WIDTH":     0.0,
        "THO_HR_THICK":     0.0,
        "THO_FP_BAL_TOPLN": 0.0,
        "THO_MS_PRT_ORDER": 0.0,
    }

    for col, default in str_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default).astype(str)
    for col, default in num_defaults.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    before = len(df)
    df = df.drop_duplicates(subset=["THO_ID_ORDER", "THO_ID_ORDER_ITEM"])
    after = len(df)
    if before != after:
        print(f"  Removed {before - after} duplicate rows.")

    return df.reset_index(drop=True)


# ── BUILD INDEX ───────────────────────────────────────────────────────────────
def build_index(csv_path: str = CSV_PATH, db_path: str = DB_PATH, reset: bool = False):
    print(f"\n{'─'*60}")
    print("  HeatMind — Ingestion Pipeline")
    print(f"{'─'*60}")
    print(f"  CSV      : {csv_path}")
    print(f"  Vector DB: {db_path}")
    print(f"  Embedder : {EMBED_MODEL} via sentence-transformers")
    print(f"{'─'*60}\n")

    print("[ 1/4 ] Loading CSV...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  ERROR: CSV not found at '{csv_path}'")
        sys.exit(1)

    print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns.")
    df = clean_dataframe(df)
    print(f"  After cleaning: {len(df):,} valid rows.\n")

    print("[ 2/4 ] Converting orders to text documents...")
    documents, ids, metadatas = [], [], []

    for _, row in df.iterrows():
        doc_id = f"{row['THO_ID_ORDER']}-{row['THO_ID_ORDER_ITEM']}"
        documents.append(order_to_text(row))
        ids.append(doc_id)
        metadatas.append({
            "order_id":   str(row["THO_ID_ORDER"]),
            "steel_type": str(row["THO_STL_TYPE"]),
            "grade":      str(row["THO_HR_QLTY"]),
            "route":      str(row["THO_TDC_ROUTE_DESC"]),
            "status":     str(row["THO_ORDER_STATUS"]),
            "buffer":     str(row["THO_TOC_BUFFER_IND"]),
            "width":      str(row["THO_HR_WIDTH"]),
            "thickness":  str(row["THO_HR_THICK"]),
            "mass":       str(row["THO_ORDER_MASS"]),
            "week":       str(row["THO_HRWK"]),
        })

    print(f"  Built {len(documents):,} text documents.\n")

    print("[ 3/4 ] Embedding documents...")
    t0    = time.time()
    model = SentenceTransformer(EMBED_MODEL)

    embeddings = []
    for i in range(0, len(documents), BATCH_SIZE):
        batch      = documents[i:i + BATCH_SIZE]
        batch_vecs = model.encode(batch).tolist()
        embeddings.extend(batch_vecs)
        print(f"  Embedded {min(i + BATCH_SIZE, len(documents)):,} / {len(documents):,}", end="\r")

    print(f"\n  Embedded {len(embeddings):,} documents in {time.time() - t0:.1f}s.\n")

    print(f"[ 4/4 ] Storing in ChromaDB at '{db_path}'...")
    client = chromadb.PersistentClient(path=db_path)

    if reset:
        print("  Reset flag set — deleting existing collection...")
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )

    chunk = 500
    for i in range(0, len(documents), chunk):
        collection.upsert(
            documents=documents[i:i+chunk],
            embeddings=embeddings[i:i+chunk],
            ids=ids[i:i+chunk],
            metadatas=metadatas[i:i+chunk],
        )
        print(f"  Stored {min(i+chunk, len(documents)):,} / {len(documents):,}", end="\r")

    print(f"\n  Collection '{COLLECTION}' now has {collection.count():,} documents.")
    print(f"\n{'─'*60}")
    print("  Ingestion complete. Start the server with:")
    print("    uvicorn main:app --reload")
    print(f"{'─'*60}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HeatMind — ingest orders into ChromaDB")
    parser.add_argument("--csv",   default=CSV_PATH, help="Path to the orders CSV")
    parser.add_argument("--db",    default=DB_PATH,  help="ChromaDB storage directory")
    parser.add_argument("--reset", action="store_true", help="Delete and rebuild the collection")
    args = parser.parse_args()
    build_index(csv_path=args.csv, db_path=args.db, reset=args.reset)
