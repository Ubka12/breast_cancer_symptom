# backend/prep_and_embed.py
"""
Build the SBERT index used by bert_symptom_checker.py.

What this script does (one-time preparation step):
  1) Load a set of short symptom sentences (from JSON in backend/data/ or a small seed list).
  2) Label each sentence with the deterministic rules (HIGH / MEDIUM / LOW).
  3) Encode the sentences with a sentence-transformer (SBERT).
  4) Save:
       - sbert_index.npz 
       - sbert_meta.json 

Inputs (first that exists, under backend/data/):
  - unified_dataset_filtered.json
  - unified_dataset.json
If neither exists or is empty, a small offline seed list is used.

Outputs (under backend/data/):
  - sbert_index.npz
  - sbert_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

# Rule labels come from the deterministic engine (same as runtime)
try:
    from .symptom_rules import rule_based_score, classify_risk  # type: ignore
except Exception:
    from symptom_rules import rule_based_score, classify_risk  # type: ignore

# ---------- Locations (always under backend/data) ----------
DATA_DIR   = (Path(__file__).resolve().parent / "data").resolve()
FILTERED   = DATA_DIR / "unified_dataset_filtered.json"
RAW        = DATA_DIR / "unified_dataset.json"
INDEX_PATH = DATA_DIR / "sbert_index.npz"
META_PATH  = DATA_DIR / "sbert_meta.json"

# Model used to embed sentences (compact, fast)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Fallback seed: plain-English paraphrases of NHS symptom statements
OFFLINE_SEED = [
    "bloody nipple discharge",
    "new nipple inversion",
    "breast skin looks like orange peel (peau d'orange)",
    "dimpling or puckering of the breast skin",
    "redness or a new rash around the nipple",
    "thickening or hardening in part of the breast",
    "change in size or shape of one breast",
    "lump in the breast",
    "swelling or lump in the armpit",
    "persistent breast pain not linked to periods",
    "flaky or crusty skin on the nipple",
    "hot inflamed breast skin",
    "clear nipple discharge",
    "itchy breast skin",
]

# ---------- IO helpers ----------
def _read_texts(path: Path) -> List[str]:
    """
    Load a JSON list of sentences (or objects with a 'text' field) from 'path'.
    - Silently returns [] if the file is missing or invalid.
    - Drops very short strings (<10 chars).
    - De-duplicates case-insensitively while preserving order.
    """
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: List[str] = []
    for row in data:
        t = (row.get("text") if isinstance(row, dict) else row) or ""
        t = str(t).strip()
        if len(t) >= 10:
            out.append(t)

    # de-dupe case-insensitively while preserving order
    seen, keep = set(), []
    for t in out:
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        keep.append(t)
    return keep


def load_texts() -> List[str]:
    """
    Load candidate sentences from the filtered JSON if present,
    otherwise from the raw JSON. If both are missing/empty, use the seed list.
    """
    texts = _read_texts(FILTERED) or _read_texts(RAW)
    if not texts:
        print("[prep] dataset empty/missing — using small built-in seed list.")
        texts = OFFLINE_SEED[:]
    return texts


def label_texts(texts: List[str]) -> List[Tuple[str, str, int]]:
    """
    Assign a rule-based label to each sentence.
    Returns a list of tuples: (text, risk_label, numeric_score)
    - Rule engine provides a numeric score; we map it to LOW/MEDIUM/HIGH.
    - If the score is 0, we set risk to LOW (neutral baseline).
    """
    rows: List[Tuple[str, str, int]] = []
    for t in texts:
        score, _ = rule_based_score(t)
        risk = classify_risk(score) if score > 0 else "LOW"
        rows.append((t, risk, int(score)))
    return rows


def embed_and_save(
    rows: List[Tuple[str, str, int]],
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> None:
    """
    Encode sentences with SBERT and persist the index and metadata.

    - rows: list of (text, risk, score) from label_texts()
    - model_name: sentence-transformers model identifier
    - batch_size: larger batches speed up encoding for bigger lists

    Produces:
      • sbert_index.npz with array X (float32, shape [N, D]), embeddings are unit-normalised
      • sbert_meta.json with [{ "text": str, "risk": "LOW|MEDIUM|HIGH" }, ...]
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    texts = [r[0] for r in rows]
    risks = [r[1] for r in rows]

    print(f"[prep] loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[prep] encoding {len(texts)} texts (batch_size={batch_size}) …")
    X = model.encode(
        texts,
        normalize_embeddings=True,     # ensures cosine = dot product
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
    ).astype(np.float32)

    # meta rows align with embedding rows by index
    meta: List[Dict[str, str]] = [{"text": t, "risk": risks[i]} for i, t in enumerate(texts)]

    # quick sanity check to avoid mismatches
    assert X.shape[0] == len(meta), "embedding/meta length mismatch"

    np.savez_compressed(INDEX_PATH, X=X)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prep] wrote {INDEX_PATH}  shape={X.shape}")
    print(f"[prep] wrote {META_PATH}  rows={len(meta)}")
    print("[prep] done.")


def main() -> None:
    """
    CLI entry point:
      python backend/prep_and_embed.py --model <name> --batch-size 64
    """
    parser = argparse.ArgumentParser(description="Prepare SBERT index and metadata.")
    parser.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    args = parser.parse_args()

    texts = load_texts()
    print(f"[prep] loaded {len(texts)} texts.")
    rows = label_texts(texts)

    # small label distribution print (useful to sanity-check dataset balance)
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, r, _ in rows:
        counts[r] = counts.get(r, 0) + 1
    print(f"[prep] rule labels → {counts}")

    embed_and_save(rows, model_name=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
