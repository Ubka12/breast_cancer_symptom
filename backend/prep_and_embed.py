# backend/prep_and_embed.py
"""
Build the SBERT index used by bert_symptom_checker.py.

Reads (first that exists, in backend/data/):
  - unified_dataset_filtered.json
  - unified_dataset.json
If neither exists or are empty, uses a small offline seed list.

Writes (to backend/data/):
  - sbert_index.npz  (array "X": float32 [N, d])
  - sbert_meta.json  (list of {"text": ..., "risk": ...})
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

# Rule labels come from your deterministic engine
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

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Fallback seed covers NHS symptoms in plain English
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
    texts = _read_texts(FILTERED) or _read_texts(RAW)
    if not texts:
        print("[prep] dataset empty/missing — using small built-in seed list.")
        texts = OFFLINE_SEED[:]
    return texts


def label_texts(texts: List[str]) -> List[Tuple[str, str, int]]:
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    texts = [r[0] for r in rows]
    risks = [r[1] for r in rows]

    print(f"[prep] loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[prep] encoding {len(texts)} texts (batch_size={batch_size}) …")
    X = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=len(texts) > batch_size,
    ).astype(np.float32)

    meta: List[Dict[str, str]] = [{"text": t, "risk": risks[i]} for i, t in enumerate(texts)]

    # sanity checks
    assert X.shape[0] == len(meta), "embedding/meta length mismatch"

    np.savez_compressed(INDEX_PATH, X=X)
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[prep] wrote {INDEX_PATH}  shape={X.shape}")
    print(f"[prep] wrote {META_PATH}  rows={len(meta)}")
    print("[prep] done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SBERT index and metadata.")
    parser.add_argument("--model", default=MODEL_NAME, help="SentenceTransformer model name")
    parser.add_argument("--batch-size", type=int, default=64, help="Encoding batch size")
    args = parser.parse_args()

    texts = load_texts()
    print(f"[prep] loaded {len(texts)} texts.")
    rows = label_texts(texts)

    # small label distribution print
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, r, _ in rows:
        counts[r] = counts.get(r, 0) + 1
    print(f"[prep] rule labels → {counts}")

    embed_and_save(rows, model_name=args.model, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
