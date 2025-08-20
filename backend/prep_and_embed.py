# backend/prep_and_embed.py
"""
Create an SBERT index for bert_symptom_checker.py.

Reads (first available):
  data/unified_dataset_filtered.json
  data/unified_dataset.json
If empty/missing, falls back to a small built-in seed.

Writes:
  data/sbert_index.npz   (array "X")
  data/sbert_meta.json   (list of {"text","risk"})
"""
from __future__ import annotations

import os, json
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

try:
    from .symptom_rules import rule_based_score, classify_risk  # type: ignore
except Exception:
    from symptom_rules import rule_based_score, classify_risk  # type: ignore

DATA_DIR   = "data"
FILTERED   = os.path.join(DATA_DIR, "unified_dataset_filtered.json")
RAW        = os.path.join(DATA_DIR, "unified_dataset.json")
INDEX_PATH = os.path.join(DATA_DIR, "sbert_index.npz")
META_PATH  = os.path.join(DATA_DIR, "sbert_meta.json")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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

def _read_texts(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        out = []
        for row in data:
            if isinstance(row, dict):
                t = str(row.get("text", "")).strip()
            else:
                t = str(row).strip()
            if len(t) >= 10:
                out.append(t)
        # dedupe
        seen, keep = set(), []
        for t in out:
            key = t.lower()
            if key in seen: 
                continue
            seen.add(key)
            keep.append(t)
        return keep
    except Exception:
        return []

def load_texts() -> List[str]:
    texts = _read_texts(FILTERED) or _read_texts(RAW)
    if not texts:
        print("Dataset empty/missing — using small built-in seed list.")
        texts = OFFLINE_SEED[:]
    return texts

def label_texts(texts: List[str]) -> List[Tuple[str, str, int]]:
    rows: List[Tuple[str, str, int]] = []
    for t in texts:
        score, _ = rule_based_score(t)
        risk = classify_risk(score) if score > 0 else "LOW"
        rows.append((t, risk, int(score)))
    return rows

def embed_and_save(rows: List[Tuple[str, str, int]]) -> None:
    texts = [r[0] for r in rows]
    risks = [r[1] for r in rows]

    print(f"Loading SBERT model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"Encoding {len(texts)} texts…")
    X = model.encode(texts, normalize_embeddings=True)
    X = np.asarray(X, dtype=np.float32)

    meta = [{"text": t, "risk": risks[i]} for i, t in enumerate(texts)]

    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(INDEX_PATH, X=X)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Wrote {INDEX_PATH} (shape={X.shape})")
    print(f"Wrote {META_PATH} (rows={len(meta)})")
    print("Done.")

def main() -> None:
    texts = load_texts()
    print(f"Loaded {len(texts)} texts.")
    rows = label_texts(texts)
    counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for _, r, _ in rows:
        counts[r] = counts.get(r, 0) + 1
    print(f"Rule labels -> {counts}")
    embed_and_save(rows)

if __name__ == "__main__":
    main()
