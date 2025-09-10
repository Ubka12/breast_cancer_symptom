# backend/bert_symptom_checker.py
# ------------------------------------------------------------
# SBERT similarity stage
#
# Goal:
#   Given free-text symptoms, find the most similar exemplar sentence
#   and return that exemplar’s risk label with a cosine similarity score.
#
# How it works:
#   • Load a small sentence-transformer (default: all-MiniLM-L6-v2).
#   • Load a prebuilt index of exemplar embeddings (or build it once from CSV / seed).
#   • Encode the query, cosine-match against the index, pick the nearest exemplar.

from __future__ import annotations

from pathlib import Path
import os
import json
import csv
from typing import List, Dict, Tuple, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

# ----------------------------
# Lazy-loaded globals (kept in memory after first use)
# ----------------------------
_MODEL: Optional[SentenceTransformer] = None
_X: Optional[np.ndarray] = None            # (N, D) float32; rows are unit-normalised exemplar embeddings
_META: Optional[List[Dict]] = None         # length N; each row like {"text": str, "risk": "LOW|MEDIUM|HIGH"}

# Built-in fallback exemplars (used only if no CSV or saved index exists)
_OFFLINE_SEED = [
    {"text": "bloody nipple discharge", "risk": "HIGH"},
    {"text": "new nipple inversion", "risk": "HIGH"},
    {"text": "breast skin looks like orange peel", "risk": "HIGH"},
    {"text": "dimpling or puckering of the breast skin", "risk": "HIGH"},
    {"text": "redness or a new rash around the nipple", "risk": "HIGH"},
    {"text": "thickening or hardening in part of the breast", "risk": "HIGH"},
    {"text": "change in size or shape of one breast", "risk": "MODERATE"},
    {"text": "lump in the breast", "risk": "HIGH"},
    {"text": "swelling or lump in the armpit", "risk": "HIGH"},
    {"text": "persistent breast pain not linked to periods", "risk": "MODERATE"},
    {"text": "flaky or crusty skin on the nipple", "risk": "MODERATE"},
]

# ----------------------------
# Paths & loading helpers
# ----------------------------
def _candidate_data_dirs() -> List[Path]:
    """Return likely data dirs in priority order (project root first, then backend/)."""
    here = Path(__file__).resolve().parent   # backend/
    root = here.parent                        # project root
    return [root / "data", here / "data"]

def _choose_data_dir() -> Path:
    """
    Pick the data dir:
      1) If it already has index+meta, choose that one.
      2) Else the first directory that exists.
      3) Else default to <project>/data.
    """
    candidates = _candidate_data_dirs()
    for d in candidates:
        if (d / "sbert_index.npz").exists() and (d / "sbert_meta.json").exists():
            return d
    for d in candidates:
        if d.exists():
            return d
    return candidates[0]

def _index_paths() -> Tuple[Path, Path, Path]:
    """Return paths to the index npz, meta json, and optional exemplar CSV."""
    d = _choose_data_dir()
    return (d / "sbert_index.npz", d / "sbert_meta.json", d / "exemplar_paraphrases.csv")

def _load_model() -> SentenceTransformer:
    """Load the sentence-transformer (once)."""
    global _MODEL
    if _MODEL is None:
        model_name = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

def _ensure_index_loaded() -> None:
    """
    Prepare the exemplar index in memory (only once):
      1) Prefer existing: sbert_index.npz + sbert_meta.json
      2) Else, build from exemplar_paraphrases.csv if present
      3) Else, build from _OFFLINE_SEED
    """
    global _X, _META
    if _X is not None and _META is not None:
        return

    index_npz, meta_json, csv_path = _index_paths()

    # 1) Prebuilt index + meta (fast path)
    if index_npz.exists() and meta_json.exists():
        _X = np.load(index_npz)["X"]
        with open(meta_json, "r", encoding="utf-8") as f:
            _META = json.load(f)
        return

    # 2) Build from CSV if available
    exemplars: List[Dict] = []
    if csv_path.exists():
        exemplars = _read_exemplars(csv_path)

    # 3) If CSV missing/empty, fall back to a small built-in seed
    if not exemplars:
        exemplars = list(_OFFLINE_SEED)

    model = _load_model()
    texts = [e["text"] for e in exemplars]
    # Normalise embeddings here so cosine reduces to dot product later
    embs = model.encode(texts, normalize_embeddings=True)  # list of (D,)
    X = np.asarray(embs, dtype=np.float32)

    # Save artifacts so we don't rebuild next time
    index_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(index_npz, X=X)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(exemplars, f, ensure_ascii=False, indent=2)

    _X, _META = X, exemplars

# ----------------------------
# CSV helpers
# ----------------------------
def _read_exemplars(csv_path: Path) -> List[Dict]:
    """
    Read exemplar_paraphrases.csv and return a clean list:
      [{"text": "...", "risk": "LOW|MEDIUM|HIGH"}, ...]
    Flexible headers supported:
      text: one of ["text", "exemplar", "phrase"]
      risk: one of ["risk", "severity", "label"]
    """
    text_keys = ("text", "exemplar", "phrase")
    risk_keys = ("risk", "severity", "label")

    out: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        if not rdr.fieldnames:
            return out
        for row in rdr:
            t_key = _first_present_key(row, text_keys)
            r_key = _first_present_key(row, risk_keys)
            t = (row.get(t_key) or "").strip()
            r = (row.get(r_key) or "LOW").strip().upper()
            if not t:
                continue
            if r not in ("LOW", "MEDIUM", "HIGH"):
                r = "LOW"
            out.append({"text": t, "risk": r})
    return out

def _first_present_key(row: Dict, candidates: Tuple[str, ...]) -> str:
    """Pick the first header name that exists in this CSV row."""
    for k in candidates:
        if k in row:
            return k
    return candidates[0]

# ----------------------------
# Cosine similarity
# ----------------------------
def _cosine_query_to_matrix(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Cosine similarity between a query vector u (D,) and matrix V (N, D).
    We normalise u here; rows of V were already normalised when the index was built.
    """
    u = u / (np.linalg.norm(u) + 1e-12)
    return (V @ u).astype(float)

# ----------------------------
# Public API used by app.py
# ----------------------------
def bert_symptom_score(text: str) -> Dict:
    """
    Return the nearest exemplar decision for a free-text symptom description.

    Output:
      {
        "risk": "LOW|MEDIUM|HIGH",      # risk label taken from the nearest exemplar
        "matched_reference": "<text>",  # the exemplar sentence we matched to
        "similarity_score": <0..1>      # cosine similarity (higher = closer)
      }
    """
    _ensure_index_loaded()
    model = _load_model()

    # Encode one query sentence
    emb = model.encode([text], normalize_embeddings=True)[0]     # (D,)
    # Compare to all exemplar embeddings
    sims = _cosine_query_to_matrix(emb, _X)                      # type: ignore[arg-type]
    i = int(np.argmax(sims))                                     # best match row index
    best = _META[i] if (isinstance(_META, list) and 0 <= i < len(_META)) else {}

    return {
        "risk": best.get("risk", "LOW"),
        "matched_reference": best.get("text", ""),
        "similarity_score": float(sims[i]),
    }
