from pathlib import Path
import os, json, numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None
_X = None
_META = None

def _find_data_dir():
    here = Path(__file__).resolve().parent           # backend/
    root = here.parent                                # project root
    candidates = [root / "data", here / "data"]       # try both
    for d in candidates:
        if (d / "sbert_index.npz").exists() and (d / "sbert_meta.json").exists():
            return d
    raise RuntimeError(
        "SBERT index not found. Looked in:\n  " +
        "\n  ".join(str(c) for c in candidates)
    )

def _lazy_load():
    global _MODEL, _X, _META
    if _MODEL is None:
        _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    if _X is None or _META is None:
        DATA = _find_data_dir()
        INDEX = DATA / "sbert_index.npz"
        META  = DATA / "sbert_meta.json"
        _X = np.load(INDEX)["X"]
        with open(META, "r", encoding="utf-8") as f:
            _META = json.load(f)

def _cosine(u, V):
    u = u / (np.linalg.norm(u) + 1e-12)
    return (V @ u).astype(float)

def bert_symptom_score(text: str):
    _lazy_load()
    emb = _MODEL.encode([text], normalize_embeddings=True)[0]
    sims = _cosine(emb, _X)
    i = int(np.argmax(sims))
    best = _META[i]
    return {
        "risk": best.get("risk", "LOW"),
        "matched_reference": best.get("text", ""),
        "similarity_score": float(sims[i]),
    }
