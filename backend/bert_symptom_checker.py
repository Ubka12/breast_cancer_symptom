# backend/bert_symptom_checker.py
from __future__ import annotations
from pathlib import Path
import os, json, csv
import numpy as np

# sentence-transformers is the only external dep for this module
from sentence_transformers import SentenceTransformer

# Globals (lazy-loaded)
_MODEL: SentenceTransformer | None = None
_X: np.ndarray | None = None           # (N, D) float32, unit-normalised
_META: list[dict] | None = None        # length N; each has {"text":..., "risk":...}

# ----------------------------
# Paths & loading
# ----------------------------
def _data_dir() -> Path:
    """Return the project data directory (…/data)."""
    here = Path(__file__).resolve().parent       # backend/
    root = here.parent                            # project root
    d = root / "data"
    if not d.exists():
        # also support backend/data as a fallback
        d2 = here / "data"
        return d2 if d2.exists() else d
    return d

def _index_paths() -> tuple[Path, Path, Path]:
    """Return paths for index.npz, meta.json, and exemplars CSV."""
    d = _data_dir()
    return d / "sbert_index.npz", d / "sbert_meta.json", d / "exemplar_paraphrases.csv"

def _load_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        model_name = os.getenv("SBERT_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

def _ensure_index_loaded():
    """Load (or build) the SBERT index + meta into globals."""
    global _X, _META
    if _X is not None and _META is not None:
        return

    index_npz, meta_json, csv_path = _index_paths()

    if index_npz.exists() and meta_json.exists():
        # Fast path: load prebuilt
        _X = np.load(index_npz)["X"]
        with open(meta_json, "r", encoding="utf-8") as f:
            _META = json.load(f)
        return

    # Fallback: build from CSV exemplars, then save both files
    if not csv_path.exists():
        raise RuntimeError(
            f"SBERT index not found and exemplar CSV missing:\n  {index_npz}\n  {meta_json}\n  {csv_path}"
        )

    exemplars = _read_exemplars(csv_path)
    if not exemplars:
        raise RuntimeError(f"No exemplars found in {csv_path}")

    model = _load_model()
    texts = [e["text"] for e in exemplars]
    embs = model.encode(texts, normalize_embeddings=True)  # (N, D) already L2-normalised
    X = np.asarray(embs, dtype=np.float32)

    # Save for reuse
    index_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(index_npz, X=X)
    with open(meta_json, "w", encoding="utf-8") as f:
        json.dump(exemplars, f, ensure_ascii=False, indent=2)

    _X, _META = X, exemplars

# ----------------------------
# CSV helpers
# ----------------------------
def _read_exemplars(csv_path: Path) -> list[dict]:
    """
    Read exemplar_paraphrases.csv and return a list of dicts with at least:
      {"text": "...", "risk": "LOW|MEDIUM|HIGH"}
    Accepts flexible column names:
      text: one of ["text", "exemplar", "phrase"]
      risk: one of ["risk", "severity", "label"]
    """
    text_keys = ("text", "exemplar", "phrase")
    risk_keys = ("risk", "severity", "label")

    exemplars: list[dict] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            # find columns
            t = _first_key(row, text_keys)
            r = _first_key(row, risk_keys)
            text = (row.get(t) or "").strip()
            risk = (row.get(r) or "LOW").strip().upper()
            if not text:
                continue
            if risk not in ("LOW", "MEDIUM", "HIGH"):
                risk = "LOW"
            exemplars.append({"text": text, "risk": risk})
    return exemplars

def _first_key(row: dict, keys: tuple[str, ...]) -> str:
    for k in keys:
        if k in row:
            return k
    # default to first key if none match
    return keys[0]

# ----------------------------
# Cosine & scoring
# ----------------------------
def _cosine_query_to_matrix(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    V is (N, D) unit-normalised; u is (D,) unit-normalised.
    Returns cosine similarities (N,).
    """
    u = u / (np.linalg.norm(u) + 1e-12)
    return (V @ u).astype(float)

# ----------------------------
# Public API
# ----------------------------
def bert_symptom_score(text: str) -> dict:
    """
    Return SBERT similarity decision for free-text symptoms.
    Output:
      {
        "risk": "LOW|MEDIUM|HIGH",
        "matched_reference": "<nearest exemplar text>",
        "similarity_score": <float in [0,1]>
      }
    """
    _ensure_index_loaded()
    model = _load_model()

    emb = model.encode([text], normalize_embeddings=True)[0]  # (D,)
    sims = _cosine_query_to_matrix(emb, _X)                   # type: ignore[arg-type]
    idx = int(np.argmax(sims))
    best = _META[idx]                                         # type: ignore[index]

    return {
        "risk": best.get("risk", "LOW"),
        "matched_reference": best.get("text", ""),
        "similarity_score": float(sims[idx]),
    }
