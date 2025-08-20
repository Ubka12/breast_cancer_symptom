# backend/build_unified_dataset.py
"""
Build a unified text dataset for SBERT.

Primary sources (if online):
 - Reddit (lay narratives)
 - NHS (symptom descriptions)

Offline fallback:
 - data/symptom_sentences.json  (if present; list[str] or [{"text":...}])
 - A small built-in seed list

Writes:
  data/unified_dataset.json
  data/unified_dataset_filtered.json
"""
from __future__ import annotations

import os, json, re
from typing import List

# Try package-mode first, then script-mode
try:
    from . import reddit_fetcher, nhs_scraper  # type: ignore
except Exception:
    import reddit_fetcher  # type: ignore
    import nhs_scraper    # type: ignore

DATA_DIR   = "data"
RAW_PATH   = os.path.join(DATA_DIR, "unified_dataset.json")
CLEAN_PATH = os.path.join(DATA_DIR, "unified_dataset_filtered.json")
LOCAL_JSON = os.path.join(DATA_DIR, "symptom_sentences.json")

_WS = re.compile(r"\s+")

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

def _norm(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())

def _dedupe_keep(texts: List[str]) -> List[str]:
    out, seen = [], set()
    for t in texts:
        tt = _norm(t)
        if len(tt) < 10:
            continue
        key = tt.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tt)
    return out

def _try_call(mod, names: List[str], *args, **kwargs) -> List[str]:
    for nm in names:
        fn = getattr(mod, nm, None)
        if callable(fn):
            try:
                res = fn(*args, **kwargs)
                if isinstance(res, list):
                    return [str(x) for x in res]
            except Exception:
                pass
    return []

def _read_local_json(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            # Accept list[str] or list[dict]
            out = []
            for row in data:
                if isinstance(row, dict):
                    out.append(str(row.get("text", "")))
                else:
                    out.append(str(row))
            return out
    except Exception:
        pass
    return []

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # Online path
    print("Fetching Reddit…")
    r = _try_call(reddit_fetcher, ["fetch_reddit_posts", "get_reddit_posts", "fetch_posts"], "breastcancer", 500)
    print(f"  Reddit: {len(r)} items")

    print("Scraping NHS…")
    n = _try_call(nhs_scraper, ["scrape_nhs_symptoms", "get_symptom_texts", "scrape"], "https://111.wales.nhs.uk/Cancerofthebreast,female/")
    print(f"  NHS:    {len(n)} items")

    texts = r + n

    # Offline fallbacks if needed
    if not texts:
        print("No online data found. Trying local data/symptom_sentences.json …")
        local = _read_local_json(LOCAL_JSON)
        print(f"  Local file: {len(local)} items")
        texts = local

    if not texts:
        print("Still empty. Using small built-in seed list.")
        texts = OFFLINE_SEED[:]

    texts = _dedupe_keep(texts)
    raw = [{"text": t} for t in texts]

    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"Wrote {RAW_PATH} ({len(raw)} rows)")

    # filtered == deduped already, but keep separate file name for clarity
    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)
    print(f"Wrote {CLEAN_PATH} ({len(raw)} unique rows)")
    print("Done.")

if __name__ == "__main__":
    main()
