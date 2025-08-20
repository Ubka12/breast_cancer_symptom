# build_embeddings.py
import json, numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

CSV  = "data/unified.csv"
EMB  = "data/sbert_index.npz"
META = "data/sbert_meta.json"

def main():
    df = pd.read_csv(CSV)
    texts = df["text"].astype(str).tolist()

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = model.encode(texts, normalize_embeddings=True)

    np.savez(EMB, X=X)
    with open(META, "w", encoding="utf-8") as f:
        json.dump(df[["text","concept","risk","source"]].to_dict("records"), f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(texts)} embeddings → {EMB} and meta → {META}")

if __name__ == "__main__":
    main()
