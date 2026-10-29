"""Encode LLM profiles and residual-quantize them into Semantic IDs."""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sid.rq_kmeans import RQKMeans  # noqa: E402


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hash_embed(texts, dim=256):
    """Deterministic fallback embedding so the pipeline can run without an API."""
    import hashlib
    vectors = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        seed = int(hashlib.md5(str(text).encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.normal(size=dim).astype(np.float32)
        nrm = np.linalg.norm(vec) + 1e-8
        vectors[i] = vec / nrm
    return vectors


def embed_profiles(texts, use_hash=False):
    if use_hash:
        print("Using hashed fallback embeddings (--hash_embed).")
        return hash_embed(texts)
    from sid.llm_client import embed_texts

    return np.asarray(embed_texts(texts), dtype=np.float32)


def quantize(ids, embeddings, n_levels, n_clusters, out_path):
    rq = RQKMeans(n_levels=n_levels, n_clusters=n_clusters)
    codes = rq.fit(embeddings)
    mapping = {key: codes[i].tolist() for i, key in enumerate(ids)}
    payload = {
        "n_levels": n_levels,
        "n_clusters": n_clusters,
        "sids": mapping,
        "codebooks": [c.tolist() for c in rq.codebooks],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print("wrote", out_path, "num:", len(mapping))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--n_clusters", type=int, default=256)
    parser.add_argument("--hash_embed", action="store_true")
    args = parser.parse_args()

    items = load_jsonl(os.path.join(args.data_dir, "item_profiles.jsonl"))
    users = load_jsonl(os.path.join(args.data_dir, "user_profiles.jsonl"))

    item_ids = [row["parent_asin"] for row in items]
    user_ids = [row["user_id"] for row in users]
    item_emb = embed_profiles([row["profile"] for row in tqdm(items, desc="item embed")], args.hash_embed)
    user_emb = embed_profiles([row["profile"] for row in tqdm(users, desc="user embed")], args.hash_embed)

    np.save(os.path.join(args.data_dir, "item_emb.npy"), item_emb)
    np.save(os.path.join(args.data_dir, "user_emb.npy"), user_emb)

    quantize(item_ids, item_emb, args.n_levels, args.n_clusters, os.path.join(args.data_dir, "item_sids.json"))
    quantize(user_ids, user_emb, args.n_levels, args.n_clusters, os.path.join(args.data_dir, "user_sids.json"))


if __name__ == "__main__":
    main()
