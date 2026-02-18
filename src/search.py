from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.utils.io import ensure_dir, load_ids, set_seed, setup_logging
from src.utils.vision import l2_normalize

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="k-NN search over specimen embeddings")
    parser.add_argument("--emb", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--ids", type=Path, required=True, help="Path to ids.txt")
    parser.add_argument("--query", type=str, required=True, help="Query specimen id")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--metric", choices=["cosine", "l2"], default="cosine")
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    X = np.load(args.emb)
    ids = load_ids(args.ids)
    if len(ids) != X.shape[0]:
        raise ValueError("ids and embedding rows mismatch")

    metric = "euclidean" if args.metric == "l2" else "cosine"
    if metric == "cosine":
        X_work = l2_normalize(X.astype(np.float32))
    else:
        X_work = X.astype(np.float32)

    try:
        q_idx = ids.index(args.query)
    except ValueError as e:
        raise ValueError(f"query id not found: {args.query}") from e

    nn = NearestNeighbors(metric=metric)
    nn.fit(X_work)

    k = min(args.topk + 1, len(ids))
    distances, indices = nn.kneighbors(X_work[q_idx : q_idx + 1], n_neighbors=k)

    rows = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == q_idx:
            continue
        rows.append({"query_id": args.query, "neighbor_id": ids[idx], "distance": float(dist)})
        if len(rows) >= args.topk:
            break

    df = pd.DataFrame(rows)
    out_csv = args.out / f"knn_{args.query}.csv"
    df.to_csv(out_csv, index=False)

    LOGGER.info("Top-%d neighbors for %s", len(df), args.query)
    for _, row in df.iterrows():
        LOGGER.info("neighbor=%s distance=%.6f", row["neighbor_id"], row["distance"])
    LOGGER.info("Saved %s", out_csv)


if __name__ == "__main__":
    main()
