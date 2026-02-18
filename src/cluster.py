from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.utils.io import ensure_dir, load_ids, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster specimen embeddings")
    parser.add_argument("--emb", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--out", type=Path, required=True, help="Output results directory")
    parser.add_argument("--ids", type=Path, default=None, help="Optional ids.txt path")
    parser.add_argument("--method", choices=["hdbscan", "kmeans"], default="hdbscan")
    parser.add_argument("--pca", type=int, default=0, help="Use PCA with specified n_components if >0")
    parser.add_argument("--k", type=int, default=20, help="K for kmeans")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def infer_ids(emb_path: Path, n: int, ids_path: Path | None) -> list[str]:
    if ids_path is not None and ids_path.exists():
        ids = load_ids(ids_path)
    else:
        candidate = emb_path.parent / "ids.txt"
        ids = load_ids(candidate) if candidate.exists() else [f"{i:03d}" for i in range(n)]
    if len(ids) != n:
        raise ValueError(f"ids count ({len(ids)}) does not match embeddings ({n})")
    return ids


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    X = np.load(args.emb)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N,D], got {X.shape}")

    ids = infer_ids(args.emb, X.shape[0], args.ids)

    if args.pca and args.pca > 0:
        n_comp = min(args.pca, X.shape[0], X.shape[1])
        LOGGER.info("Applying PCA n_components=%d", n_comp)
        X = PCA(n_components=n_comp, random_state=args.seed).fit_transform(X)

    if args.method == "hdbscan":
        import hdbscan

        clusterer = hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True)
        labels = clusterer.fit_predict(X)
        scores = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))
        score_col = "prob"
    else:
        clusterer = KMeans(n_clusters=args.k, random_state=args.seed, n_init="auto")
        labels = clusterer.fit_predict(X)
        dists = clusterer.transform(X)
        scores = dists.min(axis=1)
        score_col = "score"

    df = pd.DataFrame({"specimen_id": ids, "cluster_id": labels, score_col: scores})
    out_csv = args.out / "clusters.csv"
    df.to_csv(out_csv, index=False)
    LOGGER.info("Saved clustering results to %s", out_csv)


if __name__ == "__main__":
    main()
