from __future__ import annotations

import argparse
import itertools
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.cluster import infer_ids, preprocess_for_clustering, run_hdbscan
from src.utils.io import ensure_dir, set_seed, setup_logging

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep HDBSCAN parameters for lower noise ratio")
    parser.add_argument("--emb", type=Path, required=True, help="Path to embeddings.npy")
    parser.add_argument("--ids", type=Path, required=True, help="Path to ids.txt")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for sweep results")
    parser.add_argument("--labels", type=Path, default=None, help="Optional labels file")
    parser.add_argument("--normalize", choices=["none", "l2"], default="l2")
    parser.add_argument("--metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--umap_metric", choices=["cosine", "euclidean"], default="cosine")
    parser.add_argument("--pca_values", type=float, nargs="+", default=[0, 64, 0.95])
    parser.add_argument("--min_cluster_sizes", type=int, nargs="+", default=[5, 10, 20, 40])
    parser.add_argument("--min_samples_values", type=int, nargs="+", default=[1, 2, 5])
    parser.add_argument("--selection_methods", choices=["leaf", "eom"], nargs="+", default=["leaf", "eom"])
    parser.add_argument("--umap_options", choices=["off", "on"], nargs="+", default=["off", "on"])
    parser.add_argument("--umap_n_components", type=int, default=15)
    parser.add_argument("--umap_n_neighbors", type=int, default=30)
    parser.add_argument("--umap_min_dist", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_labels(path: Path | None, ids: list[str]) -> np.ndarray | None:
    if path is None or not path.exists():
        return None

    rows = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return None

    if "\t" in rows[0] or "," in rows[0]:
        sep = "\t" if "\t" in rows[0] else ","
        mapping: dict[str, str] = {}
        for row in rows:
            cols = row.split(sep)
            if len(cols) < 2:
                continue
            mapping[cols[0]] = cols[1]
        return np.array([mapping.get(sid, "") for sid in ids], dtype=object)

    if len(rows) != len(ids):
        LOGGER.warning("labels count mismatch (%d vs %d). labels will be skipped", len(rows), len(ids))
        return None
    return np.array(rows, dtype=object)


def purity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_pred != -1
    if np.sum(mask) == 0:
        return 0.0

    y_true = y_true[mask]
    y_pred = y_pred[mask]
    labels = np.unique(y_pred)
    total = len(y_true)
    correct = 0
    for c in labels:
        idx = y_pred == c
        members = y_true[idx]
        if len(members) == 0:
            continue
        values, counts = np.unique(members, return_counts=True)
        correct += int(counts[np.argmax(counts)])
    return float(correct / total if total else 0.0)


def evaluate_config(X: np.ndarray, labels: np.ndarray, probs: np.ndarray, y_true: np.ndarray | None) -> dict[str, float]:
    n_total = len(labels)
    n_noise = int(np.sum(labels == -1))
    n_clusters = int(len(set(labels.tolist()) - {-1}))
    noise_ratio = float(n_noise / n_total if n_total else 1.0)
    mean_prob = float(np.mean(probs)) if len(probs) else 0.0

    penalty = 0.0
    if n_clusters <= 1:
        penalty += 0.3
    if n_clusters > max(2, int(np.sqrt(max(n_total, 1))) * 3):
        penalty += 0.2

    metrics: dict[str, float] = {
        "noise_ratio": noise_ratio,
        "n_clusters": float(n_clusters),
        "mean_prob": mean_prob,
        "penalty": penalty,
        "score": (1.0 - noise_ratio) + 0.2 * mean_prob - penalty,
    }

    try:
        import hdbscan

        metrics["dbcv"] = float(hdbscan.validity_index(X, labels))
    except Exception:
        metrics["dbcv"] = float("nan")

    if y_true is not None:
        mask = (labels != -1) & (y_true != "")
        if np.sum(mask) > 1:
            metrics["ari"] = float(adjusted_rand_score(y_true[mask], labels[mask]))
            metrics["nmi"] = float(normalized_mutual_info_score(y_true[mask], labels[mask]))
            metrics["purity"] = float(purity_score(y_true[mask], labels[mask]))
        else:
            metrics["ari"] = float("nan")
            metrics["nmi"] = float("nan")
            metrics["purity"] = float("nan")
    return metrics


def run_single(
    X: np.ndarray,
    ids: list[str],
    *,
    normalize: str,
    metric: str,
    pca: float,
    min_cluster_size: int,
    min_samples: int,
    selection_method: str,
    use_umap: bool,
    umap_n_components: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
    umap_metric: str,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    pca_to_use = pca
    if use_umap and pca_to_use <= 0:
        pca_to_use = 50

    X_proc, _ = preprocess_for_clustering(
        X,
        normalize=normalize,
        pca=pca_to_use,
        pca_report=None,
        use_umap=use_umap,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        seed=seed,
    )

    labels, probs = run_hdbscan(
        X_proc,
        metric=metric,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=selection_method,
        allow_single_cluster=False,
        cluster_selection_epsilon=None,
    )

    df = pd.DataFrame({"specimen_id": ids, "cluster_id": labels, "prob": probs})
    return df, labels, probs, X_proc


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    ensure_dir(args.out)

    X = np.load(args.emb)
    if X.ndim != 2:
        raise ValueError(f"embeddings must be 2D [N,D], got {X.shape}")
    ids = infer_ids(args.emb, X.shape[0], args.ids)
    y_true = load_labels(args.labels, ids)

    rows: list[dict[str, Any]] = []
    best_score = -np.inf
    best_row: dict[str, Any] | None = None
    best_df: pd.DataFrame | None = None

    grid = itertools.product(
        args.pca_values,
        args.min_cluster_sizes,
        args.min_samples_values,
        args.selection_methods,
        args.umap_options,
    )

    for pca, min_cluster_size, min_samples, selection_method, umap_opt in grid:
        use_umap = umap_opt == "on"
        config = {
            "pca": pca,
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "selection_method": selection_method,
            "umap": use_umap,
        }
        try:
            df, labels, probs, X_used = run_single(
                X,
                ids,
                normalize=args.normalize,
                metric=args.metric,
                pca=pca,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                selection_method=selection_method,
                use_umap=use_umap,
                umap_n_components=args.umap_n_components,
                umap_n_neighbors=args.umap_n_neighbors,
                umap_min_dist=args.umap_min_dist,
                umap_metric=args.umap_metric,
                seed=args.seed,
            )
            metrics = evaluate_config(X_used, labels, probs, y_true)
        except Exception:
            LOGGER.exception("Config failed: %s", config)
            metrics = {
                "noise_ratio": 1.0,
                "n_clusters": 0.0,
                "mean_prob": 0.0,
                "penalty": 1.0,
                "score": -1.0,
                "dbcv": float("nan"),
            }
            df = pd.DataFrame({"specimen_id": ids, "cluster_id": np.full(len(ids), -1), "prob": np.zeros(len(ids))})

        row = {**config, **metrics}
        rows.append(row)

        if metrics["score"] > best_score:
            best_score = float(metrics["score"])
            best_row = row
            best_df = df

    result_df = pd.DataFrame(rows).sort_values("score", ascending=False)
    sweep_csv = args.out / "sweep_results.csv"
    result_df.to_csv(sweep_csv, index=False)
    LOGGER.info("Saved sweep results to %s", sweep_csv)

    if best_row is None or best_df is None:
        raise RuntimeError("No valid sweep result was produced")

    best_yaml = args.out / "best_config.yaml"
    best_yaml.write_text(yaml.safe_dump(best_row, sort_keys=False, allow_unicode=True), encoding="utf-8")
    LOGGER.info("Saved best config to %s", best_yaml)

    best_csv = args.out / "best_clusters.csv"
    best_df.to_csv(best_csv, index=False)
    LOGGER.info("Saved best clusters to %s", best_csv)

    summary_path = args.out / "summary.json"
    summary = {
        "n_trials": len(rows),
        "best_score": best_score,
        "best_config": best_row,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
