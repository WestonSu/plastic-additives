# -*- coding: utf-8 -*-
"""Isolated Windows-safe runner for the original ChemWalker implementation.

This worker does not reimplement ChemWalker. It imports and calls the installed
``chemwalker.rwalker.cand_pair`` and ``chemwalker.rwalker.random_walk`` functions.
It is launched as a separate Python process so the multiprocessing pool created
inside the published ``cand_pair(..., parallel=True)`` code is protected by a
real ``if __name__ == '__main__'`` boundary on Windows/Jupyter.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import inspect
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import traceback
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

WORKER_ID = "official_chemwalker_subprocess"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callable_audit(func) -> Dict[str, object]:
    source_file = inspect.getsourcefile(func)
    source_path = Path(source_file).resolve() if source_file else None
    try:
        source_text = inspect.getsource(func)
        callable_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    except Exception:
        callable_sha256 = ""
    return {
        "module": getattr(func, "__module__", ""),
        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", "")),
        "signature": str(inspect.signature(func)),
        "source_file": str(source_path) if source_path else "",
        "source_file_sha256": _sha256_file(source_path) if source_path and source_path.exists() else "",
        "callable_source_sha256": callable_sha256,
    }


def _normalize_candidate_edges(raw_edges) -> List[Tuple[str, str, float]]:
    """Normalize candidate-edge output returned by supported ChemWalker releases."""
    if raw_edges is None:
        return []
    if isinstance(raw_edges, pd.DataFrame):
        if raw_edges.empty:
            return []
        columns = list(raw_edges.columns)
        if {"source", "target", "weight"}.issubset(columns):
            values = raw_edges[["source", "target", "weight"]].itertuples(index=False, name=None)
        elif raw_edges.shape[1] >= 3:
            values = raw_edges.iloc[:, :3].itertuples(index=False, name=None)
        else:
            return []
        return [(str(source), str(target), float(weight)) for source, target, weight in values]

    normalized = []
    for edge in raw_edges:
        if len(edge) >= 3:
            normalized.append((str(edge[0]), str(edge[1]), float(edge[2])))
    return normalized


def _score_from_probabilities(
    tlid: pd.DataFrame,
    probabilities,
    graph: nx.Graph,
) -> pd.DataFrame:
    """Map random-walk probabilities to candidates and calculate dense ranks."""
    if isinstance(probabilities, dict):
        probability_map = {str(key): float(value) for key, value in probabilities.items()}
    else:
        graph_nodes = list(graph.nodes())
        probability_map = {
            str(node): float(value)
            for node, value in zip(graph_nodes, np.asarray(probabilities, dtype=float))
        }

    scored = tlid.copy()
    scored["uid"] = scored["uid"].astype(str)
    scored["network_score"] = scored["uid"].map(probability_map).fillna(0.0)
    group_max = scored.groupby("cluster index")["network_score"].transform("max")
    scored["network_reachable"] = group_max > 0
    scored["network_score_norm"] = np.where(
        group_max > 0,
        scored["network_score"] / group_max,
        0.0,
    )
    # NAP evaluation treats tied scores as sharing the same position (dense rank).
    scored["network_rank"] = scored.groupby("cluster index")[
        "network_score_norm"
    ].rank(ascending=False, method="dense")
    scored.loc[~scored["network_reachable"], "network_rank"] = np.nan
    scored["node_id"] = scored["cluster index"].astype(int)
    scored.attrs.clear()
    return scored


def run_worker(input_path: Path, output_path: Path) -> None:
    """Execute the original ChemWalker candidate graph and random walk safely."""
    started = time.time()
    with input_path.open("rb") as handle:
        payload = pickle.load(handle)

    # Import inside the protected worker process. These are the original package
    # functions; no local edge-weight or random-walk implementation is used.
    import chemwalker.rwalker as rwalker

    cand_pair = rwalker.cand_pair
    random_walk = rwalker.random_walk

    network = payload["network"].copy()
    tlid = payload["tlid"].copy()
    seed_uids = [str(value) for value in payload["seed_uids"]]
    fingerprint_method = str(payload["fingerprint_method"])
    ncores = int(payload["ncores"])
    restart_probability = float(payload["restart_probability"])

    if ncores < 1:
        raise ValueError("ncores must be >= 1 for the official parallel cand_pair path.")

    print(f"Worker identifier: {WORKER_ID}", flush=True)
    print(f"Python: {sys.executable}", flush=True)
    print(f"ChemWalker source: {inspect.getsourcefile(cand_pair)}", flush=True)
    print(
        "Calling original chemwalker.rwalker.cand_pair "
        f"with parallel=True, ncors={ncores}, fingerprint={fingerprint_method}",
        flush=True,
    )
    print(
        f"Spectral edges={len(network):,}; tlid rows={len(tlid):,}; "
        f"seed structures={len(seed_uids):,}; estimated candidate pairs="
        f"{int(payload.get('estimated_candidate_pairs', 0)):,}",
        flush=True,
    )

    edge_started = time.time()
    raw_edges = cand_pair(
        network,
        tlid,
        fingerprint_method,
        parallel=True,
        meansc=True,
        ncors=ncores,
    )
    candidate_edges = _normalize_candidate_edges(raw_edges)
    edge_seconds = time.time() - edge_started
    print(
        f"Original cand_pair completed: candidate edges={len(candidate_edges):,}; "
        f"elapsed={edge_seconds:.1f} s",
        flush=True,
    )
    if not candidate_edges:
        raise RuntimeError("Original ChemWalker cand_pair returned no candidate edges.")

    graph = nx.Graph()
    graph.add_weighted_edges_from(candidate_edges)
    valid_seed_uids = [uid for uid in seed_uids if uid in graph]
    missing_seed_uids = [uid for uid in seed_uids if uid not in graph]
    if not valid_seed_uids:
        raise RuntimeError(
            "None of the spectral-library seed structures are present in the "
            "ChemWalker candidate graph."
        )

    walk_started = time.time()
    probabilities = random_walk(
        graph,
        valid_seed_uids,
        restart_prob=restart_probability,
    )
    walk_seconds = time.time() - walk_started
    scored = _score_from_probabilities(tlid, probabilities, graph)
    total_seconds = time.time() - started

    audit = {
        "worker_id": WORKER_ID,
        "cand_pair": _callable_audit(cand_pair),
        "random_walk": _callable_audit(random_walk),
        "fingerprint_method": fingerprint_method,
        "ncores": ncores,
        "restart_probability": restart_probability,
        "n_spectral_edges": int(len(network)),
        "n_tlid_rows": int(len(tlid)),
        "estimated_candidate_pairs": int(payload.get("estimated_candidate_pairs", 0)),
        "n_candidate_edges": int(len(candidate_edges)),
        "n_candidate_graph_nodes": int(graph.number_of_nodes()),
        "n_valid_seed_uids": int(len(valid_seed_uids)),
        "n_missing_seed_uids": int(len(missing_seed_uids)),
        "missing_seed_uids": missing_seed_uids,
        "cand_pair_seconds": float(edge_seconds),
        "random_walk_seconds": float(walk_seconds),
        "total_seconds": float(total_seconds),
    }

    result = {"scored": scored, "audit": audit}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    print(f"Worker output saved: {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the installed ChemWalker implementation in a protected subprocess."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    mp.freeze_support()
    try:
        run_worker(args.input, args.output)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
