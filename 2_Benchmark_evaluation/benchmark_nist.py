# -*- coding: utf-8 -*-
"""
Reproducible benchmark of MetFrag, ChemWalker, and mNAP using the 555 NIST
tandem mass spectra selected for the published ChemWalker benchmark.

Required inputs
---------------
1. ``nist.msp``: an MSP file containing the benchmark spectra. Expected fields
   include Name, Precursor_type, Spectrum_type, PrecursorMZ, InChIKey, Formula,
   ExactMass, ID/NISTNO, Num peaks, and the peak list.
2. ``pcbi.1006089.s007.tsv``: the published benchmark table, used only to
   recover verified structures, cluster indices, and ground-truth identifiers
   for processing batch 1. Previously published method ranks are not imported.
3. A local PubChemLite candidate database.
4. The MetFrag command-line JAR.

Benchmark design
----------------
The workflow rebuilds the molecular network from the MSP peak lists, generates
one shared MetFrag candidate pool per spectrum, assigns approximately 10% seed
nodes within each connected component, and evaluates MetFrag, ChemWalker, and
mNAP on identical target nodes and candidate pools. The installed ChemWalker
``cand_pair`` and ``random_walk`` functions are used without reimplementation.

Recommended Jupyter workflow
----------------------------
    import benchmark_nist as bm
    bm.configure_paths(...)
    bm.run_step(0)
    bm.run_step(1, force=True)
    bm.run_step(2, force=True, max_nodes=5)   # validate MetFrag Score parsing
    bm.run_step(2, force=False)               # generate all candidate lists
    smoke_id = bm.choose_smoke_component()
    bm.run_step(3, repeat_ids=[0], component_ids=[smoke_id], force=True)
    bm.run_step(3, repeat_ids=[0], force=False)
    bm.run_step(3, force=False)
    outputs = bm.run_step(4)

MetFrag score validation
------------------------
Only the validated MetFrag ``Score`` field is used for baseline ranking. The
workflow never substitutes molecular mass, FragmenterScore, row number, mass
error, or reciprocal rank when the MetFrag Score cannot be identified.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger
from matchms import Spectrum

RDLogger.DisableLog("rdApp.warning")

try:
    from scipy.stats import binomtest, wilcoxon
except Exception:
    binomtest = None
    wilcoxon = None


# =========================================================
# 0. Paths and parameters
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

# Descriptive release and cache-schema identifiers are included in run
# signatures so that incompatible checkpoints are never reused silently.
RELEASE_ID = "published_benchmark"
NETWORK_CACHE_SCHEMA = "nist_network"
CANDIDATE_CACHE_SCHEMA = "nist_metfrag_candidates_2d"
SEED_ALLOCATION_SCHEME = "component_stratified_10_percent"
CHEMWALKER_EXECUTION_MODE = "official_parallel_subprocess"
CHEMWALKER_WORKER_SCRIPT = PROJECT_DIR / "chemwalker_official_worker.py"


class MetFragOutputValidationError(RuntimeError):
    """Raised when a MetFrag output field cannot be validated unambiguously."""


# Portable default locations. Use configure_paths() to override any path.
DATA_DIR = PROJECT_DIR / "data"
SOFTWARE_DIR = PROJECT_DIR / "software"
INPUT_MSP = DATA_DIR / "nist.msp"
BENCHMARK_TABLE = DATA_DIR / "pcbi.1006089.s007.tsv"
BENCHMARK_PROCESSING_BATCH = 1
PUBCHEMLITE_PATH = DATA_DIR / "PubChemLite.csv"
METFRAG_PATH = SOFTWARE_DIR / "MetFrag2.3-CL.jar"

OUT_DIR = PROJECT_DIR / "results" / "nist_benchmark"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
CACHE_DIR = OUT_DIR / "candidate_cache"
AUDIT_DIR = OUT_DIR / "metfrag_score_audit"
REPEAT_DIR = OUT_DIR / "repeat_results"
CHEMWALKER_WORK_DIR = OUT_DIR / "chemwalker_worker"
SMOKE_DIR = OUT_DIR / "smoke_tests"

for _path in [
    OUT_DIR, CHECKPOINT_DIR, CACHE_DIR, AUDIT_DIR, REPEAT_DIR,
    CHEMWALKER_WORK_DIR, SMOKE_DIR,
]:
    _path.mkdir(exist_ok=True, parents=True)

STEP1_CHECKPOINT = CHECKPOINT_DIR / "step1_network.pkl"
STEP2_CANDIDATES_PKL = CHECKPOINT_DIR / "step2_all_candidates.pkl"
STEP2_CANDIDATES_CSV = OUT_DIR / "all_metfrag_candidates.csv"


# -------------------------
# Benchmark scale
# -------------------------

# Set to None to evaluate all eligible unique 2D structures. Set to a
# positive integer only for deterministic subsampling or workflow testing.
N_COMPOUNDS_TO_SAMPLE: Optional[int] = None
MAX_SPECTRA_PER_COMPOUND = 1
RANDOM_SEED = 2026

# Following the ChemWalker benchmark, approximately 10% of nodes in each
# connected component are assigned as seeds; small components receive one seed.
SEED_FRACTION = 0.10

# Number of independent component-wise seed allocations used to assess the
# robustness of candidate re-ranking.
N_SEED_REPEATS = 10


# -------------------------
# MSP record filtering
# -------------------------

IONMODE_FILTER = "positive"       # MSP values P/POSITIVE are accepted
ADDUCT_FILTER = "[M+H]+"          # exact protonated precursor type
SPECTRUM_TYPE_FILTER = "MS2"
MIN_PEAKS = 3


# -------------------------
# Molecular network
# -------------------------

FRAGMENT_TOLERANCE = 0.01

# Retain molecular-network edges with modified cosine >= 0.60 and at least
# two matched product ions. Both thresholds are fixed for all three methods.
MIN_COSINE = 0.60
MIN_MATCHED_PEAKS = 2

# Mutual top-k limits overly dense components while preserving the network
# neighborhood logic used by molecular networking workflows.
TOPK = 10

MZ_POWER = 0.0
INTENSITY_POWER = 1.0
USE_HUNGARIAN = False

# Optional robustness output: edges in the 0.60 <= cosine < 0.70 range, as
# examined in the NAP study. This file is generated from the primary network;
# the full benchmark is not rerun automatically for this subset.
LOW_SIMILARITY_UPPER_COSINE = 0.70


# -------------------------
# Candidate generation
# -------------------------

USE_METFRAG = True
FALLBACK_TO_MASS_SCORE_IF_METFRAG_FAILS = False
PPM = 10
TOP_N_CANDIDATES_PER_NODE = 50
FORCE_INCLUDE_TRUE_CANDIDATE = False

# Candidate identity is evaluated at 2D connectivity level (first InChIKey
# block), consistent with many structure-ranking benchmarks.
DEDUPLICATE_CANDIDATES_BY_INCHIKEY1 = True

# Save detailed audits for the first successful MetFrag outputs to verify
# that the official Score field is parsed correctly.
DEBUG_METFRAG_SCORE = True
DEBUG_METFRAG_MAX_NODES = 5
DEBUG_METFRAG_ROWS = 5


# -------------------------
# ChemWalker and mNAP
# -------------------------

# The same pre-specified ChemWalker fingerprint is used for ChemWalker and
# mNAP. It is fixed before evaluation and is not selected on the test results.
FINGERPRINT_METHOD_FOR_CHEMWALKER = "MFP2-bits"

# The ChemWalker baseline calls the original installed cand_pair and
# random_walk functions. Each component is executed in a protected subprocess
# because the published parallel cand_pair path creates a multiprocessing pool.
CHEMWALKER_PARALLEL = True
CHEMWALKER_NCORES = 4
CHEMWALKER_HEARTBEAT_SECONDS = 30
CHEMWALKER_COMPONENT_TIMEOUT_SECONDS = 21600  # 6 h; set 0 for no timeout
FAIL_ON_METHOD_ERROR = True

RESTART_PROB = 0.1
CHEMWALKER_SIGMOID_A = -9.0
CHEMWALKER_SIGMOID_B = 0.60

# mNAP combines spectral similarity, structural similarity, and normalized
# MetFrag scores when constructing candidate-level edges. These parameters are
# fixed for every benchmark target and seed allocation.
MNAP_BETA = 0.5
MNAP_SCORE_EPS = 0.05
MNAP_EDGE_MODE = "sigmoid"        # "plain", "cosine", or "sigmoid"


# -------------------------
# Evaluation
# -------------------------

MAX_K_FOR_PLOT = 20
PRIMARY_TOP_K = 10
BOOTSTRAP_ITERATIONS = 1000

# When a method returns no rank for an otherwise rankable target, Top-k and MRR
# treat it as failure. Penalized rank uses n_candidates + 1.


# =========================================================
# 1. Imports from ChemWalker
# =========================================================

try:
    import chemwalker.rwalker as chemwalker_rwalker
    from chemwalker.rwalker import cand_pair, random_walk
except Exception as exc:
    raise ImportError(
        "Cannot import cand_pair/random_walk from chemwalker.rwalker. "
        "Install the same ChemWalker environment used for the benchmark."
    ) from exc

try:
    from chemwalker.utils import run_metfrag
except Exception:
    try:
        from chemwalker.rwalker import run_metfrag
    except Exception:
        run_metfrag = None
        warnings.warn(
            "Cannot import ChemWalker run_metfrag. USE_METFRAG=True will fail."
        )


# =========================================================
# 2. General helpers
# =========================================================


def show_df(df: pd.DataFrame, n: int = 5) -> None:
    try:
        display(df.head(n))  # type: ignore[name-defined]
    except Exception:
        print(df.head(n))


def clear_df_attrs(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.attrs.clear()
    return out


def clean_str(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().strip('"').strip()


def normalize_inchi(value) -> str:
    text = clean_str(value)
    if text == "" or text.lower() in {"nan", "none", "null", "n/a", "na"}:
        return ""
    return text if text.startswith("InChI=") else "InChI=" + text


def smiles_to_inchi(smiles) -> str:
    try:
        text = clean_str(smiles)
        if text == "":
            return ""
        mol = Chem.MolFromSmiles(text)
        if mol is None:
            return ""
        return Chem.MolToInchi(mol)
    except Exception:
        return ""


def inchi_to_inchikey(inchi) -> str:
    try:
        text = normalize_inchi(inchi)
        if not text.startswith("InChI="):
            return ""
        value = Chem.InchiToInchiKey(text)
        return "" if value is None else value
    except Exception:
        return ""


def inchikey1(value) -> str:
    text = clean_str(value)
    return "" if text == "" else text.split("-")[0]


def to_float(value, default=np.nan):
    try:
        if value is None:
            return default
        return float(str(value).split()[0])
    except Exception:
        return default


def meta_get(metadata, keys, default=""):
    if isinstance(keys, str):
        keys = [keys]
    if metadata is None:
        return default
    lower_map = {str(key).lower(): key for key in metadata.keys()}
    for key in keys:
        lower = str(key).lower()
        if lower in lower_map:
            value = metadata[lower_map[lower]]
            return default if value is None else value
    return default


def infer_neutral_mass(pepmass, adduct, exactmass=np.nan):
    pepmass = to_float(pepmass)
    if pd.isna(pepmass):
        return np.nan
    adduct_text = clean_str(adduct)
    proton = 1.007276466812
    if adduct_text in {"[M+H]+", "M+H", "[M+H]", "M+H+"}:
        return pepmass - proton
    return np.nan


def get_spectrum_mz_intensity(spectrum: Spectrum) -> Tuple[np.ndarray, np.ndarray]:
    try:
        return (
            np.asarray(spectrum.peaks.mz, dtype=float),
            np.asarray(spectrum.peaks.intensities, dtype=float),
        )
    except Exception:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)


def stable_seed(*parts) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32 - 1)


def save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


def json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def config_snapshot() -> Dict[str, object]:
    keys = [
        "RELEASE_ID",
        "NETWORK_CACHE_SCHEMA",
        "CANDIDATE_CACHE_SCHEMA",
        "SEED_ALLOCATION_SCHEME",
        "N_COMPOUNDS_TO_SAMPLE",
        "MAX_SPECTRA_PER_COMPOUND",
        "RANDOM_SEED",
        "SEED_FRACTION",
        "N_SEED_REPEATS",
        "IONMODE_FILTER",
        "ADDUCT_FILTER",
        "SPECTRUM_TYPE_FILTER",
        "BENCHMARK_PROCESSING_BATCH",
        "MIN_PEAKS",
        "FRAGMENT_TOLERANCE",
        "MIN_COSINE",
        "MIN_MATCHED_PEAKS",
        "TOPK",
        "PPM",
        "TOP_N_CANDIDATES_PER_NODE",
        "FINGERPRINT_METHOD_FOR_CHEMWALKER",
        "CHEMWALKER_PARALLEL",
        "CHEMWALKER_NCORES",
        "CHEMWALKER_HEARTBEAT_SECONDS",
        "CHEMWALKER_COMPONENT_TIMEOUT_SECONDS",
        "CHEMWALKER_EXECUTION_MODE",
        "CHEMWALKER_SIGMOID_A",
        "CHEMWALKER_SIGMOID_B",
        "RESTART_PROB",
        "MNAP_BETA",
        "MNAP_SCORE_EPS",
        "MNAP_EDGE_MODE",
    ]
    return {key: json_safe(globals()[key]) for key in keys}


def _path_identity(path: Path) -> Dict[str, object]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _signature(payload: Dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def callable_source_audit(func) -> Dict[str, object]:
    source_file = inspect.getsourcefile(func)
    source_path = Path(source_file).resolve() if source_file else None
    try:
        source = inspect.getsource(func)
        callable_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()
    except Exception:
        callable_hash = ""
    return {
        "module": getattr(func, "__module__", ""),
        "qualname": getattr(func, "__qualname__", getattr(func, "__name__", "")),
        "signature": str(inspect.signature(func)),
        "source_file": str(source_path) if source_path else "",
        "source_file_sha256": (
            _sha256_file(source_path) if source_path and source_path.exists() else ""
        ),
        "callable_source_sha256": callable_hash,
    }


def network_signature() -> str:
    payload = {
        "schema": NETWORK_CACHE_SCHEMA,
        "input_msp": _path_identity(INPUT_MSP),
        "benchmark_table": _path_identity(BENCHMARK_TABLE),
        "benchmark_processing_batch": BENCHMARK_PROCESSING_BATCH,
        "sample_n": N_COMPOUNDS_TO_SAMPLE,
        "max_spectra": MAX_SPECTRA_PER_COMPOUND,
        "seed": RANDOM_SEED,
        "ionmode": IONMODE_FILTER,
        "adduct": ADDUCT_FILTER,
        "spectrum_type": SPECTRUM_TYPE_FILTER,
        "min_peaks": MIN_PEAKS,
        "fragment_tolerance": FRAGMENT_TOLERANCE,
        "min_cosine": MIN_COSINE,
        "min_matched_peaks": MIN_MATCHED_PEAKS,
        "topk": TOPK,
        "mz_power": MZ_POWER,
        "intensity_power": INTENSITY_POWER,
        "hungarian": USE_HUNGARIAN,
    }
    return _signature(payload)


def candidate_signature() -> str:
    payload = {
        "cache_schema": CANDIDATE_CACHE_SCHEMA,
        "network_signature": network_signature(),
        "database": _path_identity(PUBCHEMLITE_PATH),
        "metfrag": _path_identity(METFRAG_PATH),
        "ppm": PPM,
        "top_n": TOP_N_CANDIDATES_PER_NODE,
        "deduplicate_2d": DEDUPLICATE_CANDIDATES_BY_INCHIKEY1,
        "force_true": FORCE_INCLUDE_TRUE_CANDIDATE,
    }
    return _signature(payload)


def ranking_signature() -> str:
    payload = {
        "candidate_signature": candidate_signature(),
        "release": RELEASE_ID,
        "execution": CHEMWALKER_EXECUTION_MODE,
        "seed_allocation": SEED_ALLOCATION_SCHEME,
        "seed_fraction": SEED_FRACTION,
        "n_seed_repeats": N_SEED_REPEATS,
        "fingerprint": FINGERPRINT_METHOD_FOR_CHEMWALKER,
        "restart": RESTART_PROB,
        "parallel": CHEMWALKER_PARALLEL,
        "ncores": CHEMWALKER_NCORES,
        "cand_pair": callable_source_audit(cand_pair),
        "random_walk": callable_source_audit(random_walk),
        "worker_file": _path_identity(CHEMWALKER_WORKER_SCRIPT),
        "mnap_beta": MNAP_BETA,
        "mnap_eps": MNAP_SCORE_EPS,
        "mnap_mode": MNAP_EDGE_MODE,
        "sigmoid_a": CHEMWALKER_SIGMOID_A,
        "sigmoid_b": CHEMWALKER_SIGMOID_B,
    }
    return _signature(payload)


def repeat_output_dir() -> Path:
    directory = REPEAT_DIR / ranking_signature()[:12]
    directory.mkdir(parents=True, exist_ok=True)
    return directory


# =========================================================
# 3. MSP parser and benchmark-table alignment
# =========================================================


def is_peak_line(line: str) -> bool:
    """Return True when the first two whitespace-delimited fields are numeric."""
    parts = line.strip().split()
    if len(parts) < 2:
        return False
    try:
        float(parts[0])
        float(parts[1])
        return True
    except Exception:
        return False


def parse_msp_manually(msp_file: Path) -> List[Dict[str, object]]:
    """Parse an MSP file record-by-record without loading proprietary software.

    A new record starts at ``Name:``. Peak annotations after the first two
    numeric fields are ignored because only m/z and intensity are required.
    """
    records: List[Dict[str, object]] = []
    metadata: Optional[Dict[str, str]] = None
    mz_values: List[float] = []
    intensities: List[float] = []

    def flush() -> None:
        nonlocal metadata, mz_values, intensities
        if metadata is not None:
            records.append(
                {
                    "metadata": metadata,
                    "mz": mz_values,
                    "intensity": intensities,
                }
            )
        metadata = None
        mz_values = []
        intensities = []

    with Path(msp_file).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.lower().startswith("name:"):
                if metadata is not None:
                    flush()
                metadata = {}
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
                continue
            if metadata is None or line == "":
                continue
            if is_peak_line(line):
                parts = line.split()
                mz_values.append(float(parts[0]))
                intensities.append(float(parts[1]))
                continue
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()

    if metadata is not None:
        flush()
    return records


def _normalized_column_name(value: str) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())


def _find_table_column(
    df: pd.DataFrame,
    exact: Sequence[str],
    contains: Sequence[str] = (),
    exclude: Sequence[Optional[str]] = (),
):
    excluded = {column for column in exclude if column is not None}
    normalized = {_normalized_column_name(column): column for column in df.columns}
    for candidate in exact:
        key = _normalized_column_name(candidate)
        if key in normalized and normalized[key] not in excluded:
            return normalized[key]
    for column in df.columns:
        if column in excluded:
            continue
        key = _normalized_column_name(column)
        if any(_normalized_column_name(token) in key for token in contains):
            return column
    return None


def read_chemwalker_benchmark_table(
    path: Path,
    processing_batch: int = BENCHMARK_PROCESSING_BATCH,
) -> pd.DataFrame:
    """Read the published validation table without using published ranks.

    Only identifiers, verified structure fields, precursor mass, and cluster
    index are retained. Columns named MetFrag, Random, Fusion, or Consensus are
    deliberately ignored.
    """
    raw = pd.read_csv(path, sep="\t", dtype=str)
    if raw.empty:
        raise ValueError(f"Benchmark table is empty: {path}")

    batch_col = _find_table_column(
        raw,
        exact=["Processing batch", "Processing.batch"],
        contains=["processingbatch"],
        exclude=["Processing batch.1", "Processing.batch.1"],
    )
    if batch_col is None:
        batch_col = _find_table_column(
            raw,
            exact=["Processing batch.1", "Processing.batch.1"],
            contains=["processingbatch1"],
        )
    cluster_col = _find_table_column(
        raw,
        exact=["cluster.index", "cluster index", "cluster_index"],
        contains=["clusterindex"],
    )
    mass_col = _find_table_column(
        raw,
        exact=["parent.mass", "parent mass", "precursor mass"],
        contains=["parentmass", "precursormass"],
    )
    key_col = _find_table_column(
        raw,
        exact=["InChIKey Recovered", "InChIKey", "INCHIKEY"],
        contains=["inchikeyrecovered", "inchikey"],
    )
    inchi_col = _find_table_column(
        raw,
        exact=["InChI Recovered", "InChI"],
        contains=["inchirecovered"],
        exclude=[key_col],
    )
    smiles_col = _find_table_column(
        raw,
        exact=["SMILES Recovered", "SMILES", "Smiles"],
        contains=["smilesrecovered", "smiles"],
    )
    name_col = _find_table_column(
        raw,
        exact=["Compound_Name", "Compound name", "name"],
        contains=["compoundname"],
    )

    required = {"processing batch": batch_col, "cluster index": cluster_col, "InChIKey": key_col}
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(
            "Cannot detect required benchmark columns: "
            + ", ".join(missing)
            + f". Available columns: {raw.columns.tolist()}"
        )

    out = pd.DataFrame()
    out["processing_batch"] = pd.to_numeric(raw[batch_col], errors="coerce")
    out["cluster_index"] = pd.to_numeric(raw[cluster_col], errors="coerce")
    out["parent_mass_table"] = (
        pd.to_numeric(raw[mass_col], errors="coerce") if mass_col is not None else np.nan
    )
    out["true_inchikey"] = raw[key_col].map(clean_str).str.upper()
    out["true_inchikey1"] = out["true_inchikey"].map(inchikey1)
    out["true_inchi"] = raw[inchi_col].map(normalize_inchi) if inchi_col is not None else ""
    out["true_smiles"] = raw[smiles_col].map(clean_str) if smiles_col is not None else ""
    out["compound_name_table"] = raw[name_col].map(clean_str) if name_col is not None else ""
    out["source_row"] = np.arange(len(raw), dtype=int)

    out = out.dropna(subset=["processing_batch", "cluster_index"]).copy()
    out["processing_batch"] = out["processing_batch"].astype(int)
    out["cluster_index"] = out["cluster_index"].astype(int)
    out = out[
        (out["processing_batch"] == int(processing_batch))
        & out["true_inchikey1"].astype(str).str.len().eq(14)
    ].copy()
    out = out.sort_values("cluster_index").reset_index(drop=True)

    bad_inchi = ~out["true_inchi"].astype(str).str.startswith("InChI=")
    if bad_inchi.any():
        out.loc[bad_inchi, "true_inchi"] = out.loc[bad_inchi, "true_smiles"].map(
            smiles_to_inchi
        )

    if out.duplicated("true_inchikey").any():
        out = out.drop_duplicates("true_inchikey", keep="first").copy()
    return out


def _msp_positive_mode(value: str) -> bool:
    text = clean_str(value).upper()
    return text in {"P", "POS", "POSITIVE", "+"} or text.startswith("POS")


def load_annotated_msp(
    msp_file: Path,
    benchmark_table: Path,
):
    raw_spectra = parse_msp_manually(msp_file)
    truth = read_chemwalker_benchmark_table(benchmark_table)

    full_lookup = truth.set_index("true_inchikey").to_dict("index")
    first_lookup = (
        truth.sort_values("cluster_index")
        .drop_duplicates("true_inchikey1")
        .set_index("true_inchikey1")
        .to_dict("index")
    )

    spectra: List[Spectrum] = []
    rows: List[Dict[str, object]] = []
    missing_truth: List[Dict[str, object]] = []
    skipped = {
        "not_ms2": 0,
        "not_m_plus_h": 0,
        "not_positive": 0,
        "too_few_peaks": 0,
        "missing_key": 0,
        "missing_truth": 0,
        "missing_structure": 0,
        "missing_precursor": 0,
    }
    node_id = 0

    for raw_index, raw in enumerate(raw_spectra, start=1):
        metadata = dict(raw["metadata"])
        spectrum_type = clean_str(meta_get(metadata, ["Spectrum_type", "Spectrum type"], ""))
        precursor_type = clean_str(meta_get(metadata, ["Precursor_type", "Precursor type"], ""))
        ion_mode = clean_str(meta_get(metadata, ["Ion_mode", "Ion mode"], ""))

        if SPECTRUM_TYPE_FILTER and spectrum_type.upper() not in {"MS2", "MS/MS", "MSMS"}:
            skipped["not_ms2"] += 1
            continue
        if ADDUCT_FILTER and precursor_type.replace(" ", "") != ADDUCT_FILTER.replace(" ", ""):
            skipped["not_m_plus_h"] += 1
            continue
        if IONMODE_FILTER == "positive" and ion_mode and not _msp_positive_mode(ion_mode):
            skipped["not_positive"] += 1
            continue

        mz_array = np.asarray(raw["mz"], dtype=float)
        intensity_array = np.asarray(raw["intensity"], dtype=float)
        if len(mz_array) < MIN_PEAKS:
            skipped["too_few_peaks"] += 1
            continue

        msp_key = clean_str(meta_get(metadata, ["InChIKey", "INCHIKEY"], "")).upper()
        key1 = inchikey1(msp_key)
        if len(key1) != 14:
            skipped["missing_key"] += 1
            continue

        truth_row = full_lookup.get(msp_key) or first_lookup.get(key1)
        if truth_row is None:
            skipped["missing_truth"] += 1
            missing_truth.append({"raw_spectrum_index": raw_index, "InChIKey": msp_key})
            continue

        inchi = normalize_inchi(truth_row.get("true_inchi", ""))
        smiles = clean_str(truth_row.get("true_smiles", ""))
        if not inchi.startswith("InChI=") and smiles:
            inchi = smiles_to_inchi(smiles)
        if not inchi.startswith("InChI="):
            skipped["missing_structure"] += 1
            continue
        computed_key1 = inchikey1(inchi_to_inchikey(inchi))
        if computed_key1 and computed_key1 != key1:
            raise ValueError(
                f"Ground-truth structure mismatch for MSP InChIKey {msp_key}: "
                f"computed first block {computed_key1}."
            )

        precursor_mz = to_float(meta_get(metadata, ["PrecursorMZ", "Precursor m/z"], np.nan))
        exact_mass = to_float(meta_get(metadata, ["ExactMass", "Exact mass"], np.nan))
        if pd.isna(precursor_mz):
            skipped["missing_precursor"] += 1
            continue
        neutral_mass = exact_mass if pd.notna(exact_mass) else infer_neutral_mass(
            precursor_mz, precursor_type, exactmass=exact_mass
        )
        if pd.isna(neutral_mass):
            skipped["missing_precursor"] += 1
            continue

        name = clean_str(meta_get(metadata, ["Name"], "")) or clean_str(
            truth_row.get("compound_name_table", "")
        )
        node_id += 1
        cluster_index = int(truth_row["cluster_index"])
        msp_id = clean_str(meta_get(metadata, ["ID"], ""))
        nist_no = clean_str(meta_get(metadata, ["NISTNO", "NIST No"], ""))
        formula = clean_str(meta_get(metadata, ["Formula"], ""))

        metadata2 = dict(metadata)
        metadata2.update(
            {
                "node_id": str(node_id),
                "cluster_index": str(cluster_index),
                "feature_id_original": msp_id,
                "scans_original": str(cluster_index),
                "usi": "",
                "precursor_mz": float(precursor_mz),
                "neutral_mass": float(neutral_mass),
                "inchi": inchi,
                "inchikey": msp_key,
                "inchikey1": key1,
                "name": name,
                "smiles": smiles,
                "formula": formula,
                "adduct": precursor_type,
                "ionmode": ion_mode,
            }
        )
        spectra.append(
            Spectrum(
                mz=mz_array,
                intensities=intensity_array,
                metadata=metadata2,
            )
        )
        rows.append(
            {
                "node_id": node_id,
                "cluster_index": cluster_index,
                "raw_spectrum_index": raw_index,
                "name": name,
                "feature_id_original": msp_id,
                "scans_original": str(cluster_index),
                "usi": "",
                "pepmass": float(precursor_mz),
                "neutral_mass": float(neutral_mass),
                "exactmass": exact_mass,
                "formula": formula,
                "adduct": precursor_type,
                "ionmode": ion_mode,
                "n_peaks": len(mz_array),
                "InChI": inchi,
                "SMILES": smiles,
                "InChIKey": msp_key,
                "InChIKey1": key1,
                "quality_explained_intensity": 0.0,
                "quality_explained_signals": 0.0,
                "spectype": spectrum_type,
                "NISTNO": nist_no,
                "processing_batch": int(truth_row["processing_batch"]),
            }
        )

    table = pd.DataFrame(rows)
    if missing_truth:
        pd.DataFrame(missing_truth).to_csv(
            OUT_DIR / "msp_records_missing_from_benchmark_table.csv", index=False
        )

    print("Raw MSP records:", len(raw_spectra))
    print("Benchmark rows for batch", BENCHMARK_PROCESSING_BATCH, ":", len(truth))
    print("Loaded aligned spectra:", len(spectra))
    print("Unique 2D structures:", table["InChIKey1"].nunique() if not table.empty else 0)
    print("Skipped records:", skipped)
    return spectra, table


def choose_representative_spectra(spectra, spectra_table):
    """Keep one deterministic representative spectrum per InChIKey first block."""
    table = spectra_table.copy()
    table["quality_explained_intensity"] = pd.to_numeric(
        table["quality_explained_intensity"], errors="coerce"
    ).fillna(0)
    table = table.sort_values(
        ["InChIKey1", "quality_explained_intensity", "n_peaks", "cluster_index", "node_id"],
        ascending=[True, False, False, True, True],
    )
    chosen = (
        table.groupby("InChIKey1", group_keys=False)
        .head(MAX_SPECTRA_PER_COMPOUND)
        .copy()
    )
    chosen_ids = set(chosen["node_id"].astype(int))
    chosen_spectra = [
        spectrum
        for spectrum in spectra
        if int(meta_get(spectrum.metadata, "node_id")) in chosen_ids
    ]
    return chosen_spectra, chosen


def sample_compounds(
    spectra,
    spectra_table,
    n_compounds: Optional[int],
    random_seed: int,
):
    compounds = sorted(
        spectra_table["InChIKey1"].dropna().astype(str).unique().tolist()
    )
    if n_compounds is not None and len(compounds) > n_compounds:
        rng = random.Random(random_seed)
        selected = set(rng.sample(compounds, n_compounds))
    else:
        selected = set(compounds)
    table = spectra_table[
        spectra_table["InChIKey1"].astype(str).isin(selected)
    ].copy()
    chosen_ids = set(table["node_id"].astype(int))
    sampled_spectra = [
        spectrum
        for spectrum in spectra
        if int(meta_get(spectrum.metadata, "node_id")) in chosen_ids
    ]
    print("Sampled spectra:", len(sampled_spectra))
    print("Sampled compounds:", table["InChIKey1"].nunique())
    return sampled_spectra, table


# =========================================================
# 4. Molecular network
# =========================================================


def get_modified_cosine_scorer():
    """Create a modified-cosine scorer across supported matchms APIs.

    Different matchms releases expose ``ModifiedCosineGreedy``,
    ``ModifiedCosineHungarian``, or ``ModifiedCosine``. The available class is
    selected without changing the benchmark tolerance or weighting parameters.
    """
    import matchms

    attempts = []
    if USE_HUNGARIAN:
        import_paths = [
            ("matchms.similarity", "ModifiedCosineHungarian"),
            ("matchms.similarity.ModifiedCosineHungarian", "ModifiedCosineHungarian"),
        ]
    else:
        import_paths = [
            ("matchms.similarity", "ModifiedCosineGreedy"),
            ("matchms.similarity.ModifiedCosineGreedy", "ModifiedCosineGreedy"),
            ("matchms.similarity", "ModifiedCosine"),
            ("matchms.similarity.ModifiedCosine", "ModifiedCosine"),
        ]

    for module_name, class_name in import_paths:
        try:
            module = __import__(module_name, fromlist=[class_name])
            scorer_class = getattr(module, class_name)
            scorer = scorer_class(
                tolerance=FRAGMENT_TOLERANCE,
                mz_power=MZ_POWER,
                intensity_power=INTENSITY_POWER,
            )
            version = getattr(matchms, "__version__", "unknown")
            return scorer, f"{class_name} (matchms {version})"
        except Exception as exc:
            attempts.append(f"{module_name}.{class_name}: {type(exc).__name__}: {exc}")

    raise ImportError(
        "No compatible modified-cosine scorer was found. Attempts:\n- "
        + "\n- ".join(attempts)
    )


def get_score_and_matches(score_result):
    try:
        return float(score_result["score"]), int(score_result["matches"])
    except Exception:
        pass
    try:
        return float(score_result[0]), int(score_result[1])
    except Exception as exc:
        raise ValueError(f"Cannot parse matchms score result: {score_result}") from exc


def compute_modified_cosine_edges(spectra):
    scorer, scorer_name = get_modified_cosine_scorer()
    print("Using scorer:", scorer_name)
    edges = []
    n_spectra = len(spectra)

    for i in range(n_spectra):
        if i % 100 == 0:
            print(f"Pairing {i}/{n_spectra}")
        spectrum1 = spectra[i]
        node1 = int(meta_get(spectrum1.metadata, "node_id"))
        precursor1 = to_float(meta_get(spectrum1.metadata, "precursor_mz"))

        for j in range(i + 1, n_spectra):
            spectrum2 = spectra[j]
            node2 = int(meta_get(spectrum2.metadata, "node_id"))
            precursor2 = to_float(meta_get(spectrum2.metadata, "precursor_mz"))
            result = scorer.pair(spectrum1, spectrum2)
            cosine, matches = get_score_and_matches(result)

            if cosine >= MIN_COSINE and matches >= MIN_MATCHED_PEAKS:
                edges.append(
                    {
                        "CLUSTERID1": node1,
                        "CLUSTERID2": node2,
                        "Cosine": float(cosine),
                        "matched_peaks": int(matches),
                        "DeltaMZ": float(precursor2 - precursor1),
                        "precursor_mz_1": precursor1,
                        "precursor_mz_2": precursor2,
                    }
                )

    out = pd.DataFrame(edges)
    print("Raw edges:", out.shape)
    return out


def apply_mutual_topk(edges: pd.DataFrame, topk: Optional[int]):
    if edges.empty or topk is None:
        return edges.copy()

    neighbors: Dict[int, List[Tuple[int, float]]] = {}
    for row in edges.itertuples(index=False):
        source = int(row.CLUSTERID1)
        target = int(row.CLUSTERID2)
        score = float(row.Cosine)
        neighbors.setdefault(source, []).append((target, score))
        neighbors.setdefault(target, []).append((source, score))

    top_neighbors = {}
    for node, values in neighbors.items():
        ranked = sorted(values, key=lambda value: (-value[1], value[0]))
        top_neighbors[node] = {value[0] for value in ranked[:topk]}

    keep = []
    for row in edges.itertuples(index=False):
        source = int(row.CLUSTERID1)
        target = int(row.CLUSTERID2)
        keep.append(
            target in top_neighbors.get(source, set())
            and source in top_neighbors.get(target, set())
        )

    out = edges.loc[keep].copy().reset_index(drop=True)
    print("Mutual TopK edges:", out.shape)
    return out


def add_component_index(edges: pd.DataFrame):
    out = edges.copy()
    if out.empty:
        out["ComponentIndex"] = pd.Series(dtype=int)
        return out

    graph = nx.Graph()
    graph.add_edges_from(
        out[["CLUSTERID1", "CLUSTERID2"]]
        .astype(int)
        .itertuples(index=False, name=None)
    )

    component_map = {}
    components = sorted(
        nx.connected_components(graph),
        key=lambda nodes: (-len(nodes), min(nodes)),
    )
    for component_id, nodes in enumerate(components, start=1):
        for node in nodes:
            component_map[int(node)] = component_id

    out["ComponentIndex"] = (
        out["CLUSTERID1"].astype(int).map(component_map).astype(int)
    )
    return out


def attach_components_to_nodes(nodes: pd.DataFrame, edges: pd.DataFrame):
    mapping = {}
    for row in edges[["CLUSTERID1", "ComponentIndex"]].itertuples(index=False):
        mapping[int(row.CLUSTERID1)] = int(row.ComponentIndex)
    for row in edges[["CLUSTERID2", "ComponentIndex"]].itertuples(index=False):
        mapping[int(row.CLUSTERID2)] = int(row.ComponentIndex)

    out = nodes.copy()
    out["ComponentIndex"] = out["node_id"].astype(int).map(mapping)
    out = out.dropna(subset=["ComponentIndex"]).copy()
    out["ComponentIndex"] = out["ComponentIndex"].astype(int)
    return out


def component_size_table(edges: pd.DataFrame):
    rows = []
    for component_id, group in edges.groupby("ComponentIndex"):
        nodes = set(group["CLUSTERID1"].astype(int)).union(
            set(group["CLUSTERID2"].astype(int))
        )
        rows.append(
            {
                "ComponentIndex": int(component_id),
                "n_nodes": len(nodes),
                "n_edges": int(group.shape[0]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_nodes", "n_edges"], ascending=False
    )


# =========================================================
# 5. PubChemLite and MetFrag candidates
# =========================================================


def read_table_auto(path: Path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return pd.read_csv(path)


def pick_col(df: pd.DataFrame, candidates: Sequence[str]):
    lower_map = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    for column in df.columns:
        lower = str(column).lower()
        for candidate in candidates:
            if candidate.lower() in lower:
                return column
    return None


def standardize_pubchemlite_db(path: Path):
    raw = read_table_auto(path)
    raw.columns = [str(column).strip() for column in raw.columns]
    print("Raw DB shape:", raw.shape)
    print("Raw DB columns:", raw.columns.tolist()[:50])

    id_col = pick_col(raw, ["Identifier", "CID", "PubChemCID", "compound_id", "id"])
    mass_col = pick_col(
        raw,
        ["MonoisotopicMass", "monoisotopic_mass", "exactmass", "ExactMass", "mass"],
    )
    inchi_col = pick_col(raw, ["InChI", "inchi", "INCHI"])
    smiles_col = pick_col(
        raw, ["SMILES", "smiles", "CanonicalSMILES", "canonical_smiles"]
    )
    formula_col = pick_col(raw, ["MolecularFormula", "formula", "molecular_formula"])

    if id_col is None or mass_col is None:
        raise ValueError("PubChemLite requires Identifier and MonoisotopicMass columns.")
    if inchi_col is None and smiles_col is None:
        raise ValueError("PubChemLite requires InChI or SMILES.")

    db = pd.DataFrame()
    db["Identifier"] = raw[id_col].astype(str)
    db["MonoisotopicMass"] = pd.to_numeric(raw[mass_col], errors="coerce")
    db["MolecularFormula"] = (
        raw[formula_col].astype(str) if formula_col is not None else ""
    )
    db["SMILES"] = raw[smiles_col].astype(str) if smiles_col is not None else ""
    db["InChI"] = (
        raw[inchi_col].apply(normalize_inchi) if inchi_col is not None else ""
    )

    bad_inchi = (
        db["InChI"].isna()
        | db["InChI"].astype(str).str.strip().eq("")
        | ~db["InChI"].astype(str).str.startswith("InChI=")
    )
    if bad_inchi.any() and smiles_col is not None:
        db.loc[bad_inchi, "InChI"] = db.loc[bad_inchi, "SMILES"].apply(
            smiles_to_inchi
        )

    db = db.dropna(subset=["MonoisotopicMass"]).copy()
    db = db[db["InChI"].astype(str).str.startswith("InChI=")].copy()
    db["InChIKey"] = db["InChI"].apply(inchi_to_inchikey)
    db = db[
        db["InChIKey"].notna()
        & db["InChIKey"].astype(str).str.contains("-", regex=False)
    ].copy()

    key_parts = db["InChIKey"].astype(str).str.split("-", n=2, expand=True)
    db["InChIKey1"] = key_parts[0]
    db["InChIKey2"] = key_parts[1] if key_parts.shape[1] > 1 else ""
    db["InChIKey3"] = key_parts[2] if key_parts.shape[1] > 2 else ""
    db = db[
        db["InChIKey1"].astype(str).str.len().gt(5)
        & db["InChIKey2"].astype(str).str.len().gt(0)
    ].copy()

    keep = [
        "Identifier",
        "MonoisotopicMass",
        "MolecularFormula",
        "SMILES",
        "InChI",
        "InChIKey",
        "InChIKey1",
        "InChIKey2",
        "InChIKey3",
    ]
    db = db[keep].drop_duplicates("Identifier").reset_index(drop=True)
    print("Standardized DB shape:", db.shape)
    return db


def mass_filter_db(db: pd.DataFrame, neutral_mass: float, ppm: float):
    tolerance = neutral_mass * ppm * 1e-6
    return db[
        db["MonoisotopicMass"].between(
            neutral_mass - tolerance,
            neutral_mass + tolerance,
            inclusive="both",
        )
    ].copy()


def build_metfrag_like_spec(spectrum: Spectrum):
    mz, intensity = get_spectrum_mz_intensity(spectrum)
    node_id = str(meta_get(spectrum.metadata, "node_id"))
    return {
        "m/z array": mz,
        "intensity array": intensity,
        "params": {
            "pepmass": [float(meta_get(spectrum.metadata, "precursor_mz"))],
            "charge": [1],
            "scans": node_id,
            "feature_id": node_id,
            "title": meta_get(spectrum.metadata, "name", node_id),
        },
    }


# ---------- robust MetFrag PSV parsing ----------

OFFICIAL_METFRAG_SCORE_COLUMN = "Score"
METFRAG_WRAPPER_INDEX_COLUMN = "__metfrag_wrapper_index__"
METFRAG_WRAPPER_INDEX_DEFAULT_COLUMN = "__metfrag_wrapper_index_is_default__"
METFRAG_SCORE_SOURCE_COLUMN = "Score column"
METFRAG_SCORE_SOURCE_INDEX = "DataFrame index (recovered PSV Score)"


def _make_unique_column_names(columns: Sequence[str]):
    counts: Dict[str, int] = {}
    names = []
    for column in columns:
        base = str(column).strip()
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}__{counts[base]}")
    return names


def _base_name(column: str) -> str:
    return str(column).split("__", 1)[0]


def _columns_with_base(df: pd.DataFrame, base: str):
    return [column for column in df.columns if _base_name(column) == base]


def _looks_like_normalized_metfrag_score(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return False
    return bool(
        numeric.min() >= -1e-12
        and numeric.max() <= 1.000001
        and np.isfinite(numeric).all()
    )


def _numeric_summary(values: pd.Series) -> Dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce")
    return {
        "n": int(numeric.notna().sum()),
        "min": float(numeric.min()) if numeric.notna().any() else None,
        "max": float(numeric.max()) if numeric.notna().any() else None,
        "head": numeric.head(DEBUG_METFRAG_ROWS).tolist(),
    }


def normalize_metfrag_output(result: pd.DataFrame):
    if result is None or len(result) == 0:
        return pd.DataFrame()

    df = clear_df_attrs(result)
    raw_index = df.index.copy()
    is_default_range = (
        isinstance(raw_index, pd.RangeIndex)
        and raw_index.start == 0
        and raw_index.stop == len(df)
        and raw_index.step == 1
    )
    df.columns = _make_unique_column_names(
        [str(column).strip() for column in df.columns]
    )
    df.insert(0, METFRAG_WRAPPER_INDEX_COLUMN, list(raw_index))
    df.insert(1, METFRAG_WRAPPER_INDEX_DEFAULT_COLUMN, bool(is_default_range))
    df = df.reset_index(drop=True)
    df.attrs.clear()
    return df


def extract_verified_metfrag_score(df: pd.DataFrame, node_id: int):
    score_columns = _columns_with_base(df, OFFICIAL_METFRAG_SCORE_COLUMN)
    if not score_columns:
        raise MetFragOutputValidationError(
            f"MetFrag output for node {node_id} has no visible Score column."
        )

    visible_column = max(
        score_columns,
        key=lambda column: pd.to_numeric(df[column], errors="coerce")
        .notna()
        .sum(),
    )
    visible_score = pd.to_numeric(df[visible_column], errors="coerce")
    index_values = pd.to_numeric(df[METFRAG_WRAPPER_INDEX_COLUMN], errors="coerce")
    index_is_default = bool(
        df[METFRAG_WRAPPER_INDEX_DEFAULT_COLUMN].astype(bool).all()
    )

    if _looks_like_normalized_metfrag_score(visible_score):
        return visible_score, METFRAG_SCORE_SOURCE_COLUMN, visible_column

    if (
        not index_is_default
        and _looks_like_normalized_metfrag_score(index_values)
    ):
        return index_values, METFRAG_SCORE_SOURCE_INDEX, METFRAG_WRAPPER_INDEX_COLUMN

    raise MetFragOutputValidationError(
        f"Unable to identify the true MetFrag Score for node {node_id}.\n"
        f"Visible Score: {_numeric_summary(visible_score)}\n"
        f"DataFrame index: {_numeric_summary(index_values)}\n"
        "Benchmark stopped to avoid ranking with a mislabeled mass/score field."
    )


def _find_best_inchi_column(df: pd.DataFrame):
    candidates = []
    for column in df.columns:
        values = df[column].astype(str)
        count = int(values.str.contains("InChI=", regex=False, na=False).sum())
        exact_bonus = int(_base_name(column) == "InChI")
        candidates.append((count, exact_bonus, column))
    candidates.sort(reverse=True)
    if not candidates or candidates[0][0] <= 0:
        return None, 0
    return candidates[0][2], candidates[0][0]


def _normalize_identifier(value) -> str:
    text = clean_str(value)
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except Exception:
            return text
    return text


def _map_identifiers_from_candidate_db(
    df: pd.DataFrame,
    candidate_db: Optional[pd.DataFrame],
):
    out = df.copy()
    out["Identifier"] = ""

    if candidate_db is None or candidate_db.empty:
        out["Identifier"] = [
            f"cand_{index}_{key}"
            for index, key in enumerate(out["InChIKey1"].astype(str))
        ]
        return out

    mapping = candidate_db.copy()
    mapping["Identifier"] = mapping["Identifier"].map(_normalize_identifier)
    full_key_map = (
        mapping.drop_duplicates("InChIKey")
        .set_index("InChIKey")["Identifier"]
        .to_dict()
    )
    inchi_map = (
        mapping.drop_duplicates("InChI")
        .set_index("InChI")["Identifier"]
        .to_dict()
    )

    out["Identifier"] = out["InChIKey"].map(full_key_map).fillna("")
    missing = out["Identifier"].astype(str).eq("")
    out.loc[missing, "Identifier"] = (
        out.loc[missing, "InChI"].map(inchi_map).fillna("")
    )
    missing = out["Identifier"].astype(str).eq("")
    if missing.any():
        out.loc[missing, "Identifier"] = [
            f"cand_{index}_{key}"
            for index, key in zip(
                out.index[missing],
                out.loc[missing, "InChIKey1"].astype(str),
            )
        ]
    return out


def empty_candidate_table():
    return pd.DataFrame(
        columns=[
            "uid",
            "node_id",
            "Identifier",
            "InChI",
            "InChIKey",
            "InChIKey1",
            "Score",
            "MetFragScoreRaw",
            "score_source",
            "score_norm",
            "forced_true_candidate",
        ]
    )


_METFRAG_AUDIT_COUNTER = 0


def audit_metfrag_result(
    raw_result: pd.DataFrame,
    cleaned: pd.DataFrame,
    node_id: int,
) -> None:
    global _METFRAG_AUDIT_COUNTER
    if not DEBUG_METFRAG_SCORE:
        return
    if _METFRAG_AUDIT_COUNTER >= DEBUG_METFRAG_MAX_NODES:
        return

    _METFRAG_AUDIT_COUNTER += 1
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    normalized = normalize_metfrag_output(raw_result)
    raw_path = AUDIT_DIR / f"node_{node_id}_raw.csv"
    clean_path = AUDIT_DIR / f"node_{node_id}_cleaned.csv"
    normalized.to_csv(raw_path, index=False)
    cleaned.to_csv(clean_path, index=False)

    visible_columns = _columns_with_base(normalized, "Score")
    visible = (
        pd.to_numeric(normalized[visible_columns[0]], errors="coerce")
        if visible_columns
        else pd.Series(dtype=float)
    )
    index_values = pd.to_numeric(
        normalized[METFRAG_WRAPPER_INDEX_COLUMN], errors="coerce"
    )

    print("\n" + "=" * 90)
    print(f"MetFrag score verification for node {node_id}")
    print("Raw columns:", list(raw_result.columns))
    print("Visible Score head:", visible.head(DEBUG_METFRAG_ROWS).tolist())
    print("DataFrame index head:", index_values.head(DEBUG_METFRAG_ROWS).tolist())
    print("Verified candidates:")
    preview_columns = [
        "Identifier",
        "Score",
        "MetFragScoreRaw",
        "score_source",
        "InChIKey1",
    ]
    print(
        cleaned[preview_columns]
        .head(DEBUG_METFRAG_ROWS)
        .to_string(index=False)
    )
    print("Raw audit:", raw_path)
    print("Cleaned audit:", clean_path)


def clean_metfrag_result(
    result: pd.DataFrame,
    node_id: int,
    candidate_db: Optional[pd.DataFrame],
):
    df = normalize_metfrag_output(result)
    if df.empty:
        return empty_candidate_table()

    verified_score, score_source, _ = extract_verified_metfrag_score(
        df, node_id=node_id
    )
    df["MetFragScoreRaw"] = pd.to_numeric(verified_score, errors="coerce")
    df["Score"] = df["MetFragScoreRaw"]
    df["score_source"] = score_source

    inchi_column, count = _find_best_inchi_column(df)
    if inchi_column is None:
        raise MetFragOutputValidationError(
            f"MetFrag returned {len(df)} rows for node {node_id}, but no field "
            "contained an InChI string."
        )

    df["InChI"] = df[inchi_column].apply(normalize_inchi)
    df = df[df["InChI"].astype(str).str.startswith("InChI=")].copy()
    if df.empty:
        raise MetFragOutputValidationError(
            f"No valid candidate InChI for node {node_id}; raw count={count}."
        )

    df["node_id"] = int(node_id)
    df["InChIKey"] = df["InChI"].apply(inchi_to_inchikey)
    df["InChIKey1"] = df["InChIKey"].apply(inchikey1)
    df = df[df["InChIKey1"].astype(str).ne("")].copy()
    df = _map_identifiers_from_candidate_db(df, candidate_db)

    df = df.sort_values(
        ["Score", "Identifier"], ascending=[False, True]
    ).copy()
    if DEDUPLICATE_CANDIDATES_BY_INCHIKEY1:
        df = df.drop_duplicates("InChIKey1", keep="first")
    df = df.head(TOP_N_CANDIDATES_PER_NODE).copy()

    df["Identifier"] = df["Identifier"].astype(str)
    df["uid"] = (
        df["node_id"].astype(int).astype(str)
        + "_"
        + df["Identifier"].astype(str)
    )
    df["score_norm"] = (
        df["Score"] / df["Score"].max() if df["Score"].max() > 0 else 0.0
    )
    df["forced_true_candidate"] = False

    keep = [
        "uid",
        "node_id",
        "Identifier",
        "InChI",
        "InChIKey",
        "InChIKey1",
        "Score",
        "MetFragScoreRaw",
        "score_source",
        "score_norm",
        "forced_true_candidate",
    ]
    out = df[keep].reset_index(drop=True)
    out.attrs.clear()
    return out


def candidate_cache_path(node_id: int):
    return CACHE_DIR / (
        f"{CANDIDATE_CACHE_SCHEMA}_{candidate_signature()[:12]}_"
        f"node_{node_id}_ppm{PPM}_top{TOP_N_CANDIDATES_PER_NODE}.pkl"
    )


def generate_candidates_for_node(
    spectrum: Spectrum,
    true_row: pd.Series,
    db: pd.DataFrame,
    force: bool = False,
):
    node_id = int(meta_get(spectrum.metadata, "node_id"))
    neutral_mass = float(meta_get(spectrum.metadata, "neutral_mass"))
    cache_path = candidate_cache_path(node_id)

    if cache_path.exists() and not force:
        out = clear_df_attrs(load_pickle(cache_path))
        return out

    candidate_db = mass_filter_db(db, neutral_mass, ppm=PPM)
    if candidate_db.empty:
        out = empty_candidate_table()
        save_pickle(out, cache_path)
        return out

    if not USE_METFRAG or run_metfrag is None:
        raise RuntimeError("MetFrag is required and run_metfrag could not be imported.")

    try:
        metfrag_input = build_metfrag_like_spec(spectrum)
        raw_result = run_metfrag(
            metfrag_input,
            candidate_db,
            node_id,
            metpath=str(METFRAG_PATH),
            ppm=PPM,
        )
        out = clean_metfrag_result(
            raw_result,
            node_id=node_id,
            candidate_db=candidate_db,
        )
        if not out.empty:
            audit_metfrag_result(raw_result, out, node_id=node_id)
    except MetFragOutputValidationError:
        # A score/field parsing problem invalidates the baseline and therefore
        # stops the benchmark immediately instead of being counted as a missing
        # candidate list.
        raise
    except Exception as exc:
        if FALLBACK_TO_MASS_SCORE_IF_METFRAG_FAILS:
            raise NotImplementedError(
                "Mass-score fallback is intentionally disabled for formal benchmarking."
            ) from exc
        print(f"MetFrag execution failed for node {node_id}: {exc}")
        out = empty_candidate_table()

    if FORCE_INCLUDE_TRUE_CANDIDATE and not out.empty:
        true_key = str(true_row["InChIKey1"])
        if not out["InChIKey1"].astype(str).eq(true_key).any():
            raise ValueError(
                "FORCE_INCLUDE_TRUE_CANDIDATE must remain False for formal evaluation."
            )

    save_pickle(out, cache_path)
    return out


# =========================================================
# 6. Seed allocation by connected component
# =========================================================


def assign_component_seeds(nodes: pd.DataFrame, repeat_id: int):
    assignments = []
    for component_id, group in nodes.groupby("ComponentIndex"):
        node_ids = sorted(group["node_id"].astype(int).unique().tolist())
        n_nodes = len(node_ids)
        if n_nodes < 2:
            continue

        n_seed = max(1, int(math.ceil(n_nodes * SEED_FRACTION)))
        n_seed = min(n_seed, n_nodes - 1)
        rng = random.Random(
            stable_seed(SEED_ALLOCATION_SCHEME, RANDOM_SEED, repeat_id, component_id)
        )
        seed_nodes = set(rng.sample(node_ids, n_seed))

        for node_id in node_ids:
            assignments.append(
                {
                    "repeat_id": int(repeat_id),
                    "ComponentIndex": int(component_id),
                    "node_id": int(node_id),
                    "split": "seed" if node_id in seed_nodes else "target",
                    "n_component_nodes": n_nodes,
                    "n_component_seeds": n_seed,
                }
            )

    assignment = pd.DataFrame(assignments)
    return nodes.merge(
        assignment,
        on=["ComponentIndex", "node_id"],
        how="inner",
        validate="one_to_one",
    )


def build_library_match_seed_rows(assignment: pd.DataFrame):
    """Represent each selected seed spectrum by its verified library structure.

    This matches the original ChemWalker candidate graph: a spectral-library
    match contributes one known structure node, whereas non-seed spectra are
    represented by their MetFrag candidate lists. ``Identifier`` is only a
    unique graph label; the seed chemistry is the verified InChI/InChIKey from
    the benchmark table.
    """
    seed = assignment[assignment["split"] == "seed"].copy()
    seed["Identifier"] = "LibraryMatch_" + seed["node_id"].astype(str)
    seed["Score"] = 1.0
    seed["score_norm"] = 1.0
    seed["forced_true_candidate"] = False
    seed["is_seed"] = True
    seed["seed_type"] = "verified_spectral_library_match"
    seed["uid"] = (
        seed["node_id"].astype(int).astype(str)
        + "_"
        + seed["Identifier"].astype(str)
    )
    keep = [
        "uid",
        "node_id",
        "Identifier",
        "InChI",
        "InChIKey",
        "InChIKey1",
        "Score",
        "score_norm",
        "forced_true_candidate",
        "is_seed",
        "seed_type",
    ]
    return clear_df_attrs(seed[keep])


# Backward-compatible name for callers outside this package.
def build_seed_tlid(assignment: pd.DataFrame):
    return build_library_match_seed_rows(assignment)


# =========================================================
# 7. Candidate graphs, ChemWalker, and mNAP
# =========================================================

_FP_CACHE: Dict[Tuple[str, str], object] = {}


def mol_from_inchi(inchi):
    try:
        text = normalize_inchi(inchi)
        return Chem.MolFromInchi(text) if text.startswith("InChI=") else None
    except Exception:
        return None


def get_structural_fingerprint(
    inchi,
    method: str = FINGERPRINT_METHOD_FOR_CHEMWALKER,
):
    """Return the same RDKit fingerprint definition used by ChemWalker.

    The names follow chemwalker.rwalker.getTanimoto(). Caching is added here
    because mNAP evaluates the same candidate structures across many edges.
    """
    text = normalize_inchi(inchi)
    cache_key = (text, method)
    if cache_key in _FP_CACHE:
        return _FP_CACHE[cache_key]

    mol = mol_from_inchi(text)
    if mol is None:
        _FP_CACHE[cache_key] = None
        return None

    if method == "MFP2-bits":
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, 2)
    elif method == "MFP1-bits":
        fp = rdMolDescriptors.GetHashedMorganFingerprint(mol, 1)
    elif method == "MFP2":
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 2)
    elif method == "MFP1":
        fp = rdMolDescriptors.GetMorganFingerprint(mol, 1)
    elif method == "RDKit7-linear":
        fp = Chem.RDKFingerprint(mol, maxPath=7, branchedPaths=False)
    else:
        # Keep compatibility with any other fingerprint supported by the
        # installed ChemWalker package. This fallback is slower and uncached at
        # the pair level, so the main benchmark should use a pre-specified
        # method implemented above.
        _FP_CACHE[cache_key] = mol
        return mol

    _FP_CACHE[cache_key] = fp
    return fp


def tanimoto(
    inchi1,
    inchi2,
    method: str = FINGERPRINT_METHOD_FOR_CHEMWALKER,
):
    fp1 = get_structural_fingerprint(inchi1, method=method)
    fp2 = get_structural_fingerprint(inchi2, method=method)
    if fp1 is None or fp2 is None:
        return np.nan

    # For unsupported methods the cache stores RDKit Mol objects; delegate to
    # the exact ChemWalker implementation to avoid silently changing the
    # fingerprint definition.
    if isinstance(fp1, Chem.Mol) or isinstance(fp2, Chem.Mol):
        try:
            return float(
                chemwalker_rwalker.getTanimoto(
                    [normalize_inchi(inchi1), normalize_inchi(inchi2)],
                    method,
                )
            )
        except Exception:
            return np.nan

    return float(DataStructs.TanimotoSimilarity(fp1, fp2))


def chemwalker_sigmoid(
    value: float,
    a: float = CHEMWALKER_SIGMOID_A,
    b: float = CHEMWALKER_SIGMOID_B,
):
    return 1.0 / (1.0 + np.exp(a * (value - b)))


def build_mnap_candidate_edges(spectral_edges, full_tlid):
    """Construct weighted candidate-level edges for mNAP.

    For every molecular-network edge, all cross-node candidate pairs are
    compared by structural Tanimoto similarity. Under the default ``sigmoid``
    mode, the edge weight is the sigmoid-transformed product of spectral cosine
    and structural similarity, multiplied by the normalized MetFrag scores of
    both candidates raised to ``MNAP_BETA``.
    """
    candidate_edges = []
    groups = {
        int(node_id): clear_df_attrs(group)
        for node_id, group in full_tlid.groupby("node_id")
    }

    for edge in spectral_edges.itertuples(index=False):
        node1 = int(edge.CLUSTERID1)
        node2 = int(edge.CLUSTERID2)
        cosine = float(edge.Cosine)
        candidates1 = groups.get(node1)
        candidates2 = groups.get(node2)
        if candidates1 is None or candidates2 is None:
            continue

        for candidate1 in candidates1.itertuples(index=False):
            for candidate2 in candidates2.itertuples(index=False):
                structure_similarity = tanimoto(
                    candidate1.InChI,
                    candidate2.InChI,
                    method=FINGERPRINT_METHOD_FOR_CHEMWALKER,
                )
                if pd.isna(structure_similarity):
                    continue

                score1 = max(float(candidate1.score_norm), MNAP_SCORE_EPS)
                score2 = max(float(candidate2.score_norm), MNAP_SCORE_EPS)

                if MNAP_EDGE_MODE == "plain":
                    similarity_term = structure_similarity
                elif MNAP_EDGE_MODE == "cosine":
                    similarity_term = cosine * structure_similarity
                elif MNAP_EDGE_MODE == "sigmoid":
                    similarity_term = chemwalker_sigmoid(
                        cosine * structure_similarity
                    )
                else:
                    raise ValueError(
                        "MNAP_EDGE_MODE must be plain, cosine, or sigmoid."
                    )

                weight = (
                    similarity_term
                    * (score1 ** MNAP_BETA)
                    * (score2 ** MNAP_BETA)
                )
                if weight > 0 and np.isfinite(weight):
                    candidate_edges.append(
                        (candidate1.uid, candidate2.uid, float(weight))
                    )

    return candidate_edges


def to_chemwalker_tlid(full_tlid: pd.DataFrame):
    tlid = clear_df_attrs(full_tlid)
    tlid["cluster index"] = tlid["node_id"].astype(int)
    tlid["Identifier"] = tlid["Identifier"].astype(str)
    tlid["InChI"] = tlid["InChI"].astype(str)
    # The official ChemWalker edge formula expects the normalized MetFrag
    # Score emitted by MetFrag itself. The verified ``Score`` column is already
    # constrained to [0, 1], so it is passed without an additional rescaling.
    tlid["Score"] = pd.to_numeric(
        tlid["Score"], errors="coerce"
    ).fillna(0.0)
    tlid["uid"] = (
        tlid["cluster index"].astype(int).astype(str)
        + "_"
        + tlid["Identifier"].astype(str)
    )
    return tlid


def normalize_cand_pair_edges(raw_edges):
    if raw_edges is None:
        return []
    if isinstance(raw_edges, pd.DataFrame):
        raw_edges = clear_df_attrs(raw_edges)
        columns = list(raw_edges.columns)
        if {"source", "target", "weight"}.issubset(columns):
            return list(
                raw_edges[["source", "target", "weight"]].itertuples(
                    index=False, name=None
                )
            )
        if raw_edges.shape[1] >= 3:
            return list(
                raw_edges.iloc[:, :3].itertuples(index=False, name=None)
            )
        return []

    out = []
    for edge in raw_edges:
        if len(edge) >= 3:
            out.append((edge[0], edge[1], float(edge[2])))
    return out


def run_random_walk(graph: nx.Graph, seed_uids: Sequence[str]):
    valid_seeds = [seed for seed in seed_uids if seed in graph.nodes]
    if not valid_seeds:
        return {}
    probabilities = random_walk(
        graph,
        valid_seeds,
        restart_prob=RESTART_PROB,
    )
    if isinstance(probabilities, dict):
        return probabilities
    nodes = list(graph.nodes())
    return {
        node: float(score)
        for node, score in zip(nodes, probabilities)
    }


def score_and_rank_with_network(full_tlid, probabilities, id_col):
    scored = clear_df_attrs(full_tlid)
    scored["network_score"] = scored["uid"].map(probabilities).fillna(0.0)
    group_max = scored.groupby(id_col)["network_score"].transform("max")
    scored["network_reachable"] = group_max > 0
    scored["network_score_norm"] = np.where(
        group_max > 0,
        scored["network_score"] / group_max,
        0.0,
    )
    scored["network_rank"] = scored.groupby(id_col)[
        "network_score_norm"
    ].rank(ascending=False, method="dense")
    scored.loc[~scored["network_reachable"], "network_rank"] = np.nan
    return scored


def estimate_candidate_pair_count(
    spectral_edges: pd.DataFrame,
    tlid: pd.DataFrame,
) -> int:
    counts = tlid.groupby("cluster index").size().astype(int).to_dict()
    total = 0
    for edge in spectral_edges.itertuples(index=False):
        total += counts.get(int(edge.CLUSTERID1), 0) * counts.get(int(edge.CLUSTERID2), 0)
    return int(total)


def _tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _worker_artifact_paths(repeat_id: int, component_id: int):
    directory = (
        CHEMWALKER_WORK_DIR
        / ranking_signature()[:12]
        / f"repeat_{repeat_id:03d}"
        / f"component_{component_id:04d}"
    )
    directory.mkdir(parents=True, exist_ok=True)
    return {
        "directory": directory,
        "input": directory / "input.pkl",
        "output": directory / "output.pkl",
        "log": directory / "worker.log",
        "audit": directory / "audit.json",
    }


def run_original_chemwalker_worker(
    network: pd.DataFrame,
    tlid: pd.DataFrame,
    seed_uids: Sequence[str],
    repeat_id: int,
    component_id: int,
    fingerprint_method: str,
    force: bool = False,
):
    """Run the unmodified installed ChemWalker code in a safe subprocess.

    The official ``cand_pair`` function constructs a ``multiprocessing.Pool``
    and only implements its returned ``scandpair`` object for ``parallel=True``.
    Calling it directly from a Windows Jupyter kernel can stall during process
    spawning. A standalone worker with a normal ``__main__`` guard preserves
    the original algorithm while providing a safe multiprocessing boundary.
    """
    if not CHEMWALKER_PARALLEL:
        raise ValueError(
            "CHEMWALKER_PARALLEL must remain True: the original cand_pair "
            "implementation has no completed parallel=False return path."
        )
    if not CHEMWALKER_WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"ChemWalker worker is missing: {CHEMWALKER_WORKER_SCRIPT}")
    if CHEMWALKER_NCORES < 1:
        raise ValueError("CHEMWALKER_NCORES must be >= 1.")

    artifacts = _worker_artifact_paths(int(repeat_id), int(component_id))
    if artifacts["output"].exists() and not force:
        result = load_pickle(artifacts["output"])
        return clear_df_attrs(result["scored"]), result.get("audit", {})

    estimated_pairs = estimate_candidate_pair_count(network, tlid)
    payload = {
        "network": clear_df_attrs(network),
        "tlid": clear_df_attrs(tlid),
        "seed_uids": [str(value) for value in seed_uids],
        "fingerprint_method": fingerprint_method,
        "ncores": int(CHEMWALKER_NCORES),
        "restart_probability": float(RESTART_PROB),
        "estimated_candidate_pairs": int(estimated_pairs),
    }
    save_pickle(payload, artifacts["input"])
    if artifacts["output"].exists():
        artifacts["output"].unlink()

    command = [
        sys.executable,
        "-u",
        str(CHEMWALKER_WORKER_SCRIPT),
        "--input",
        str(artifacts["input"]),
        "--output",
        str(artifacts["output"]),
    ]
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")

    print(
        "Original ChemWalker worker started: "
        f"repeat={repeat_id}, component={component_id}, "
        f"spectral_edges={len(network):,}, tlid_rows={len(tlid):,}, "
        f"estimated_candidate_pairs={estimated_pairs:,}, cores={CHEMWALKER_NCORES}"
    )
    print("Worker log:", artifacts["log"])
    started = time.time()
    with artifacts["log"].open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
        )

    heartbeat = max(1, int(CHEMWALKER_HEARTBEAT_SECONDS))
    timeout = int(CHEMWALKER_COMPONENT_TIMEOUT_SECONDS)
    next_message = started + heartbeat
    while process.poll() is None:
        time.sleep(min(2, heartbeat))
        now = time.time()
        if now >= next_message:
            print(
                "  original ChemWalker still running; "
                f"elapsed={(now - started) / 60:.1f} min; "
                f"component={component_id}; candidate_pairs={estimated_pairs:,}"
            )
            next_message = now + heartbeat
        if timeout > 0 and now - started > timeout:
            process.kill()
            process.wait()
            raise TimeoutError(
                f"Original ChemWalker exceeded {timeout} s for component {component_id}.\n"
                + _tail_text(artifacts["log"])
            )

    elapsed = time.time() - started
    if process.returncode != 0 or not artifacts["output"].exists():
        raise RuntimeError(
            "Original ChemWalker worker failed. The benchmark was stopped rather "
            "than silently treating the method as missing.\n"
            f"Return code: {process.returncode}\n"
            f"Log: {artifacts['log']}\n"
            + _tail_text(artifacts["log"])
        )

    result = load_pickle(artifacts["output"])
    audit = result.get("audit", {})
    audit["parent_observed_elapsed_seconds"] = float(elapsed)
    artifacts["audit"].write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=json_safe),
        encoding="utf-8",
    )
    print(
        "Original ChemWalker completed: "
        f"component={component_id}; elapsed={elapsed / 60:.1f} min; "
        f"candidate_edges={audit.get('n_candidate_edges', 'NA')}"
    )
    return clear_df_attrs(result["scored"]), audit


def compute_chemwalker_for_component(
    spectral_edges,
    full_tlid,
    seed_uids: Sequence[str],
    repeat_id: int,
    component_id: int,
    fingerprint_method: str = FINGERPRINT_METHOD_FOR_CHEMWALKER,
    force: bool = False,
):
    """Call the original installed ChemWalker cand_pair and random_walk code."""
    tlid = to_chemwalker_tlid(full_tlid)
    present_nodes = set(tlid["cluster index"].astype(int))
    network = spectral_edges[
        spectral_edges["CLUSTERID1"].astype(int).isin(present_nodes)
        & spectral_edges["CLUSTERID2"].astype(int).isin(present_nodes)
    ].copy()
    if network.empty:
        return pd.DataFrame(), {}

    network["CLUSTERID1"] = network["CLUSTERID1"].astype(int)
    network["CLUSTERID2"] = network["CLUSTERID2"].astype(int)
    network["Cosine"] = pd.to_numeric(
        network["Cosine"], errors="coerce"
    ).fillna(0.0)

    scored, audit = run_original_chemwalker_worker(
        network=network,
        tlid=tlid,
        seed_uids=seed_uids,
        repeat_id=int(repeat_id),
        component_id=int(component_id),
        fingerprint_method=fingerprint_method,
        force=force,
    )
    return scored, audit


def compute_mnap_for_component(spectral_edges, full_tlid):
    """Run mNAP for one connected component using the shared seeds and candidates."""
    present_nodes = set(full_tlid["node_id"].astype(int))
    network = spectral_edges[
        spectral_edges["CLUSTERID1"].astype(int).isin(present_nodes)
        & spectral_edges["CLUSTERID2"].astype(int).isin(present_nodes)
    ].copy()
    if network.empty:
        return pd.DataFrame()

    candidate_edges = build_mnap_candidate_edges(network, full_tlid)
    if not candidate_edges:
        return pd.DataFrame()

    graph = nx.Graph()
    graph.add_weighted_edges_from(candidate_edges)
    seed_uids = full_tlid.loc[
        full_tlid["is_seed"].fillna(False).astype(bool),
        "uid",
    ].astype(str).tolist()
    probabilities = run_random_walk(graph, seed_uids)
    return score_and_rank_with_network(
        full_tlid=full_tlid,
        probabilities=probabilities,
        id_col="node_id",
    )


# =========================================================
# 8. Component evaluation
# =========================================================


def evaluate_component_repeat(
    repeat_id: int,
    component_id: int,
    spectral_edges: pd.DataFrame,
    assignment: pd.DataFrame,
    all_candidates: pd.DataFrame,
    fingerprint_method: str = FINGERPRINT_METHOD_FOR_CHEMWALKER,
    force_chemwalker: bool = False,
):
    """Evaluate MetFrag, ChemWalker, and mNAP on one component and seed repeat."""
    component_table = assignment[
        assignment["ComponentIndex"] == component_id
    ].copy()
    seed_nodes = set(
        component_table.loc[
            component_table["split"] == "seed", "node_id"
        ].astype(int)
    )
    target_nodes = set(
        component_table.loc[
            component_table["split"] == "target", "node_id"
        ].astype(int)
    )
    if not seed_nodes or not target_nodes:
        return []

    # In the published candidate graph, each selected spectral-library match is
    # represented by one verified structure node. Non-seed spectra are replaced
    # by their shared MetFrag candidate lists.
    seed_rows = build_library_match_seed_rows(component_table)
    target_candidates = clear_df_attrs(
        all_candidates[
            all_candidates["node_id"].astype(int).isin(target_nodes)
        ].copy()
    )
    target_candidates["is_seed"] = False
    target_candidates["seed_type"] = ""

    full_tlid = clear_df_attrs(
        pd.concat(
            [clear_df_attrs(seed_rows), target_candidates],
            ignore_index=True,
            sort=False,
        )
    )
    full_tlid["Score"] = pd.to_numeric(
        full_tlid["Score"], errors="coerce"
    ).fillna(0.0)
    full_tlid["score_norm"] = full_tlid.groupby("node_id")[
        "Score"
    ].transform(lambda values: values / values.max() if values.max() > 0 else 0.0)
    full_tlid.loc[full_tlid["is_seed"].fillna(False), "score_norm"] = 1.0
    full_tlid["uid"] = (
        full_tlid["node_id"].astype(int).astype(str)
        + "_"
        + full_tlid["Identifier"].astype(str)
    )
    seed_uids = full_tlid.loc[
        full_tlid["is_seed"].fillna(False).astype(bool), "uid"
    ].astype(str).tolist()

    baseline = {}
    for target_node in sorted(target_nodes):
        truth = component_table.loc[
            component_table["node_id"].astype(int) == target_node,
            "InChIKey1",
        ].iloc[0]
        node_candidates = target_candidates[
            target_candidates["node_id"].astype(int) == target_node
        ].copy()
        if node_candidates.empty:
            baseline[target_node] = {
                "n_candidates": 0,
                "true_found": False,
                "rank": np.nan,
            }
            continue
        node_candidates["rank"] = node_candidates["Score"].rank(
            ascending=False,
            method="dense",
        )
        hit = node_candidates["InChIKey1"].astype(str).eq(str(truth))
        baseline[target_node] = {
            "n_candidates": int(node_candidates.shape[0]),
            "true_found": bool(hit.any()),
            "rank": float(node_candidates.loc[hit, "rank"].min())
            if hit.any()
            else np.nan,
        }

    chemwalker_audit: Dict[str, object] = {}
    try:
        chemwalker_scored, chemwalker_audit = compute_chemwalker_for_component(
            spectral_edges=spectral_edges,
            full_tlid=full_tlid,
            seed_uids=seed_uids,
            repeat_id=int(repeat_id),
            component_id=int(component_id),
            fingerprint_method=fingerprint_method,
            force=force_chemwalker,
        )
    except Exception as exc:
        if FAIL_ON_METHOD_ERROR:
            raise RuntimeError(
                f"Original ChemWalker failed for repeat={repeat_id}, "
                f"component={component_id}."
            ) from exc
        warnings.warn(str(exc))
        chemwalker_scored = pd.DataFrame()

    try:
        mnap_scored = compute_mnap_for_component(
            spectral_edges,
            full_tlid,
        )
    except Exception as exc:
        if FAIL_ON_METHOD_ERROR:
            raise RuntimeError(
                f"mNAP failed for repeat={repeat_id}, component={component_id}."
            ) from exc
        warnings.warn(str(exc))
        mnap_scored = pd.DataFrame()

    rows = []
    for target_node in sorted(target_nodes):
        truth_row = component_table[
            component_table["node_id"].astype(int) == target_node
        ].iloc[0]
        true_key = str(truth_row["InChIKey1"])

        chem_rank = np.nan
        chem_reachable = False
        if not chemwalker_scored.empty:
            subset = chemwalker_scored[
                chemwalker_scored["node_id"].astype(int).eq(target_node)
            ].copy()
            hit = subset["InChIKey1"].astype(str).eq(true_key)
            if hit.any():
                values = pd.to_numeric(
                    subset.loc[hit, "network_rank"], errors="coerce"
                ).dropna()
                if not values.empty:
                    chem_rank = float(values.min())
                    chem_reachable = True

        mnap_rank = np.nan
        mnap_reachable = False
        if not mnap_scored.empty:
            subset = mnap_scored[
                mnap_scored["node_id"].astype(int).eq(target_node)
            ].copy()
            hit = subset["InChIKey1"].astype(str).eq(true_key)
            if hit.any():
                values = pd.to_numeric(
                    subset.loc[hit, "network_rank"], errors="coerce"
                ).dropna()
                if not values.empty:
                    mnap_rank = float(values.min())
                    mnap_reachable = True

        base = baseline[target_node]
        rows.append(
            {
                "repeat_id": int(repeat_id),
                "component_id": int(component_id),
                "target_node": int(target_node),
                "target_name": truth_row["name"],
                "true_inchikey1": true_key,
                "n_component_nodes": int(truth_row["n_component_nodes"]),
                "n_component_seeds": int(truth_row["n_component_seeds"]),
                "n_component_edges": int(spectral_edges.shape[0]),
                "n_target_candidates": int(base["n_candidates"]),
                "true_candidate_found": bool(base["true_found"]),
                "MetFrag_rank": base["rank"],
                "ChemWalker_rank": chem_rank,
                "mNAP_rank": mnap_rank,
                "ChemWalker_reachable": chem_reachable,
                "mNAP_reachable": mnap_reachable,
                "ChemWalker_fingerprint": fingerprint_method,
                "ChemWalker_execution": CHEMWALKER_EXECUTION_MODE,
                "ChemWalker_candidate_edges": chemwalker_audit.get(
                    "n_candidate_edges", np.nan
                ),
                "ChemWalker_estimated_candidate_pairs": chemwalker_audit.get(
                    "estimated_candidate_pairs", np.nan
                ),
                "ChemWalker_seconds": chemwalker_audit.get("total_seconds", np.nan),
            }
        )

    return rows


# =========================================================
# 9. Evaluation and plotting
# =========================================================

METHOD_COLUMNS = {
    "MetFrag": "MetFrag_rank",
    "ChemWalker": "ChemWalker_rank",
    "mNAP": "mNAP_rank",
}


def build_fixed_denominator_rank_long(results: pd.DataFrame):
    eligible = results[results["true_candidate_found"].fillna(False)].copy()
    rows = []
    for result in eligible.itertuples(index=False):
        failure_rank = int(result.n_target_candidates) + 1
        for method, column in METHOD_COLUMNS.items():
            raw_rank = getattr(result, column)
            finite = pd.notna(raw_rank) and np.isfinite(float(raw_rank))
            rows.append(
                {
                    "repeat_id": int(result.repeat_id),
                    "component_id": int(result.component_id),
                    "target_node": int(result.target_node),
                    "method": method,
                    "rank": float(raw_rank) if finite else np.inf,
                    "ranked": bool(finite),
                    "penalized_rank": float(raw_rank) if finite else failure_rank,
                    "n_candidates": int(result.n_target_candidates),
                }
            )
    return pd.DataFrame(rows)


def summarize_fixed_denominator(rank_long: pd.DataFrame):
    rows = []
    for method, group in rank_long.groupby("method"):
        ranks = pd.to_numeric(group["rank"], errors="coerce")
        ranked = np.isfinite(ranks)
        reciprocal = np.where(ranked, 1.0 / ranks, 0.0)
        row = {
            "method": method,
            "n_total": int(group.shape[0]),
            "n_ranked": int(ranked.sum()),
            "ranking_coverage": float(ranked.mean()),
            "Top1": float(np.mean(ranks <= 1)),
            "Top5": float(np.mean(ranks <= 5)),
            "Top10": float(np.mean(ranks <= 10)),
            "Top20": float(np.mean(ranks <= 20)),
            "MRR_fixed_denominator": float(np.mean(reciprocal)),
            "mean_penalized_rank": float(group["penalized_rank"].mean()),
            "median_penalized_rank": float(group["penalized_rank"].median()),
            "mean_rank_among_ranked": float(ranks[ranked].mean())
            if ranked.any()
            else np.nan,
            "median_rank_among_ranked": float(ranks[ranked].median())
            if ranked.any()
            else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def common_rank_table(results: pd.DataFrame):
    eligible = results[results["true_candidate_found"].fillna(False)].copy()
    for column in METHOD_COLUMNS.values():
        eligible[column] = pd.to_numeric(eligible[column], errors="coerce")
    return eligible.dropna(subset=list(METHOD_COLUMNS.values())).copy()


def pairwise_statistics(results: pd.DataFrame, primary_k: int):
    eligible = results[results["true_candidate_found"].fillna(False)].copy()
    comparisons = [
        ("ChemWalker", "MetFrag"),
        ("mNAP", "MetFrag"),
        ("mNAP", "ChemWalker"),
    ]
    rows = []

    for method_a, method_b in comparisons:
        col_a = METHOD_COLUMNS[method_a]
        col_b = METHOD_COLUMNS[method_b]
        ranks_a = pd.to_numeric(eligible[col_a], errors="coerce")
        ranks_b = pd.to_numeric(eligible[col_b], errors="coerce")

        success_a = ranks_a.le(primary_k).fillna(False)
        success_b = ranks_b.le(primary_k).fillna(False)
        a_only = int((success_a & ~success_b).sum())
        b_only = int((~success_a & success_b).sum())
        both = int((success_a & success_b).sum())
        neither = int((~success_a & ~success_b).sum())

        exact_p = np.nan
        if binomtest is not None and (a_only + b_only) > 0:
            exact_p = float(
                binomtest(
                    min(a_only, b_only),
                    n=a_only + b_only,
                    p=0.5,
                    alternative="two-sided",
                ).pvalue
            )

        common = eligible.loc[ranks_a.notna() & ranks_b.notna()].copy()
        delta = (
            pd.to_numeric(common[col_a], errors="coerce")
            - pd.to_numeric(common[col_b], errors="coerce")
        )
        wilcoxon_p = np.nan
        if wilcoxon is not None and common.shape[0] > 0 and (delta != 0).any():
            try:
                wilcoxon_p = float(
                    wilcoxon(
                        pd.to_numeric(common[col_a], errors="coerce"),
                        pd.to_numeric(common[col_b], errors="coerce"),
                        zero_method="wilcox",
                        alternative="two-sided",
                    ).pvalue
                )
            except Exception:
                wilcoxon_p = np.nan

        rows.append(
            {
                "comparison": f"{method_a} vs {method_b}",
                "n_fixed_denominator": int(eligible.shape[0]),
                f"{method_a}_Top{primary_k}_only": a_only,
                f"{method_b}_Top{primary_k}_only": b_only,
                "both_success": both,
                "neither_success": neither,
                "paired_exact_p": exact_p,
                "n_common_finite_rank": int(common.shape[0]),
                f"{method_a}_better_rank": int((delta < 0).sum()),
                f"{method_b}_better_rank": int((delta > 0).sum()),
                "rank_tie": int((delta == 0).sum()),
                f"mean_rank_delta_{method_a}_minus_{method_b}": float(delta.mean())
                if not delta.empty
                else np.nan,
                "wilcoxon_p": wilcoxon_p,
            }
        )

    return pd.DataFrame(rows)


def bootstrap_topk_differences(
    results: pd.DataFrame,
    primary_k: int,
    iterations: int,
):
    """Component-cluster bootstrap for paired Top-k accuracy differences.

    All seed repeats and target nodes belonging to the same molecular-network
    component are kept together. Components, rather than repeat-component rows,
    are resampled to avoid treating repeated seed allocations as independent
    chemical families.
    """
    eligible = results[results["true_candidate_found"].fillna(False)].copy()
    units = sorted(eligible["component_id"].astype(int).unique().tolist())
    if not units:
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_SEED)
    comparisons = [
        ("ChemWalker", "MetFrag"),
        ("mNAP", "MetFrag"),
        ("mNAP", "ChemWalker"),
    ]
    values = {comparison: [] for comparison in comparisons}
    grouped = {
        unit: eligible[eligible["component_id"].astype(int) == unit].copy()
        for unit in units
    }

    for _ in range(iterations):
        sampled_units = rng.choice(units, size=len(units), replace=True)
        sampled = pd.concat(
            [grouped[int(unit)] for unit in sampled_units],
            ignore_index=True,
        )
        for method_a, method_b in comparisons:
            success_a = pd.to_numeric(
                sampled[METHOD_COLUMNS[method_a]], errors="coerce"
            ).le(primary_k).fillna(False)
            success_b = pd.to_numeric(
                sampled[METHOD_COLUMNS[method_b]], errors="coerce"
            ).le(primary_k).fillna(False)
            values[(method_a, method_b)].append(
                float(success_a.mean() - success_b.mean())
            )

    rows = []
    for (method_a, method_b), difference in values.items():
        array = np.asarray(difference, dtype=float)
        rows.append(
            {
                "comparison": f"{method_a} minus {method_b}",
                f"Top{primary_k}_difference_mean": float(array.mean()),
                "CI_2.5%": float(np.quantile(array, 0.025)),
                "CI_97.5%": float(np.quantile(array, 0.975)),
                "bootstrap_iterations": int(iterations),
                "cluster_unit": "component_id (all seed repeats kept together)",
            }
        )
    return pd.DataFrame(rows)


def plot_topk_fixed(rank_long: pd.DataFrame, output_stem: Path, max_k: int):
    curve_rows = []
    plt.figure(figsize=(7, 5))
    for method in ["MetFrag", "ChemWalker", "mNAP"]:
        group = rank_long[rank_long["method"] == method]
        if group.empty:
            continue
        ranks = pd.to_numeric(group["rank"], errors="coerce")
        x_values = list(range(1, max_k + 1))
        y_values = [float(np.mean(ranks <= cutoff)) for cutoff in x_values]
        for cutoff, accuracy in zip(x_values, y_values):
            curve_rows.append(
                {
                    "method": method,
                    "k": cutoff,
                    "topk_accuracy_fixed_denominator": accuracy,
                }
            )
        plt.plot(x_values, y_values, marker="o", label=method)

    plt.xlabel("Rank cutoff k")
    plt.ylabel("Fraction of correct structures within top k")
    plt.title("NIST benchmark: candidate re-ranking")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.show()
    return pd.DataFrame(curve_rows)



def _refresh_output_paths() -> None:
    """Rebuild all output/checkpoint paths after OUT_DIR changes in a notebook."""
    global CHECKPOINT_DIR, CACHE_DIR, AUDIT_DIR, REPEAT_DIR
    global CHEMWALKER_WORK_DIR, SMOKE_DIR
    global STEP1_CHECKPOINT, STEP2_CANDIDATES_PKL, STEP2_CANDIDATES_CSV

    CHECKPOINT_DIR = OUT_DIR / "checkpoints"
    CACHE_DIR = OUT_DIR / "candidate_cache"
    AUDIT_DIR = OUT_DIR / "metfrag_score_audit"
    REPEAT_DIR = OUT_DIR / "repeat_results"
    CHEMWALKER_WORK_DIR = OUT_DIR / "chemwalker_worker"
    SMOKE_DIR = OUT_DIR / "smoke_tests"
    for directory in [
        OUT_DIR, CHECKPOINT_DIR, CACHE_DIR, AUDIT_DIR, REPEAT_DIR,
        CHEMWALKER_WORK_DIR, SMOKE_DIR,
    ]:
        directory.mkdir(exist_ok=True, parents=True)
    STEP1_CHECKPOINT = CHECKPOINT_DIR / "step1_network.pkl"
    STEP2_CANDIDATES_PKL = CHECKPOINT_DIR / "step2_all_candidates.pkl"
    STEP2_CANDIDATES_CSV = OUT_DIR / "all_metfrag_candidates.csv"


def configure_paths(
    input_msp: Optional[Path] = None,
    benchmark_table: Optional[Path] = None,
    pubchemlite_path: Optional[Path] = None,
    metfrag_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """Update file paths safely from a Jupyter configuration cell."""
    global INPUT_MSP, BENCHMARK_TABLE, PUBCHEMLITE_PATH, METFRAG_PATH, OUT_DIR
    if input_msp is not None:
        INPUT_MSP = Path(input_msp)
    if benchmark_table is not None:
        BENCHMARK_TABLE = Path(benchmark_table)
    if pubchemlite_path is not None:
        PUBCHEMLITE_PATH = Path(pubchemlite_path)
    if metfrag_path is not None:
        METFRAG_PATH = Path(metfrag_path)
    if output_dir is not None:
        OUT_DIR = Path(output_dir)
    _refresh_output_paths()
    return {
        "INPUT_MSP": str(INPUT_MSP),
        "BENCHMARK_TABLE": str(BENCHMARK_TABLE),
        "PUBCHEMLITE_PATH": str(PUBCHEMLITE_PATH),
        "METFRAG_PATH": str(METFRAG_PATH),
        "OUT_DIR": str(OUT_DIR),
    }


# =========================================================
# 10. Stepwise workflow
# =========================================================


def step_0_validate():
    print("Release identifier:", RELEASE_ID)
    print("Python:", sys.version)
    print("Input MSP:", INPUT_MSP, INPUT_MSP.exists())
    print("Benchmark table:", BENCHMARK_TABLE, BENCHMARK_TABLE.exists())
    print("PubChemLite:", PUBCHEMLITE_PATH, PUBCHEMLITE_PATH.exists())
    print("MetFrag JAR:", METFRAG_PATH, METFRAG_PATH.exists())
    print("Output directory:", OUT_DIR.resolve())
    print("\nConfiguration:")
    print(json.dumps(config_snapshot(), indent=2, ensure_ascii=False))
    print("\nChemWalker functions:")
    print("cand_pair module:", inspect.getmodule(cand_pair))
    print("cand_pair signature:", inspect.signature(cand_pair))
    print("random_walk signature:", inspect.signature(random_walk))
    print("cand_pair source audit:", json.dumps(callable_source_audit(cand_pair), indent=2))
    print("random_walk source audit:", json.dumps(callable_source_audit(random_walk), indent=2))
    print("Worker script:", CHEMWALKER_WORKER_SCRIPT, CHEMWALKER_WORKER_SCRIPT.exists())
    if not CHEMWALKER_PARALLEL:
        raise ValueError("CHEMWALKER_PARALLEL must be True for the original cand_pair implementation.")
    if CHEMWALKER_NCORES < 1:
        raise ValueError("CHEMWALKER_NCORES must be >= 1.")
    if hasattr(chemwalker_rwalker, "sigmoid"):
        print("ChemWalker sigmoid signature:", inspect.signature(chemwalker_rwalker.sigmoid))
    if hasattr(chemwalker_rwalker, "sc"):
        print("ChemWalker sc signature:", inspect.signature(chemwalker_rwalker.sc))
    print("Network signature:", network_signature())
    print("Candidate signature:", candidate_signature())
    print("Ranking signature:", ranking_signature())

    # The published ChemWalker implementation uses sigmoid(a=-9, b=0.6) and
    # alpha=0.3. Warn if the installed package differs from this benchmark.
    try:
        sigmoid_signature = inspect.signature(chemwalker_rwalker.sigmoid)
        sigmoid_b = sigmoid_signature.parameters["b"].default
        sc_signature = inspect.signature(chemwalker_rwalker.sc)
        sc_alpha = sc_signature.parameters["alpha"].default
        if float(sigmoid_b) != 0.6 or float(sc_alpha) != 0.3:
            warnings.warn(
                "Installed ChemWalker defaults differ from the published benchmark: "
                f"sigmoid b={sigmoid_b}, alpha={sc_alpha}."
            )
    except Exception:
        warnings.warn("Could not verify ChemWalker sigmoid/alpha defaults.")

    missing = [
        path
        for path in [
            INPUT_MSP, BENCHMARK_TABLE, PUBCHEMLITE_PATH, METFRAG_PATH,
            CHEMWALKER_WORKER_SCRIPT,
        ]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing required paths: {missing}")

    manifest = {
        "release_id": RELEASE_ID,
        "configuration": config_snapshot(),
        "network_signature": network_signature(),
        "candidate_signature": candidate_signature(),
        "ranking_signature": ranking_signature(),
        "input_msp": _path_identity(INPUT_MSP),
        "benchmark_table": _path_identity(BENCHMARK_TABLE),
        "pubchemlite": _path_identity(PUBCHEMLITE_PATH),
        "metfrag_jar": _path_identity(METFRAG_PATH),
        "chemwalker_cand_pair": callable_source_audit(cand_pair),
        "chemwalker_random_walk": callable_source_audit(random_walk),
        "chemwalker_worker": _path_identity(CHEMWALKER_WORKER_SCRIPT),
    }
    (OUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return manifest


def step_1_prepare_network(force: bool = False):
    expected_signature = network_signature()
    if STEP1_CHECKPOINT.exists() and not force:
        checkpoint = load_pickle(STEP1_CHECKPOINT)
        observed_signature = checkpoint.get("network_signature")
        if observed_signature != expected_signature:
            raise ValueError(
                "The saved network checkpoint was produced with different input or "
                "network parameters. Re-run run_step(1, force=True)."
            )
        print("Loaded Step 1 checkpoint:", STEP1_CHECKPOINT)
        print("Network signature:", expected_signature[:12])
        print("Network nodes:", checkpoint["nodes"].shape[0])
        print("Network edges:", checkpoint["edges"].shape[0])
        return checkpoint

    print("Step 1A. Read MSP and align it to the published benchmark table")
    spectra_all, table_all = load_annotated_msp(INPUT_MSP, BENCHMARK_TABLE)
    table_all.to_csv(OUT_DIR / "all_annotated_spectra_table.csv", index=False)

    print("\nStep 1B. Select one representative spectrum per 2D structure")
    spectra_rep, table_rep = choose_representative_spectra(
        spectra_all,
        table_all,
    )

    print("\nStep 1C. Deterministic benchmark sampling")
    spectra_sampled, table_sampled = sample_compounds(
        spectra_rep,
        table_rep,
        n_compounds=N_COMPOUNDS_TO_SAMPLE,
        random_seed=RANDOM_SEED,
    )
    table_sampled.to_csv(OUT_DIR / "sampled_spectra_table.csv", index=False)

    print("\nStep 1D. Build modified-cosine molecular network")
    edges = compute_modified_cosine_edges(spectra_sampled)
    edges = apply_mutual_topk(edges, TOPK)
    edges = add_component_index(edges)
    if edges.empty:
        raise ValueError(
            "No molecular-network edges were retained. Check MIN_COSINE, "
            "MIN_MATCHED_PEAKS, and the selected spectra."
        )

    nodes = attach_components_to_nodes(table_sampled, edges)
    network_node_ids = set(nodes["node_id"].astype(int))
    spectra_network = [
        spectrum
        for spectrum in spectra_sampled
        if int(meta_get(spectrum.metadata, "node_id")) in network_node_ids
    ]

    edges.to_csv(
        OUT_DIR / "modified_cosine_network_edges.tsv",
        sep="\t",
        index=False,
    )
    low_edges = edges[
        edges["Cosine"].astype(float) < LOW_SIMILARITY_UPPER_COSINE
    ].copy()
    low_edges.to_csv(
        OUT_DIR / "network_edges_cosine_0.60_to_0.70.tsv",
        sep="\t",
        index=False,
    )
    nodes.to_csv(OUT_DIR / "benchmark_nodes_in_network.csv", index=False)

    sizes = component_size_table(edges)
    sizes.to_csv(OUT_DIR / "component_size_summary.csv", index=False)
    print("Network nodes:", nodes.shape[0])
    print("Network edges:", edges.shape[0])
    print("Connected components:", sizes.shape[0])
    show_df(sizes, n=10)

    checkpoint = {
        "spectra": spectra_network,
        "nodes": nodes,
        "edges": edges,
        "component_sizes": sizes,
        "config": config_snapshot(),
        "network_signature": expected_signature,
    }
    save_pickle(checkpoint, STEP1_CHECKPOINT)
    print("Saved Step 1 checkpoint:", STEP1_CHECKPOINT)
    return checkpoint


def step_2_generate_metfrag_candidates(
    force: bool = False,
    max_nodes: Optional[int] = None,
):
    global _METFRAG_AUDIT_COUNTER
    _METFRAG_AUDIT_COUNTER = 0

    checkpoint = step_1_prepare_network(force=False)
    spectra = checkpoint["spectra"]
    nodes = checkpoint["nodes"].copy()
    spectra_by_node = {
        int(meta_get(spectrum.metadata, "node_id")): spectrum
        for spectrum in spectra
    }

    if force and max_nodes is None:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if STEP2_CANDIDATES_PKL.exists():
            STEP2_CANDIDATES_PKL.unlink()

    db = standardize_pubchemlite_db(PUBCHEMLITE_PATH)
    node_ids = sorted(nodes["node_id"].astype(int).tolist())
    if max_nodes is not None:
        node_ids = node_ids[: int(max_nodes)]

    candidate_frames = []
    status_rows = []
    for index, node_id in enumerate(node_ids, start=1):
        if index == 1 or index % 25 == 0:
            print(f"MetFrag node {index}/{len(node_ids)}: node_id={node_id}")
        truth = nodes[nodes["node_id"].astype(int) == node_id].iloc[0]
        spectrum = spectra_by_node[node_id]
        candidates = generate_candidates_for_node(
            spectrum,
            truth,
            db,
            force=force and max_nodes is not None,
        )
        candidates = clear_df_attrs(candidates)
        if not candidates.empty:
            candidate_frames.append(candidates)
        true_found = (
            not candidates.empty
            and candidates["InChIKey1"]
            .astype(str)
            .eq(str(truth["InChIKey1"]))
            .any()
        )
        status_rows.append(
            {
                "node_id": node_id,
                "n_candidates": int(candidates.shape[0]),
                "true_candidate_found": bool(true_found),
            }
        )

    status = pd.DataFrame(status_rows)
    status.to_csv(OUT_DIR / "metfrag_candidate_status.csv", index=False)
    print("Candidate status:")
    print(status.describe(include="all"))

    if max_nodes is not None:
        print(
            "Small MetFrag test completed. The full Step 2 checkpoint was not overwritten."
        )
        return pd.concat(candidate_frames, ignore_index=True) if candidate_frames else empty_candidate_table()

    all_candidates = (
        clear_df_attrs(pd.concat(candidate_frames, ignore_index=True, sort=False))
        if candidate_frames
        else empty_candidate_table()
    )
    save_pickle(
        {
            "candidate_signature": candidate_signature(),
            "candidates": all_candidates,
        },
        STEP2_CANDIDATES_PKL,
    )
    all_candidates.to_csv(STEP2_CANDIDATES_CSV, index=False)
    print("Saved all candidates:", STEP2_CANDIDATES_PKL)
    print("Candidate signature:", candidate_signature()[:12])
    print("Candidate rows:", all_candidates.shape[0])
    print(
        "Unique nodes with candidates:",
        all_candidates["node_id"].nunique() if not all_candidates.empty else 0,
    )
    return all_candidates


def _component_result_path(repeat_id: int, component_id: int):
    directory = repeat_output_dir() / f"repeat_{repeat_id:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"component_{component_id:04d}.csv"


def _repeat_result_path(repeat_id: int) -> Path:
    return repeat_output_dir() / f"benchmark_repeat_{repeat_id:03d}.csv"


def choose_smoke_component(max_nodes: int = 10) -> int:
    """Choose a small nontrivial component for a quick official-code smoke test."""
    checkpoint = step_1_prepare_network(force=False)
    sizes = checkpoint["component_sizes"].copy()
    eligible = sizes[(sizes["n_nodes"] >= 2) & (sizes["n_nodes"] <= max_nodes)].copy()
    if eligible.empty:
        eligible = sizes[sizes["n_nodes"] >= 2].copy()
    eligible = eligible.sort_values(["n_nodes", "n_edges", "ComponentIndex"])
    return int(eligible.iloc[0]["ComponentIndex"])


def _smoke_result_path(repeat_id: int, component_id: int) -> Path:
    directory = SMOKE_DIR / ranking_signature()[:12]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"repeat_{repeat_id:03d}_component_{component_id:04d}.csv"


def step_3_run_benchmark(
    repeat_ids: Optional[Sequence[int]] = None,
    component_ids: Optional[Sequence[int]] = None,
    force: bool = False,
):
    checkpoint = step_1_prepare_network(force=False)
    if not STEP2_CANDIDATES_PKL.exists():
        raise FileNotFoundError(
            "Step 2 candidates are missing. Run run_step(2) first."
        )

    nodes = checkpoint["nodes"].copy()
    edges = checkpoint["edges"].copy()
    candidate_checkpoint = load_pickle(STEP2_CANDIDATES_PKL)
    if not isinstance(candidate_checkpoint, dict):
        raise ValueError(
            "The candidate checkpoint uses an incompatible schema. "
            "Re-run run_step(2, force=True)."
        )
    if candidate_checkpoint.get("candidate_signature") != candidate_signature():
        raise ValueError(
            "The candidate checkpoint does not match the current database, "
            "MetFrag, or candidate parameters. Re-run run_step(2, force=True)."
        )
    all_candidates = clear_df_attrs(candidate_checkpoint["candidates"])

    if repeat_ids is None:
        repeat_ids = list(range(N_SEED_REPEATS))

    all_component_ids = sorted(nodes["ComponentIndex"].astype(int).unique().tolist())
    partial_run = component_ids is not None
    if component_ids is None:
        selected_components = all_component_ids
    else:
        selected_components = [int(value) for value in component_ids]
        unknown = sorted(set(selected_components) - set(all_component_ids))
        if unknown:
            raise ValueError(f"Unknown component IDs: {unknown}")
        print(
            "Partial/smoke run: formal repeat result files will not be overwritten. "
            f"Selected components: {selected_components}"
        )

    all_results = []
    for repeat_id in repeat_ids:
        print("\n" + "=" * 90)
        print(f"Seed repeat {repeat_id}")
        assignment = assign_component_seeds(nodes, repeat_id=int(repeat_id))
        assignment_path = repeat_output_dir() / f"seed_assignment_repeat_{repeat_id:03d}.csv"
        assignment.to_csv(assignment_path, index=False)
        print(assignment["split"].value_counts())

        repeat_records = []
        for index, component_id in enumerate(selected_components, start=1):
            output_path = (
                _smoke_result_path(int(repeat_id), int(component_id))
                if partial_run
                else _component_result_path(int(repeat_id), int(component_id))
            )
            if output_path.exists() and not force:
                component_result = pd.read_csv(output_path)
                if not component_result.empty:
                    repeat_records.append(component_result)
                continue

            component_edges = edges[
                edges["ComponentIndex"].astype(int) == int(component_id)
            ].copy()
            component_node_count = assignment[
                assignment["ComponentIndex"] == component_id
            ].shape[0]
            print(
                f"Repeat {repeat_id}; component {index}/{len(selected_components)}; "
                f"component_id={component_id}; nodes={component_node_count}; "
                f"edges={component_edges.shape[0]}"
            )

            records = evaluate_component_repeat(
                repeat_id=int(repeat_id),
                component_id=int(component_id),
                spectral_edges=component_edges,
                assignment=assignment,
                all_candidates=all_candidates,
                fingerprint_method=FINGERPRINT_METHOD_FOR_CHEMWALKER,
                force_chemwalker=force,
            )
            component_result = pd.DataFrame(records)
            component_result.to_csv(output_path, index=False)
            if not component_result.empty:
                repeat_records.append(component_result)

        repeat_result = (
            pd.concat(repeat_records, ignore_index=True, sort=False)
            if repeat_records
            else pd.DataFrame()
        )
        if not partial_run:
            repeat_result.to_csv(
                _repeat_result_path(int(repeat_id)),
                index=False,
            )
        if not repeat_result.empty:
            all_results.append(repeat_result)

    combined = (
        pd.concat(all_results, ignore_index=True, sort=False)
        if all_results
        else pd.DataFrame()
    )
    if not combined.empty:
        if partial_run:
            combined.to_csv(
                SMOKE_DIR / f"smoke_combined_{ranking_signature()[:12]}.csv",
                index=False,
            )
        else:
            combined.to_csv(OUT_DIR / "benchmark_raw_results.csv", index=False)
        print("Combined benchmark rows:", combined.shape[0])
    return combined


def step_4_evaluate():
    result_files = sorted(repeat_output_dir().glob("benchmark_repeat_*.csv"))
    if not result_files:
        raise FileNotFoundError(
            "No repeat result files found. Run Step 3 first."
        )
    results = pd.concat(
        [pd.read_csv(path) for path in result_files],
        ignore_index=True,
        sort=False,
    )
    results.to_csv(OUT_DIR / "benchmark_raw_results.csv", index=False)

    unique_targets = results[
        ["target_node", "true_candidate_found", "n_target_candidates"]
    ].drop_duplicates("target_node")
    coverage = pd.DataFrame(
        [
            {
                "n_unique_target_nodes": int(unique_targets.shape[0]),
                "n_unique_targets_with_true_candidate": int(
                    unique_targets["true_candidate_found"].fillna(False).sum()
                ),
                "candidate_coverage": float(
                    unique_targets["true_candidate_found"].fillna(False).mean()
                ),
                "candidate_cap": TOP_N_CANDIDATES_PER_NODE,
                "mass_tolerance_ppm": PPM,
            }
        ]
    )
    coverage.to_csv(OUT_DIR / "candidate_coverage_summary.csv", index=False)

    rank_long = build_fixed_denominator_rank_long(results)
    rank_long.to_csv(OUT_DIR / "rank_long_fixed_denominator.csv", index=False)

    summary = summarize_fixed_denominator(rank_long)
    summary.to_csv(OUT_DIR / "benchmark_summary_fixed_denominator.csv", index=False)
    print("\nFixed-denominator summary:")
    show_df(summary, n=20)

    common = common_rank_table(results)
    common.to_csv(OUT_DIR / "benchmark_common_finite_ranks.csv", index=False)

    pairwise = pairwise_statistics(results, primary_k=PRIMARY_TOP_K)
    pairwise["analysis_scope"] = "all seed repeats; descriptive paired instances"
    pairwise.to_csv(OUT_DIR / "pairwise_statistics_all_repeats.csv", index=False)
    print("\nPairwise statistics across all seed repeats (descriptive):")
    show_df(pairwise, n=20)

    first_repeat_id = int(results["repeat_id"].min())
    first_repeat_results = results[results["repeat_id"].astype(int) == first_repeat_id].copy()
    pairwise_first = pairwise_statistics(
        first_repeat_results, primary_k=PRIMARY_TOP_K
    )
    pairwise_first["analysis_scope"] = f"single pre-specified seed repeat {first_repeat_id}"
    pairwise_first.to_csv(
        OUT_DIR / "pairwise_statistics_first_repeat.csv", index=False
    )

    per_repeat_rows = []
    for repeat_id, repeat_results in results.groupby("repeat_id"):
        repeat_long = build_fixed_denominator_rank_long(repeat_results)
        repeat_summary = summarize_fixed_denominator(repeat_long)
        repeat_summary.insert(0, "repeat_id", int(repeat_id))
        per_repeat_rows.append(repeat_summary)
    summary_by_repeat = (
        pd.concat(per_repeat_rows, ignore_index=True)
        if per_repeat_rows else pd.DataFrame()
    )
    summary_by_repeat.to_csv(
        OUT_DIR / "benchmark_summary_by_seed_repeat.csv", index=False
    )

    bootstrap = bootstrap_topk_differences(
        results,
        primary_k=PRIMARY_TOP_K,
        iterations=BOOTSTRAP_ITERATIONS,
    )
    bootstrap.to_csv(OUT_DIR / "component_bootstrap_topk.csv", index=False)

    curve = plot_topk_fixed(
        rank_long,
        output_stem=OUT_DIR / "topk_curve_fixed_denominator",
        max_k=MAX_K_FOR_PLOT,
    )
    curve.to_csv(OUT_DIR / "topk_curve_fixed_denominator.csv", index=False)

    first_repeat = rank_long[rank_long["repeat_id"].astype(int) == first_repeat_id]
    if not first_repeat.empty:
        first_curve = plot_topk_fixed(
            first_repeat,
            output_stem=OUT_DIR / "topk_curve_first_seed_repeat",
            max_k=MAX_K_FOR_PLOT,
        )
        first_curve.to_csv(
            OUT_DIR / "topk_curve_first_seed_repeat.csv",
            index=False,
        )

    print("\nCandidate coverage:")
    show_df(coverage, n=5)
    print("\nOutputs saved to:", OUT_DIR.resolve())
    return {
        "results": results,
        "coverage": coverage,
        "rank_long": rank_long,
        "summary": summary,
        "common": common,
        "pairwise_all_repeats": pairwise,
        "pairwise_first_repeat": pairwise_first,
        "summary_by_repeat": summary_by_repeat,
        "bootstrap": bootstrap,
        "curve": curve,
    }


def run_step(step_number: int, **kwargs):
    steps = {
        0: step_0_validate,
        1: step_1_prepare_network,
        2: step_2_generate_metfrag_candidates,
        3: step_3_run_benchmark,
        4: step_4_evaluate,
    }
    if step_number not in steps:
        raise ValueError(f"Valid steps: {sorted(steps)}")
    return steps[step_number](**kwargs)


def main():
    step_0_validate()
    step_1_prepare_network(force=False)
    step_2_generate_metfrag_candidates(force=False)
    step_3_run_benchmark(force=False)
    step_4_evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Stepwise MetFrag/ChemWalker/mNAP benchmark using the NIST MSP dataset."
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[0, 1, 2, 3, 4],
        help="Run one workflow step. Omit to print usage only.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-nodes", type=int, default=None)
    parser.add_argument(
        "--repeat-ids",
        type=str,
        default=None,
        help="Comma-separated seed repeat IDs, e.g. 0,1,2",
    )
    parser.add_argument(
        "--component-ids", type=str, default=None,
        help="Comma-separated component IDs for a partial smoke run",
    )
    parser.add_argument("--all", action="store_true", help="Run all steps")
    args = parser.parse_args()

    if args.all:
        main()
    elif args.step is None:
        print(__doc__)
        print("\nExample: python benchmark_nist.py --step 0")
    elif args.step == 1:
        run_step(1, force=args.force)
    elif args.step == 2:
        run_step(2, force=args.force, max_nodes=args.max_nodes)
    elif args.step == 3:
        repeats = (
            [int(value) for value in args.repeat_ids.split(",")]
            if args.repeat_ids
            else None
        )
        components = (
            [int(value) for value in args.component_ids.split(",")]
            if args.component_ids else None
        )
        run_step(
            3, force=args.force, repeat_ids=repeats, component_ids=components
        )
    else:
        run_step(args.step)
