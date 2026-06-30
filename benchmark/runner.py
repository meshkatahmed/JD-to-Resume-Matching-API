"""
benchmark/runner.py
-------------------
Core benchmark runner for the JD-to-Resume Matching API.

Loads `dataset_imputed.xlsx`, passes every (Job Description, Resume) pair
through every requested scorer from the registry, and returns a structured
DataFrame with all system scores alongside the ground-truth Score(%).

To add a new scoring method, see benchmark/scorers.py — no changes needed here.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so service imports resolve from
# any working directory.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import registry (scorers.py self-registers all built-ins on import)
from benchmark.scorers import REGISTRY, available_methods
from services.text_extraction_engine import TextExtractionEngine
from services.text_processing_engine import TextProcessingEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET_PATH = _PROJECT_ROOT / "datasets" / "dataset_imputed.xlsx"

GT_COL     = "Score(%)"        # ground-truth column name in the dataset
JD_COL     = "Job Description"
RESUME_COL = "Resume"


def score_col(method: str) -> str:
    """Return the DataFrame column name for a given method's scores."""
    return f"system_score_{method}_%"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_path: Path = DATASET_PATH,
    methods: list[str] = ("cbow", "tfidf"),
    max_rows: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the benchmark against every row in the dataset.

    Parameters
    ----------
    dataset_path : Path
        Path to `dataset_imputed.xlsx`.
    methods : list of str
        Which scoring methods to evaluate. Must be keys in the scorer registry.
        Use ``available_methods()`` to see what's registered.
    max_rows : int or None
        If set, only process the first *max_rows* rows (useful for quick tests).
    verbose : bool
        Print a progress bar / ETA to stdout.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with one extra column per method:
        ``system_score_<method>_%`` — score as integer percentage (0-100).
    """
    # Validate requested methods against the registry
    unknown = [m for m in methods if m not in REGISTRY]
    if unknown:
        raise ValueError(
            f"Unknown scoring method(s): {unknown}. "
            f"Registered methods: {available_methods()}"
        )

    df = pd.read_excel(dataset_path)
    if max_rows is not None:
        df = df.head(max_rows).copy()

    total = len(df)

    extraction_engine = TextExtractionEngine()
    processing_engine = TextProcessingEngine()

    # Accumulate scores per method
    scores: dict[str, list[int]] = {m: [] for m in methods}

    t_start = time.perf_counter()

    for idx, row in df.iterrows():
        jd_text     = str(row[JD_COL])
        resume_text = str(row[RESUME_COL])

        for method in methods:
            scorer = REGISTRY[method]
            raw = scorer(jd_text, resume_text, extraction_engine, processing_engine)
            scores[method].append(round(raw * 100))

        if verbose and ((idx + 1) % 50 == 0 or (idx + 1) == total):  # type: ignore[operator]
            elapsed = time.perf_counter() - t_start
            rate    = (idx + 1) / elapsed  # type: ignore[operator]
            eta     = (total - (idx + 1)) / rate if rate > 0 else 0  # type: ignore[operator]
            print(
                f"  [{idx + 1:>5}/{total}]  "  # type: ignore[operator]
                f"elapsed={elapsed:6.1f}s  "
                f"rate={rate:5.1f} rows/s  "
                f"ETA={eta:6.1f}s",
                flush=True,
            )

    for method in methods:
        df[score_col(method)] = scores[method]

    return df
