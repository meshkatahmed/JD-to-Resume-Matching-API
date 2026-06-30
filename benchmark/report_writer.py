"""
benchmark/report_writer.py
--------------------------
Saves benchmark results to disk.

Outputs
~~~~~~~
1. ``benchmark_results/raw_scores.xlsx``   — full DataFrame with all system
   scores alongside ground-truth; useful for manual inspection or further analysis.
2. ``benchmark_results/accuracy_report.txt`` — human-readable metrics report.
3. ``benchmark_results/metrics_summary.xlsx`` — metrics table for both methods
   side-by-side; easy to share or open in Excel.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "benchmark_results"


def save_all(
    results_df: pd.DataFrame,
    metrics_list: list[dict],
    report_text: str,
    output_dir: Path = _OUTPUT_DIR,
) -> dict[str, Path]:
    """
    Persist all benchmark outputs and return a dict of {label: path}.

    Parameters
    ----------
    results_df   : DataFrame returned by ``runner.run_benchmark()``.
    metrics_list : List of metric dicts from ``metrics.compute_metrics()``.
    report_text  : Formatted string from ``metrics.format_report()``.
    output_dir   : Directory to write outputs into (created if missing).

    Returns
    -------
    dict mapping short labels to absolute Path objects of saved files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # 1 ── raw scores spreadsheet
    raw_path = output_dir / "raw_scores.xlsx"
    results_df.to_excel(raw_path, index=False)
    paths["raw_scores"] = raw_path

    # 2 ── plain-text report
    report_path = output_dir / "accuracy_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    paths["report"] = report_path

    # 3 ── metrics summary spreadsheet
    summary_df = pd.DataFrame(metrics_list).set_index("method").T
    summary_path = output_dir / "metrics_summary.xlsx"
    summary_df.to_excel(summary_path)
    paths["metrics_summary"] = summary_path

    return paths
