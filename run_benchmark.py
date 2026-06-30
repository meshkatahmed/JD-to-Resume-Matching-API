#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_benchmark.py
================
Entry-point for the JD-to-Resume Matching API benchmark.

Usage
-----
# Full benchmark — all 1 011 rows, all built-in methods:
    python run_benchmark.py

# Only specific methods:
    python run_benchmark.py --methods cbow,jaccard,bm25

# Quick smoke-test (first 20 rows, CBOW only):
    python run_benchmark.py --max-rows 20 --methods cbow

# List all registered methods:
    python run_benchmark.py --list-methods

# Suppress progress output:
    python run_benchmark.py --quiet

# Skip writing output files:
    python run_benchmark.py --no-save

Options
-------
--dataset PATH     Path to the Excel dataset. Defaults to
                   datasets/dataset_imputed.xlsx relative to this file.
--methods LIST     Comma-separated method names. Run --list-methods to see
                   all available options. Default: cbow,tfidf
--max-rows N       Only process the first N rows (0 = all). Default: 0 (all).
--quiet            Suppress per-batch progress output.
--no-save          Skip writing output files (just print the report).
--output-dir PATH  Directory for output files. Default: benchmark_results/
--list-methods     Print all registered scoring methods and exit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ── ensure stdout uses UTF-8 on Windows so Greek chars (ρ, τ) print fine ───
import io as _io
sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── ensure project root is on sys.path ──────────────────────────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Import benchmark package (scorers.py self-registers all built-ins on import)
from benchmark.runner        import run_benchmark, DATASET_PATH, GT_COL, score_col
from benchmark.scorers       import available_methods
from benchmark.metrics       import compute_metrics, format_report
from benchmark.report_writer import save_all, _OUTPUT_DIR


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the JD-to-Resume Matching API against dataset_imputed.xlsx",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset", type=Path, default=DATASET_PATH,
        metavar="PATH",
        help="Path to the benchmark Excel file.",
    )
    parser.add_argument(
        "--methods", type=str, default="jaccard,overlap,bm25,cbow,tfidf,sbert",
        metavar="LIST",
        help="Comma-separated scoring methods. Use --list-methods to see all options.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        metavar="N",
        help="Limit to first N rows (0 = all rows).",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Skip writing output files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_OUTPUT_DIR,
        metavar="PATH",
        help="Directory for output files.",
    )
    parser.add_argument(
        "--list-methods", action="store_true",
        help="Print all registered scoring methods and exit.",
    )
    return parser.parse_args()


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # ── List methods and exit ─────────────────────────────────────────────────
    if args.list_methods:
        print("Registered scoring methods:")
        for name in available_methods():
            print(f"  {name}")
        return

    methods: list[str] = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    max_rows = args.max_rows if args.max_rows > 0 else None

    # ── Step 1: Run the benchmark ────────────────────────────────────────────
    print("=" * 70)
    print("  JD-to-Resume Matching API — Benchmark Runner")
    print("=" * 70)
    print(f"  Dataset  : {args.dataset}")
    print(f"  Methods  : {', '.join(methods)}")
    print(f"  Max rows : {'all' if max_rows is None else max_rows}")
    print(f"  Output   : {'(suppressed)' if args.no_save else str(args.output_dir)}")
    print()

    t0 = time.perf_counter()
    print("  Scoring — please wait …")
    results_df = run_benchmark(
        dataset_path=args.dataset,
        methods=methods,
        max_rows=max_rows,
        verbose=not args.quiet,
    )
    elapsed_scoring = time.perf_counter() - t0
    print(f"\n  Scoring complete in {elapsed_scoring:.1f}s ({len(results_df)} rows)\n")

    # ── Step 2: Compute metrics ──────────────────────────────────────────────
    actual = results_df[GT_COL].tolist()

    metrics_list: list[dict] = []
    for method in methods:
        col = score_col(method)
        m = compute_metrics(
            actual=actual,
            predicted=results_df[col].tolist(),
            method_name=method.upper(),
        )
        metrics_list.append(m)

    # ── Step 3: Print report ─────────────────────────────────────────────────
    report = format_report(metrics_list)
    print(report)

    # ── Step 4: Save outputs ─────────────────────────────────────────────────
    if not args.no_save:
        paths = save_all(
            results_df=results_df,
            metrics_list=metrics_list,
            report_text=report,
            output_dir=args.output_dir,
        )
        print("\n  Output files written:")
        for label, path in paths.items():
            print(f"    [{label}]  {path}")
    print()


if __name__ == "__main__":
    main()
