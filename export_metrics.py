#!/usr/bin/env python3
"""
Export metrics from summary.json files in subdirectories to a CSV file.

Usage:
    python export_metrics.py <folder_path>

Example:
    python export_metrics.py output/baseline_trace/v4/minimax/200000
    python export_metrics.py output/baseline_trace/v7/minimax/200000

Output:
    <folder_path>/metrics_summary.csv
"""

import json
import csv
import sys
import os
from pathlib import Path


def find_summary_json_files(base_folder: Path):
    """Find all *_summary.json files in subdirectories of base_folder."""
    summary_files = []
    for subdir in base_folder.iterdir():
        if subdir.is_dir():
            for file in subdir.glob("*_summary.json"):
                summary_files.append(file)
    return sorted(summary_files)


def extract_metrics(summary_path: Path):
    """Extract relevant fields from a summary.json file."""
    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    label = data.get("label", "")
    timestamp = data.get("timestamp", "")
    metrics = data.get("metrics", {})

    row = {
        "label": label,
        "timestamp": timestamp,
        "target": metrics.get("target", ""),
        "nontarget": metrics.get("nontarget", ""),
        "total": metrics.get("total", ""),
        "tp": metrics.get("tp", ""),
        "tn": metrics.get("tn", ""),
        "fp": metrics.get("fp", ""),
        "fn": metrics.get("fn", ""),
        "correct": metrics.get("correct", ""),
        "errors": metrics.get("errors", ""),
        "uncertain": metrics.get("uncertain", ""),
        "uncertain_pos": metrics.get("uncertain_pos", ""),
        "uncertain_neg": metrics.get("uncertain_neg", ""),
        "predict_target": metrics.get("predict_target", ""),
        "predict_nontarget": metrics.get("predict_nontarget", ""),
        "attack_recall": metrics.get("attack_recall", ""),
        "attack_precision": metrics.get("attack_precision", ""),
        "negative_precision": metrics.get("negative_precision", ""),
    }
    return row


def export_metrics(folder_path: str):
    """Export metrics summary to CSV."""
    base_folder = Path(folder_path)

    if not base_folder.exists():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    if not base_folder.is_dir():
        print(f"Error: Not a directory: {folder_path}")
        sys.exit(1)

    summary_files = find_summary_json_files(base_folder)

    if not summary_files:
        print(f"No summary.json files found in subdirectories of: {folder_path}")
        sys.exit(1)

    print(f"Found {len(summary_files)} summary.json files")

    # Extract all rows
    rows = []
    for summary_file in summary_files:
        try:
            row = extract_metrics(summary_file)
            rows.append(row)
        except Exception as e:
            print(f"Warning: Failed to process {summary_file}: {e}")

    if not rows:
        print("No valid data to export")
        sys.exit(1)

    # Define CSV columns
    fieldnames = [
        "label",
        "timestamp",
        "target",
        "nontarget",
        "total",
        "tp",
        "tn",
        "fp",
        "fn",
        "correct",
        "errors",
        "uncertain",
        "uncertain_pos",
        "uncertain_neg",
        "predict_target",
        "predict_nontarget",
        "attack_recall",
        "attack_precision",
        "negative_precision",
    ]

    # Write CSV
    output_path = base_folder / "metrics_summary.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Exported to: {output_path}")

    # Print summary
    print("\nSummary:")
    for row in rows:
        print(f"  {row['label']}: recall={row['attack_recall']}, precision={row['attack_precision']}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("Error: Missing folder path argument")
        sys.exit(1)

    folder_path = sys.argv[1]
    export_metrics(folder_path)


if __name__ == "__main__":
    main()
