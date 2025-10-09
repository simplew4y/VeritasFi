"""Utility to aggregate `num_recalls` metrics from retrieval experiment outputs.

Example usage:
    python collect_num_recalls.py --input-dir /path/to/experiment/step2
    python collect_num_recalls.py --input-dir /path/to/step2 --filename result.json

Requirements:
    pip install datasets
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

from datasets import load_dataset


@dataclass
class FileStats:
    """Statistics for num_recalls from a single result file."""
    path: Path
    count: int
    values: list[float]
    average: float | None
    minimum: float | None
    maximum: float | None
    unique_recall_values: list[int]
    unique_recall_average: float | None
    unique_recall_minimum: int | None
    unique_recall_maximum: int | None


def load_json_records(file_path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a file using HuggingFace datasets.
    
    Automatically handles both JSON and JSONL formats.
    """
    try:
        # Load dataset using HuggingFace datasets
        # This automatically detects JSON vs JSONL format
        dataset = load_dataset('json', data_files=str(file_path), split='train')
        
        # Convert dataset to list of dictionaries
        records = [dict(example) for example in dataset]
        
        return records
        
    except Exception as e:
        print(f"Warning: Failed to load {file_path} with datasets library: {e}")
        return []


def extract_num_recalls(records: list[dict[str, Any]]) -> list[float]:
    """Extract num_recalls values from a list of records."""
    values = []
    for record in records:
        num_recalls = record.get("num_recalls")
        if isinstance(num_recalls, (int, float)):
            values.append(float(num_recalls))
    return values


def extract_unique_recalls(records: list[dict[str, Any]]) -> list[int]:
    """Extract unique recall counts from a list of records.
    
    For each record, count the number of unique chunks based on string equality.
    Chunks are extracted from 'query_chunks' field if available.
    """
    unique_counts = []
    for record in records:
        query_chunks = record.get("query_chunks")
        if query_chunks is not None and isinstance(query_chunks, list):
            # Count unique chunks by converting to set (removes duplicates)
            unique_chunks = set(query_chunks)
            unique_counts.append(len(unique_chunks))
    return unique_counts


def collect_num_recalls(input_dir: Path, filename: str = "result.json") -> list[FileStats]:
    """Collect num_recalls statistics from all matching files in the directory tree."""
    stats_list = []
    
    # Find all matching files recursively
    for result_path in sorted(input_dir.rglob(filename)):
        if not result_path.is_file():
            continue
        
        try:
            records = load_json_records(result_path)
            values = extract_num_recalls(records)
            unique_recall_values = extract_unique_recalls(records)
            
            if values:
                stats = FileStats(
                    path=result_path,
                    count=len(values),
                    values=values,
                    average=sum(values) / len(values),
                    minimum=min(values),
                    maximum=max(values),
                    unique_recall_values=unique_recall_values,
                    unique_recall_average=sum(unique_recall_values) / len(unique_recall_values) if unique_recall_values else None,
                    unique_recall_minimum=min(unique_recall_values) if unique_recall_values else None,
                    unique_recall_maximum=max(unique_recall_values) if unique_recall_values else None,
                )
            else:
                stats = FileStats(
                    path=result_path,
                    count=0,
                    values=[],
                    average=None,
                    minimum=None,
                    maximum=None,
                    unique_recall_values=[],
                    unique_recall_average=None,
                    unique_recall_minimum=None,
                    unique_recall_maximum=None,
                )
            
            stats_list.append(stats)
            
        except Exception as e:
            print(f"Error processing {result_path}: {e}")
    
    return stats_list


def print_summary(file_stats: list[FileStats], base_dir: Path, output_file: Path | None = None) -> None:
    """Print a formatted summary of num_recalls statistics.
    
    Args:
        file_stats: List of file statistics to summarize
        base_dir: Base directory for relative path calculations
        output_file: Optional path to save the summary. If None, only prints to console.
    """
    if not file_stats:
        message = "No result files found."
        print(message)
        if output_file:
            output_file.write_text(message)
        return
    
    # Build summary as a list of lines
    lines = []
    lines.append("=" * 70)
    lines.append("NUM_RECALLS SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Base directory: {base_dir}")
    lines.append(f"Total files processed: {len(file_stats)}")
    lines.append("")
    
    # Collect all values for overall statistics
    all_values = []
    all_unique_values = []
    files_with_data = 0
    
    for stats in file_stats:
        rel_path = stats.path.relative_to(base_dir)
        lines.append(f"📁 {rel_path}")
        lines.append(f"   Records: {stats.count}")
        
        if stats.average is not None:
            lines.append(f"   Average num_recalls: {stats.average:.2f}")
            lines.append(f"   Min num_recalls:     {stats.minimum:.2f}")
            lines.append(f"   Max num_recalls:     {stats.maximum:.2f}")
            all_values.extend(stats.values)
            files_with_data += 1
        else:
            lines.append(f"   No num_recalls data found")
        
        if stats.unique_recall_average is not None:
            lines.append(f"   Average unique_recall: {stats.unique_recall_average:.2f}")
            lines.append(f"   Min unique_recall:     {stats.unique_recall_minimum}")
            lines.append(f"   Max unique_recall:     {stats.unique_recall_maximum}")
            all_unique_values.extend(stats.unique_recall_values)
        else:
            lines.append(f"   No unique_recall data found")
        lines.append("")
    
    # Print overall statistics
    if all_values or all_unique_values:
        lines.append("=" * 70)
        lines.append("OVERALL STATISTICS")
        lines.append("=" * 70)
        lines.append(f"Files with data: {files_with_data}/{len(file_stats)}")
        
        if all_values:
            lines.append("")
            lines.append("Num Recalls:")
            lines.append(f"  Total records: {len(all_values)}")
            lines.append(f"  Average:       {sum(all_values) / len(all_values):.2f}")
            lines.append(f"  Min:           {min(all_values):.2f}")
            lines.append(f"  Max:           {max(all_values):.2f}")
        
        if all_unique_values:
            lines.append("")
            lines.append("Unique Recalls:")
            lines.append(f"  Total records: {len(all_unique_values)}")
            lines.append(f"  Average:       {sum(all_unique_values) / len(all_unique_values):.2f}")
            lines.append(f"  Min:           {min(all_unique_values)}")
            lines.append(f"  Max:           {max(all_unique_values)}")
        
        lines.append("=" * 70)
    else:
        lines.append("No num_recalls data found in any files.")
    
    # Print to console
    summary_text = "\n".join(lines)
    print(summary_text)
    
    # Save to file if specified
    if output_file:
        output_file.write_text(summary_text + "\n")
        print(f"\nSummary saved to: {output_file}")


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Aggregate num_recalls statistics from experiment result files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python collect_num_recalls.py --input-dir /path/to/experiment/step2
  python collect_num_recalls.py --input-dir /path/to/step2 --filename "result*.json"
        """
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Base directory containing experiment outputs"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="result.json",
        help="Filename pattern to search for (default: result.json)"
    )
    
    args = parser.parse_args()
    
    # Resolve and validate input directory
    input_dir = args.input_dir.expanduser().resolve()
    
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        return
    
    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        return
    
    # Collect and print statistics
    file_stats = collect_num_recalls(input_dir, args.filename)
    
    # Save summary to file in the input directory
    output_file = input_dir / "num_recalls_summary.txt"
    print_summary(file_stats, input_dir, output_file)


if __name__ == "__main__":
    main()
