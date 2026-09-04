#!/usr/bin/env python3
"""
Process benchmark JSON files recursively and output formatted results.
Usage: python process_benchmarks.py <directory>
"""

import json
import sys
import statistics
from pathlib import Path


def format_number(value):
    """Format number to 5 decimal places."""
    return f"{value:.5f}"


def process_json_file(filepath):
    """Process a single JSON file and return formatted results."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return None
    
    # Extract data
    cold_run = data.get('cold_run')
    hot_runs = data.get('hot_runs', [])
    
    if cold_run is None or not hot_runs:
        print(f"Warning: {filepath} missing required fields", file=sys.stderr)
        return None
    
    # Calculate statistics
    avg_hot = statistics.mean(hot_runs)
    stdev_hot = statistics.stdev(hot_runs) if len(hot_runs) > 1 else 0.0
    
    # Format output
    formatted_cold = format_number(cold_run)
    formatted_hot_runs = "\t".join(format_number(x) for x in hot_runs)
    formatted_avg = format_number(avg_hot)
    formatted_stdev = format_number(stdev_hot)
    
    return f"{filepath.name}\t{formatted_cold}\t{formatted_hot_runs}\t{formatted_avg}\t{formatted_stdev}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python process_benchmarks.py <directory>", file=sys.stderr)
        sys.exit(1)
    
    directory = Path(sys.argv[1])
    
    if not directory.is_dir():
        print(f"Error: {directory} is not a valid directory", file=sys.stderr)
        sys.exit(1)
    
    # Print header
    print("File\tCold Run\tHot Runs\tAvg (Hot)\tStdDev (Hot)")
    print("-" * 80)
    
    # Find all JSON files recursively, excluding friends_of_friends_series.json
    json_files = sorted([
        f for f in directory.rglob('*.json')
        # if f.name != 'friends_of_friends_series.json'
    ])
    
    if not json_files:
        print("No JSON files found", file=sys.stderr)
        return
    
    # Process each file
    for filepath in json_files:
        result = process_json_file(filepath)
        if result:
            print(result)


if __name__ == '__main__':
    main()