#!/usr/bin/env python3
"""
Use Case 1: CSV-based Protein Stability Prediction using SPIRED-Stab

This script demonstrates how to predict protein stability changes (ddG and dTm)
from a CSV file containing variant sequences using SPIRED-Stab.

Usage:
    python examples/use_case_1_csv_stability_prediction.py --input examples/data/variants.csv --output results.csv
    python examples/use_case_1_csv_stability_prediction.py --input examples/data.csv  # Use existing demo data
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add the project root and scripts to the Python path
project_root = Path(__file__).parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from src.spired_stab_mcp import predict_stability_direct


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein stability from CSV file using SPIRED-Stab"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="examples/data.csv",
        help="Path to input CSV file with variant sequences (default: examples/data.csv)"
    )
    parser.add_argument(
        "--wt_fasta", "-w",
        type=str,
        default=None,
        help="Path to wild-type FASTA file (default: wt.fasta in same directory as input)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to output CSV file (default: {input}_spired_stab_pred.csv)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda:0",
        help="Device to use for computation (default: cuda:0, use 'cpu' if no GPU)"
    )

    args = parser.parse_args()

    # Convert to absolute paths
    input_file = os.path.abspath(args.input)

    # Validate input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        print(f"Current working directory: {os.getcwd()}")
        print("Make sure to run this script from the project root directory.")
        sys.exit(1)

    # Check if input file has required columns
    try:
        df = pd.read_csv(input_file)
        if 'seq' not in df.columns:
            print(f"Error: Input CSV must contain a 'seq' column with protein sequences.")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
        print(f"Found {len(df)} sequences in input file.")
    except Exception as e:
        print(f"Error reading input CSV: {e}")
        sys.exit(1)

    # Determine wild-type FASTA file
    wt_fasta_file = args.wt_fasta
    if wt_fasta_file is None:
        wt_fasta_file = str(Path(input_file).parent / 'wt.fasta')
    wt_fasta_file = os.path.abspath(wt_fasta_file)

    # Check if wild-type FASTA exists
    if not os.path.exists(wt_fasta_file):
        print(f"Error: Wild-type FASTA file not found: {wt_fasta_file}")
        print("Please provide a wild-type FASTA file using --wt_fasta option.")
        sys.exit(1)

    # Determine output file
    output_file = args.output
    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_spired_stab_pred.csv")
    output_file = os.path.abspath(output_file)

    print(f"Input file: {input_file}")
    print(f"Wild-type FASTA: {wt_fasta_file}")
    print(f"Output file: {output_file}")
    print(f"Device: {args.device}")
    print()

    try:
        # Run SPIRED-Stab prediction using the MCP function
        print("Starting SPIRED-Stab prediction...")
        result = predict_stability_direct(
            variant_file=input_file,
            wt_fasta_file=wt_fasta_file,
            output_file=output_file,
            device=args.device
        )

        print("Prediction completed successfully!")
        print(result)

        # Display some results
        if os.path.exists(output_file):
            results_df = pd.read_csv(output_file)
            print(f"\nFirst 5 predictions:")
            print(results_df.head().to_string(index=False))
            print(f"\nSummary statistics:")
            print(f"Number of variants: {len(results_df)}")
            print(f"Mean ddG: {results_df['ddG'].mean():.4f} ± {results_df['ddG'].std():.4f}")
            print(f"Mean dTm: {results_df['dTm'].mean():.4f} ± {results_df['dTm'].std():.4f}")

    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()