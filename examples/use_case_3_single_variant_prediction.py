#!/usr/bin/env python3
"""
Use Case 3: Single Variant Protein Stability Prediction using SPIRED-Stab

This script demonstrates how to predict protein stability changes for a single
variant sequence compared to a wild-type sequence using SPIRED-Stab.

Usage:
    python examples/use_case_3_single_variant_prediction.py --variant_seq "AQTVPYGVSQI..." --output results.csv
    python examples/use_case_3_single_variant_prediction.py --mutation "I31L,S3T" --output results.csv
"""

import os
import sys
import argparse
import tempfile
import pandas as pd
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Add the project root and scripts to the Python path
project_root = Path(__file__).parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from src.spired_stab_mcp import predict_stability_direct


def apply_mutations(wt_sequence, mutations_str):
    """
    Apply mutations to wild-type sequence.

    Args:
        wt_sequence: Wild-type protein sequence
        mutations_str: Mutations in format "I31L,S3T" or "I31L/S3T"

    Returns:
        Modified sequence with mutations applied
    """
    sequence = list(wt_sequence)

    # Parse mutations (handle both comma and slash separators)
    mutations = mutations_str.replace('/', ',').split(',')

    for mutation in mutations:
        mutation = mutation.strip()
        if not mutation:
            continue

        # Parse mutation format like "I31L" (from I to L at position 31)
        if len(mutation) < 3:
            print(f"Warning: Invalid mutation format: {mutation}")
            continue

        wt_aa = mutation[0]
        mut_aa = mutation[-1]
        pos_str = mutation[1:-1]

        try:
            position = int(pos_str) - 1  # Convert to 0-based indexing
        except ValueError:
            print(f"Warning: Invalid position in mutation: {mutation}")
            continue

        if position < 0 or position >= len(sequence):
            print(f"Warning: Position {position + 1} is out of range for sequence length {len(sequence)}")
            continue

        if sequence[position] != wt_aa:
            print(f"Warning: Expected {wt_aa} at position {position + 1}, but found {sequence[position]}")

        sequence[position] = mut_aa
        print(f"Applied mutation: {wt_aa}{position + 1}{mut_aa}")

    return ''.join(sequence)


def main():
    parser = argparse.ArgumentParser(
        description="Predict protein stability for a single variant using SPIRED-Stab"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--variant_seq",
        type=str,
        help="Complete variant protein sequence"
    )
    input_group.add_argument(
        "--mutation",
        type=str,
        help="Mutation(s) to apply to wild-type sequence (format: I31L,S3T or I31L/S3T)"
    )

    parser.add_argument(
        "--wt_fasta", "-w",
        type=str,
        default="examples/wt.fasta",
        help="Path to wild-type FASTA file (default: examples/wt.fasta)"
    )
    parser.add_argument(
        "--variant_id",
        type=str,
        default="variant_1",
        help="ID for the variant sequence (default: variant_1)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="single_variant_pred.csv",
        help="Path to output CSV file (default: single_variant_pred.csv)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda:0",
        help="Device to use for computation (default: cuda:0, use 'cpu' if no GPU)"
    )

    args = parser.parse_args()

    # Load wild-type sequence
    wt_fasta_file = os.path.abspath(args.wt_fasta)
    if not os.path.exists(wt_fasta_file):
        print(f"Error: Wild-type FASTA file not found: {wt_fasta_file}")
        sys.exit(1)

    try:
        wt_sequences = list(SeqIO.parse(wt_fasta_file, "fasta"))
        if len(wt_sequences) == 0:
            print(f"Error: No sequences found in wild-type FASTA file: {wt_fasta_file}")
            sys.exit(1)
        wt_sequence = str(wt_sequences[0].seq)
        print(f"Loaded wild-type sequence: {wt_sequences[0].id} (length: {len(wt_sequence)})")
    except Exception as e:
        print(f"Error reading wild-type FASTA: {e}")
        sys.exit(1)

    # Determine variant sequence
    if args.variant_seq:
        variant_sequence = args.variant_seq
        print(f"Using provided variant sequence (length: {len(variant_sequence)})")
    elif args.mutation:
        print(f"Applying mutation(s): {args.mutation}")
        variant_sequence = apply_mutations(wt_sequence, args.mutation)
        print(f"Generated variant sequence (length: {len(variant_sequence)})")

    # Validate sequence lengths match
    if len(variant_sequence) != len(wt_sequence):
        print(f"Warning: Variant sequence length ({len(variant_sequence)}) differs from wild-type ({len(wt_sequence)})")

    # Count differences
    differences = sum(1 for a, b in zip(wt_sequence, variant_sequence) if a != b)
    print(f"Number of amino acid differences: {differences}")

    # Create temporary FASTA file for the variant
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
    variant_record = SeqRecord(
        Seq(variant_sequence),
        id=args.variant_id,
        description=""
    )
    SeqIO.write([variant_record], temp_fasta, "fasta")
    temp_fasta.close()

    output_file = os.path.abspath(args.output)

    print(f"Wild-type FASTA: {wt_fasta_file}")
    print(f"Variant sequence ID: {args.variant_id}")
    print(f"Output file: {output_file}")
    print(f"Device: {args.device}")
    print()

    try:
        # Run SPIRED-Stab prediction using the MCP function
        print("Starting SPIRED-Stab prediction...")
        result = predict_stability_direct(
            variant_file=temp_fasta.name,
            wt_fasta_file=wt_fasta_file,
            output_file=output_file,
            device=args.device
        )

        print("Prediction completed successfully!")
        print(result)

        # Clean up temporary file
        os.unlink(temp_fasta.name)

        # Display results
        if os.path.exists(output_file):
            results_df = pd.read_csv(output_file)
            print(f"\nPrediction Results:")
            print(f"Variant ID: {results_df.iloc[0]['id']}")
            print(f"ΔΔG (stability change): {results_df.iloc[0]['ddG']:.4f}")
            print(f"ΔTm (melting temp change): {results_df.iloc[0]['dTm']:.4f}")

            ddG = results_df.iloc[0]['ddG']
            if ddG < 0:
                print(f"Interpretation: Variant is MORE stable than wild-type (ΔΔG = {ddG:.4f})")
            elif ddG > 0:
                print(f"Interpretation: Variant is LESS stable than wild-type (ΔΔG = {ddG:.4f})")
            else:
                print(f"Interpretation: Variant has similar stability to wild-type (ΔΔG = {ddG:.4f})")

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Clean up temporary file
        if os.path.exists(temp_fasta.name):
            os.unlink(temp_fasta.name)
        sys.exit(1)


if __name__ == "__main__":
    main()