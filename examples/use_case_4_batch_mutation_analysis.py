#!/usr/bin/env python3
"""
Use Case 4: Batch Mutation Analysis using SPIRED-Stab

This script generates multiple variants with specific mutations and analyzes
their stability changes in batch. Useful for systematic mutation studies.

Usage:
    python examples/use_case_4_batch_mutation_analysis.py --positions 31,67,124 --output batch_results.csv
    python examples/use_case_4_batch_mutation_analysis.py --mutations "I31L,H67Y,M124L" --output specific_muts.csv
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
from itertools import product

# Add the project root and scripts to the Python path
project_root = Path(__file__).parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from src.spired_stab_mcp import predict_stability_direct


# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def generate_single_mutations(wt_sequence, positions):
    """
    Generate all possible single mutations at specified positions.

    Args:
        wt_sequence: Wild-type protein sequence
        positions: List of positions to mutate (1-based indexing)

    Returns:
        List of tuples (mutation_name, mutant_sequence)
    """
    variants = []

    for pos in positions:
        if pos < 1 or pos > len(wt_sequence):
            print(f"Warning: Position {pos} is out of range for sequence length {len(wt_sequence)}")
            continue

        wt_aa = wt_sequence[pos - 1]  # Convert to 0-based indexing

        for mut_aa in AMINO_ACIDS:
            if mut_aa != wt_aa:  # Skip wild-type amino acid
                # Create mutant sequence
                mutant_seq = list(wt_sequence)
                mutant_seq[pos - 1] = mut_aa
                mutant_sequence = ''.join(mutant_seq)

                mutation_name = f"{wt_aa}{pos}{mut_aa}"
                variants.append((mutation_name, mutant_sequence))

    return variants


def generate_specific_mutations(wt_sequence, mutations_list):
    """
    Generate variants with specific mutations.

    Args:
        wt_sequence: Wild-type protein sequence
        mutations_list: List of mutations like ["I31L", "H67Y", "M124L"]

    Returns:
        List of tuples (mutation_name, mutant_sequence)
    """
    variants = []

    for mutation in mutations_list:
        mutation = mutation.strip()
        if len(mutation) < 3:
            print(f"Warning: Invalid mutation format: {mutation}")
            continue

        wt_aa = mutation[0]
        mut_aa = mutation[-1]
        pos_str = mutation[1:-1]

        try:
            position = int(pos_str)
        except ValueError:
            print(f"Warning: Invalid position in mutation: {mutation}")
            continue

        if position < 1 or position > len(wt_sequence):
            print(f"Warning: Position {position} is out of range for sequence length {len(wt_sequence)}")
            continue

        if wt_sequence[position - 1] != wt_aa:
            print(f"Warning: Expected {wt_aa} at position {position}, but found {wt_sequence[position - 1]}")

        # Create mutant sequence
        mutant_seq = list(wt_sequence)
        mutant_seq[position - 1] = mut_aa
        mutant_sequence = ''.join(mutant_seq)

        variants.append((mutation, mutant_sequence))

    return variants


def main():
    parser = argparse.ArgumentParser(
        description="Batch mutation analysis using SPIRED-Stab"
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--positions",
        type=str,
        help="Positions to mutate (comma-separated, 1-based indexing), e.g., '31,67,124'"
    )
    input_group.add_argument(
        "--mutations",
        type=str,
        help="Specific mutations to analyze (comma-separated), e.g., 'I31L,H67Y,M124L'"
    )

    parser.add_argument(
        "--wt_fasta", "-w",
        type=str,
        default="examples/wt.fasta",
        help="Path to wild-type FASTA file (default: examples/wt.fasta)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="batch_mutation_analysis.csv",
        help="Path to output CSV file (default: batch_mutation_analysis.csv)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda:0",
        help="Device to use for computation (default: cuda:0, use 'cpu' if no GPU)"
    )
    parser.add_argument(
        "--max_variants",
        type=int,
        default=100,
        help="Maximum number of variants to generate (default: 100)"
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

    # Generate variants
    if args.positions:
        positions = [int(p.strip()) for p in args.positions.split(',')]
        print(f"Generating all possible single mutations at positions: {positions}")
        variants = generate_single_mutations(wt_sequence, positions)
    elif args.mutations:
        mutations_list = [m.strip() for m in args.mutations.split(',')]
        print(f"Generating specific mutations: {mutations_list}")
        variants = generate_specific_mutations(wt_sequence, mutations_list)

    # Limit number of variants if too many
    if len(variants) > args.max_variants:
        print(f"Too many variants ({len(variants)}), limiting to {args.max_variants}")
        variants = variants[:args.max_variants]

    print(f"Generated {len(variants)} variant sequences")

    # Create temporary FASTA file for variants
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
    variant_records = []
    for i, (mutation_name, variant_sequence) in enumerate(variants):
        variant_record = SeqRecord(
            Seq(variant_sequence),
            id=mutation_name,
            description=""
        )
        variant_records.append(variant_record)

    SeqIO.write(variant_records, temp_fasta, "fasta")
    temp_fasta.close()

    output_file = os.path.abspath(args.output)

    print(f"Wild-type FASTA: {wt_fasta_file}")
    print(f"Number of variants: {len(variants)}")
    print(f"Output file: {output_file}")
    print(f"Device: {args.device}")
    print()

    try:
        # Run SPIRED-Stab prediction using the MCP function
        print("Starting SPIRED-Stab batch prediction...")
        result = predict_stability_direct(
            variant_file=temp_fasta.name,
            wt_fasta_file=wt_fasta_file,
            output_file=output_file,
            device=args.device
        )

        print("Batch prediction completed successfully!")
        print(result)

        # Clean up temporary file
        os.unlink(temp_fasta.name)

        # Analyze results
        if os.path.exists(output_file):
            results_df = pd.read_csv(output_file)

            print(f"\n=== BATCH MUTATION ANALYSIS RESULTS ===")
            print(f"Total variants analyzed: {len(results_df)}")
            print(f"\nSummary Statistics:")
            print(f"ΔΔG: {results_df['ddG'].mean():.4f} ± {results_df['ddG'].std():.4f}")
            print(f"ΔTm: {results_df['dTm'].mean():.4f} ± {results_df['dTm'].std():.4f}")

            # Find stabilizing vs destabilizing mutations
            stabilizing = results_df[results_df['ddG'] < 0]
            destabilizing = results_df[results_df['ddG'] > 0]

            print(f"\nStabilizing mutations (ΔΔG < 0): {len(stabilizing)}")
            print(f"Destabilizing mutations (ΔΔG > 0): {len(destabilizing)}")

            print(f"\nTop 5 most stabilizing mutations:")
            if len(stabilizing) > 0:
                print(stabilizing.nsmallest(5, 'ddG')[['id', 'ddG', 'dTm']].to_string(index=False))
            else:
                print("No stabilizing mutations found")

            print(f"\nTop 5 most destabilizing mutations:")
            if len(destabilizing) > 0:
                print(destabilizing.nlargest(5, 'ddG')[['id', 'ddG', 'dTm']].to_string(index=False))
            else:
                print("No destabilizing mutations found")

            # Save summary statistics
            summary_file = output_file.replace('.csv', '_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f"Batch Mutation Analysis Summary\n")
                f.write(f"================================\n\n")
                f.write(f"Total variants: {len(results_df)}\n")
                f.write(f"Stabilizing mutations: {len(stabilizing)}\n")
                f.write(f"Destabilizing mutations: {len(destabilizing)}\n\n")
                f.write(f"ΔΔG statistics:\n")
                f.write(f"  Mean: {results_df['ddG'].mean():.4f}\n")
                f.write(f"  Std: {results_df['ddG'].std():.4f}\n")
                f.write(f"  Min: {results_df['ddG'].min():.4f}\n")
                f.write(f"  Max: {results_df['ddG'].max():.4f}\n\n")
                f.write(f"ΔTm statistics:\n")
                f.write(f"  Mean: {results_df['dTm'].mean():.4f}\n")
                f.write(f"  Std: {results_df['dTm'].std():.4f}\n")
                f.write(f"  Min: {results_df['dTm'].min():.4f}\n")
                f.write(f"  Max: {results_df['dTm'].max():.4f}\n")

            print(f"\nSummary saved to: {summary_file}")

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Clean up temporary file
        if os.path.exists(temp_fasta.name):
            os.unlink(temp_fasta.name)
        sys.exit(1)


if __name__ == "__main__":
    main()