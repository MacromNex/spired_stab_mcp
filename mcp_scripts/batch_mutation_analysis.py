#!/usr/bin/env python3
"""
Script: batch_mutation_analysis.py
Description: Batch mutation analysis using SPIRED-Stab for systematic mutation studies

Original Use Case: examples/use_case_4_batch_mutation_analysis.py
Dependencies Removed: Direct MCP import (inlined core functionality)

Usage:
    python mcp_scripts/batch_mutation_analysis.py --positions "31,67,124" --wt_fasta wt.fasta --output batch_results.csv
    python mcp_scripts/batch_mutation_analysis.py --mutations "I31L,H67Y,M124L" --wt_fasta wt.fasta --output specific_muts.csv

Example:
    python mcp_scripts/batch_mutation_analysis.py --mutations "I31L,H67Y" --wt_fasta examples/data/wt.fasta --output results/batch_analysis.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, Any, List, Tuple
import pandas as pd

# Essential scientific packages
import numpy as np
import torch

# Biopython for FASTA handling
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "device": "cpu",  # Default to CPU for compatibility
    "output_format": "csv",
    "progress_bar": True,
    "max_variants": 100,
    "save_summary": True,
    "show_analysis": True
}

# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# ==============================================================================
# Path Configuration
# ==============================================================================
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# ==============================================================================
# Inlined Utility Functions (simplified from repo)
# ==============================================================================
def load_wt_fasta(file_path: Path) -> str:
    """Load wild-type sequence from FASTA file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Wild-type FASTA not found: {file_path}")

    try:
        sequences = list(SeqIO.parse(file_path, "fasta"))
        if not sequences:
            raise ValueError("No sequences found in wild-type FASTA")
        return str(sequences[0].seq)
    except Exception as e:
        raise ValueError(f"Error reading wild-type FASTA: {e}")


def generate_single_mutations(wt_sequence: str, positions: List[int]) -> List[Tuple[str, str]]:
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


def generate_specific_mutations(wt_sequence: str, mutations_list: List[str]) -> List[Tuple[str, str]]:
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


def save_results(data: pd.DataFrame, file_path: Path) -> None:
    """Save results to CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)


# ==============================================================================
# Lazy Loading for Heavy Dependencies
# ==============================================================================
def get_spired_models(device: str = "cpu"):
    """Lazy load SPIRED-Stab models to minimize startup time."""
    sys.path.insert(0, str(SCRIPTS_DIR))

    try:
        from src.model import SPIRED_Stab
        from src.utils_train_valid import getStabDataTest
        import tqdm

        # Load SPIRED-Stab model
        model = SPIRED_Stab(device_list=[device, device, device, device])
        model_path = SCRIPTS_DIR / "data" / "model" / "SPIRED-Stab.pth"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model.load_state_dict(torch.load(str(model_path), map_location=device))
        model.to(device)
        model.eval()

        # Load ESM-2 models
        esm2_650M, _ = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
        esm2_650M.to(device)
        esm2_650M.eval()

        esm2_3B, esm2_alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D')
        esm2_3B.to(device)
        esm2_3B.eval()
        esm2_batch_converter = esm2_alphabet.get_batch_converter()

        return model, esm2_650M, esm2_3B, esm2_batch_converter, getStabDataTest, tqdm

    except Exception as e:
        raise ImportError(f"Failed to load models: {e}")


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_batch_mutation_analysis(
    wt_fasta_file: Union[str, Path],
    positions: Optional[List[int]] = None,
    mutations: Optional[List[str]] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Batch mutation analysis using SPIRED-Stab for systematic mutation studies.

    Args:
        wt_fasta_file: Path to wild-type sequence FASTA file
        positions: List of positions for saturation mutagenesis (1-based indexing)
        mutations: List of specific mutations (e.g., ["I31L", "H67Y"])
        output_file: Path to save output CSV (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: DataFrame with predictions (id, seq, ddG, dTm)
            - output_file: Path to output file (if saved)
            - summary_file: Path to summary file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_batch_mutation_analysis("wt.fasta", mutations=["I31L", "H67Y"], output_file="batch.csv")
        >>> print(result['metadata']['stabilizing_count'])
    """
    # Setup
    wt_fasta_file = Path(wt_fasta_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not positions and not mutations:
        raise ValueError("Either positions or mutations must be provided")

    # Load wild-type sequence
    wt_sequence = load_wt_fasta(wt_fasta_file)
    print(f"Loaded wild-type sequence (length: {len(wt_sequence)})")

    # Generate variants
    if positions:
        print(f"Generating all possible single mutations at positions: {positions}")
        variants = generate_single_mutations(wt_sequence, positions)
    elif mutations:
        print(f"Generating specific mutations: {mutations}")
        variants = generate_specific_mutations(wt_sequence, mutations)

    # Limit number of variants if too many
    if len(variants) > config['max_variants']:
        print(f"Too many variants ({len(variants)}), limiting to {config['max_variants']}")
        variants = variants[:config['max_variants']]

    print(f"Generated {len(variants)} variant sequences")

    # Load models
    device = config['device']
    print(f"Loading models on device: {device}")

    model, esm2_650M, esm2_3B, esm2_batch_converter, getStabDataTest, tqdm = get_spired_models(device)

    # Embed wild-type sequence
    print("Embedding wild-type sequence...")
    f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(
        wt_sequence, esm2_3B, esm2_650M, esm2_batch_converter, device=device
    )
    wt_data = {
        'target_tokens': target_tokens,
        'esm2-3B': f1d_esm2_3B,
        'embedding': f1d_esm2_650M
    }

    # Process variants
    results = []
    iterator = tqdm.tqdm(variants, desc="Batch mutation analysis") if config['progress_bar'] else variants

    for mutation_name, variant_sequence in iterator:
        # Calculate mutation positions
        mut_pos_torch_list = torch.tensor(
            (np.array(list(wt_sequence)) != np.array(list(variant_sequence))).astype(int).tolist()
        )

        # Embed variant sequence
        f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(
            variant_sequence, esm2_3B, esm2_650M, esm2_batch_converter, device=device
        )
        mut_data = {
            'target_tokens': target_tokens,
            'esm2-3B': f1d_esm2_3B,
            'embedding': f1d_esm2_650M
        }

        # Predict stability
        with torch.no_grad():
            ddG, dTm, wt_features, mut_features = model(wt_data, mut_data, mut_pos_torch_list)
            results.append({
                'id': mutation_name,
                'seq': variant_sequence,
                'ddG': ddG.item(),
                'dTm': dTm.item()
            })

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        save_results(results_df, output_path)
        print(f"Results saved to: {output_path}")

    # Analyze results
    stabilizing = results_df[results_df['ddG'] < 0]
    destabilizing = results_df[results_df['ddG'] > 0]

    # Show analysis if enabled
    if config['show_analysis']:
        print(f"\n=== BATCH MUTATION ANALYSIS RESULTS ===")
        print(f"Total variants analyzed: {len(results_df)}")
        print(f"\nSummary Statistics:")
        print(f"ΔΔG: {results_df['ddG'].mean():.4f} ± {results_df['ddG'].std():.4f}")
        print(f"ΔTm: {results_df['dTm'].mean():.4f} ± {results_df['dTm'].std():.4f}")

        print(f"\nStabilizing mutations (ΔΔG < 0): {len(stabilizing)}")
        print(f"Destabilizing mutations (ΔΔG > 0): {len(destabilizing)}")

        if len(stabilizing) > 0:
            print(f"\nTop 5 most stabilizing mutations:")
            print(stabilizing.nsmallest(5, 'ddG')[['id', 'ddG', 'dTm']].to_string(index=False))

        if len(destabilizing) > 0:
            print(f"\nTop 5 most destabilizing mutations:")
            print(destabilizing.nlargest(5, 'ddG')[['id', 'ddG', 'dTm']].to_string(index=False))

    # Save summary statistics if enabled
    summary_path = None
    if config['save_summary'] and output_file:
        summary_path = Path(str(output_path).replace('.csv', '_summary.txt'))
        with open(summary_path, 'w') as f:
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
        print(f"Summary saved to: {summary_path}")

    return {
        "result": results_df,
        "output_file": str(output_path) if output_path else None,
        "summary_file": str(summary_path) if summary_path else None,
        "metadata": {
            "wt_fasta_file": str(wt_fasta_file),
            "num_variants": len(results_df),
            "stabilizing_count": len(stabilizing),
            "destabilizing_count": len(destabilizing),
            "device": device,
            "config": config,
            "mean_ddG": results_df['ddG'].mean(),
            "std_ddG": results_df['ddG'].std(),
            "mean_dTm": results_df['dTm'].mean(),
            "std_dTm": results_df['dTm'].std()
        }
    }


# ==============================================================================
# CLI Interface
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--positions', help='Positions to mutate (comma-separated, 1-based), e.g., "31,67,124"')
    input_group.add_argument('--mutations', help='Specific mutations (comma-separated), e.g., "I31L,H67Y,M124L"')

    parser.add_argument('--wt_fasta', '-w', required=True, help='Wild-type FASTA file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--device', '-d', default='cpu', help='Device (cuda:0, cpu)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--max_variants', type=int, default=100, help='Maximum number of variants')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_analysis', action='store_true', help='Disable analysis output')
    parser.add_argument('--no_summary', action='store_true', help='Disable summary file')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)

    # Parse positions or mutations
    positions = None
    mutations = None
    if args.positions:
        positions = [int(p.strip()) for p in args.positions.split(',')]
    if args.mutations:
        mutations = [m.strip() for m in args.mutations.split(',')]

    # Override config with CLI arguments
    config_overrides = {
        'device': args.device,
        'max_variants': args.max_variants,
        'progress_bar': not args.no_progress,
        'show_analysis': not args.no_analysis,
        'save_summary': not args.no_summary
    }

    # Auto-generate output file if not provided
    output_file = args.output
    if not output_file:
        if args.mutations:
            output_file = f"batch_mutations_{'-'.join(args.mutations.split(','))}_analysis.csv"
        elif args.positions:
            output_file = f"batch_positions_{'-'.join(args.positions.split(','))}_analysis.csv"
        output_file = output_file.replace(',', '_')

    # Run analysis
    try:
        result = run_batch_mutation_analysis(
            wt_fasta_file=args.wt_fasta,
            positions=positions,
            mutations=mutations,
            output_file=output_file,
            config=config,
            **config_overrides
        )

        print(f"\n✅ Success: {result['output_file']}")
        print(f"Processed {result['metadata']['num_variants']} variants")
        print(f"Stabilizing: {result['metadata']['stabilizing_count']}, Destabilizing: {result['metadata']['destabilizing_count']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()