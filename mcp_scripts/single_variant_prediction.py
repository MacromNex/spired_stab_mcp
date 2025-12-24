#!/usr/bin/env python3
"""
Script: single_variant_prediction.py
Description: Predict protein stability change for a single variant using SPIRED-Stab

Original Use Case: examples/use_case_3_single_variant_prediction.py
Dependencies Removed: Direct MCP import (inlined core functionality)

Usage:
    python mcp_scripts/single_variant_prediction.py --mutation "I31L" --wt_fasta wt.fasta --output output.csv
    python mcp_scripts/single_variant_prediction.py --variant_seq "AQTVPYGVSQI..." --wt_fasta wt.fasta --output output.csv

Example:
    python mcp_scripts/single_variant_prediction.py --mutation "I31L" --wt_fasta examples/data/wt.fasta --output results/single_variant.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, Any
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
    "variant_id": "variant_1",
    "output_format": "csv",
    "show_interpretation": True
}

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


def apply_mutations(wt_sequence: str, mutations_str: str) -> str:
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

        return model, esm2_650M, esm2_3B, esm2_batch_converter, getStabDataTest

    except Exception as e:
        raise ImportError(f"Failed to load models: {e}")


# ==============================================================================
# Core Function (main logic extracted from use case)
# ==============================================================================
def run_single_variant_prediction(
    wt_fasta_file: Union[str, Path],
    variant_seq: Optional[str] = None,
    mutation: Optional[str] = None,
    variant_id: Optional[str] = None,
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict protein stability change for a single variant using SPIRED-Stab.

    Args:
        wt_fasta_file: Path to wild-type sequence FASTA file
        variant_seq: Complete variant protein sequence (alternative to mutation)
        mutation: Mutation(s) to apply to WT sequence, e.g., "I31L,S3T" (alternative to variant_seq)
        variant_id: ID for the variant sequence (optional)
        output_file: Path to save output CSV (optional)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: DataFrame with prediction (id, seq, ddG, dTm)
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_single_variant_prediction("wt.fasta", mutation="I31L", output_file="output.csv")
        >>> print(result['result']['ddG'].iloc[0])
    """
    # Setup
    wt_fasta_file = Path(wt_fasta_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    if not variant_seq and not mutation:
        raise ValueError("Either variant_seq or mutation must be provided")

    # Load wild-type sequence
    wt_sequence = load_wt_fasta(wt_fasta_file)
    print(f"Loaded wild-type sequence (length: {len(wt_sequence)})")

    # Determine variant sequence
    if variant_seq:
        variant_sequence = variant_seq
        print(f"Using provided variant sequence (length: {len(variant_sequence)})")
    elif mutation:
        print(f"Applying mutation(s): {mutation}")
        variant_sequence = apply_mutations(wt_sequence, mutation)
        print(f"Generated variant sequence (length: {len(variant_sequence)})")

    # Validate sequence lengths match
    if len(variant_sequence) != len(wt_sequence):
        print(f"Warning: Variant sequence length ({len(variant_sequence)}) differs from wild-type ({len(wt_sequence)})")

    # Count differences
    differences = sum(1 for a, b in zip(wt_sequence, variant_sequence) if a != b)
    print(f"Number of amino acid differences: {differences}")

    # Set variant ID
    if not variant_id:
        variant_id = config['variant_id']

    # Load models
    device = config['device']
    print(f"Loading models on device: {device}")

    model, esm2_650M, esm2_3B, esm2_batch_converter, getStabDataTest = get_spired_models(device)

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

    # Calculate mutation positions
    mut_pos_torch_list = torch.tensor(
        (np.array(list(wt_sequence)) != np.array(list(variant_sequence))).astype(int).tolist()
    )

    # Embed variant sequence
    print("Embedding variant sequence...")
    f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(
        variant_sequence, esm2_3B, esm2_650M, esm2_batch_converter, device=device
    )
    mut_data = {
        'target_tokens': target_tokens,
        'esm2-3B': f1d_esm2_3B,
        'embedding': f1d_esm2_650M
    }

    # Predict stability
    print("Predicting stability...")
    with torch.no_grad():
        ddG, dTm, wt_features, mut_features = model(wt_data, mut_data, mut_pos_torch_list)

    # Create results DataFrame
    result_dict = {
        'id': variant_id,
        'seq': variant_sequence,
        'ddG': ddG.item(),
        'dTm': dTm.item()
    }
    results_df = pd.DataFrame([result_dict])

    # Save output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        save_results(results_df, output_path)
        print(f"Results saved to: {output_path}")

    # Show interpretation if enabled
    if config['show_interpretation']:
        ddG_value = ddG.item()
        print(f"\n=== Prediction Results ===")
        print(f"Variant ID: {variant_id}")
        print(f"ΔΔG (stability change): {ddG_value:.4f} kcal/mol")
        print(f"ΔTm (melting temp change): {dTm.item():.4f} °C")

        if ddG_value < 0:
            print(f"Interpretation: Variant is MORE stable than wild-type (ΔΔG = {ddG_value:.4f})")
        elif ddG_value > 0:
            print(f"Interpretation: Variant is LESS stable than wild-type (ΔΔG = {ddG_value:.4f})")
        else:
            print(f"Interpretation: Variant has similar stability to wild-type (ΔΔG = {ddG_value:.4f})")

    return {
        "result": results_df,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "wt_fasta_file": str(wt_fasta_file),
            "variant_id": variant_id,
            "mutation": mutation,
            "num_differences": differences,
            "device": device,
            "config": config,
            "ddG": ddG.item(),
            "dTm": dTm.item()
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
    input_group.add_argument('--variant_seq', help='Complete variant protein sequence')
    input_group.add_argument('--mutation', help='Mutation(s) to apply (e.g., I31L,S3T)')

    parser.add_argument('--wt_fasta', '-w', required=True, help='Wild-type FASTA file')
    parser.add_argument('--variant_id', default='variant_1', help='Variant sequence ID')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--device', '-d', default='cpu', help='Device (cuda:0, cpu)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--no_interpretation', action='store_true', help='Disable interpretation output')

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)

    # Override config with CLI arguments
    config_overrides = {
        'device': args.device,
        'variant_id': args.variant_id,
        'show_interpretation': not args.no_interpretation
    }

    # Auto-generate output file if not provided
    output_file = args.output
    if not output_file and args.mutation:
        output_file = f"single_variant_{args.mutation.replace(',', '_')}_pred.csv"
    elif not output_file:
        output_file = "single_variant_pred.csv"

    # Run prediction
    try:
        result = run_single_variant_prediction(
            wt_fasta_file=args.wt_fasta,
            variant_seq=args.variant_seq,
            mutation=args.mutation,
            variant_id=args.variant_id,
            output_file=output_file,
            config=config,
            **config_overrides
        )

        print(f"\n✅ Success: {result['output_file']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()