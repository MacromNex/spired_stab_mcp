#!/usr/bin/env python3
"""
Script: fasta_stability_prediction.py
Description: Predict protein stability changes from FASTA file using SPIRED-Stab

Original Use Case: examples/use_case_2_fasta_stability_prediction.py
Dependencies Removed: Direct MCP import (inlined core functionality)

Usage:
    python mcp_scripts/fasta_stability_prediction.py --input sequences.fasta --wt_fasta wt.fasta --output output.csv

Example:
    python mcp_scripts/fasta_stability_prediction.py --input examples/data/sequences.fasta --wt_fasta examples/data/wt.fasta --output results/fasta_predictions.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any
import pandas as pd

# Essential scientific packages
import numpy as np
import torch

# Biopython for FASTA handling
from Bio import SeqIO

# ==============================================================================
# Configuration (extracted from use case)
# ==============================================================================
DEFAULT_CONFIG = {
    "device": "cpu",  # Default to CPU for compatibility
    "input_format": "fasta",
    "output_format": "csv",
    "progress_bar": True,
    "show_analysis": True
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
def load_fasta_input(file_path: Path) -> list:
    """Load FASTA input file and validate sequences."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        sequences = list(SeqIO.parse(file_path, "fasta"))
        if not sequences:
            raise ValueError("No sequences found in FASTA file")
        return sequences
    except Exception as e:
        raise ValueError(f"Error reading FASTA: {e}")


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
def run_fasta_stability_prediction(
    input_file: Union[str, Path],
    wt_fasta_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Predict protein stability changes from FASTA file using SPIRED-Stab.

    Args:
        input_file: Path to FASTA file containing variant sequences
        wt_fasta_file: Path to wild-type sequence FASTA file
        output_file: Path to save output CSV (optional, auto-generated if not provided)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: DataFrame with predictions (id, seq, ddG, dTm)
            - output_file: Path to output file (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_fasta_stability_prediction("variants.fasta", "wt.fasta", "output.csv")
        >>> print(result['metadata']['num_variants'])
    """
    # Setup
    input_file = Path(input_file)
    wt_fasta_file = Path(wt_fasta_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load input data
    sequences = load_fasta_input(input_file)
    wt_sequence = load_wt_fasta(wt_fasta_file)

    print(f"Loaded {len(sequences)} sequences from {input_file}")
    print(f"First few sequence IDs: {[seq.id for seq in sequences[:3]]}")
    print(f"Wild-type sequence length: {len(wt_sequence)}")

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
    iterator = tqdm.tqdm(sequences, desc="Predicting stability") if config['progress_bar'] else sequences

    for seq_record in iterator:
        mut_seq = str(seq_record.seq)
        seq_id = seq_record.id

        # Calculate mutation positions
        mut_pos_torch_list = torch.tensor(
            (np.array(list(wt_sequence)) != np.array(list(mut_seq))).astype(int).tolist()
        )

        # Embed mutant sequence
        f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(
            mut_seq, esm2_3B, esm2_650M, esm2_batch_converter, device=device
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
                'id': seq_id,
                'seq': mut_seq,
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

    # Show analysis if enabled
    if config['show_analysis']:
        print(f"\n=== Analysis ===")
        print(f"Most stable variants (lowest ddG):")
        print(results_df.nsmallest(3, 'ddG')[['id', 'ddG', 'dTm']].to_string(index=False))
        print(f"\nLeast stable variants (highest ddG):")
        print(results_df.nlargest(3, 'ddG')[['id', 'ddG', 'dTm']].to_string(index=False))

    return {
        "result": results_df,
        "output_file": str(output_path) if output_path else None,
        "metadata": {
            "input_file": str(input_file),
            "wt_fasta_file": str(wt_fasta_file),
            "num_variants": len(results_df),
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
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file with sequences')
    parser.add_argument('--wt_fasta', '-w', required=True, help='Wild-type FASTA file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--device', '-d', default='cpu', help='Device (cuda:0, cpu)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_analysis', action='store_true', help='Disable analysis output')

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
        'progress_bar': not args.no_progress,
        'show_analysis': not args.no_analysis
    }

    # Auto-generate output file if not provided
    output_file = args.output
    if not output_file:
        input_path = Path(args.input)
        output_file = input_path.parent / f"{input_path.stem}_stability_pred.csv"

    # Run prediction
    try:
        result = run_fasta_stability_prediction(
            input_file=args.input,
            wt_fasta_file=args.wt_fasta,
            output_file=output_file,
            config=config,
            **config_overrides
        )

        print(f"\n✅ Success: {result['output_file']}")
        print(f"Processed {result['metadata']['num_variants']} variants")
        print(f"Mean ΔΔG: {result['metadata']['mean_ddG']:.4f} ± {result['metadata']['std_ddG']:.4f}")
        print(f"Mean ΔTm: {result['metadata']['mean_dTm']:.4f} ± {result['metadata']['std_dTm']:.4f}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()