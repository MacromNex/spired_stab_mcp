#!/usr/bin/env python3
"""
Script: comparative_stability_analysis.py
Description: Comparative stability analysis using SPIRED-Stab with rankings and classifications

Original Use Case: examples/use_case_5_comparative_stability_analysis.py
Dependencies Removed: Direct MCP import (inlined core functionality), optional plotting dependencies

Usage:
    python mcp_scripts/comparative_stability_analysis.py --input variants.csv --wt_fasta wt.fasta --output comparative.csv

Example:
    python mcp_scripts/comparative_stability_analysis.py --input examples/data/variants.csv --wt_fasta examples/data/wt.fasta --output results/comparative_analysis.csv
"""

# ==============================================================================
# Minimal Imports (only essential packages)
# ==============================================================================
import argparse
import os
import sys
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple
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
    "input_format": "csv",
    "output_format": "csv",
    "progress_bar": True,
    "show_analysis": True,
    "save_enhanced": True,
    "save_report": True,
    "stable_threshold": -1.0,
    "unstable_threshold": 1.0
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
def load_csv_input(file_path: Path) -> pd.DataFrame:
    """Load CSV input file and validate required columns."""
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        if 'seq' not in df.columns:
            raise ValueError(f"CSV must contain 'seq' column. Available: {list(df.columns)}")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")


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


def classify_stability(ddG_value: float, thresholds: Tuple[float, float] = (-1.0, 1.0)) -> str:
    """
    Classify stability based on ΔΔG value.

    Args:
        ddG_value: The ΔΔG value
        thresholds: Tuple of (stabilizing_threshold, destabilizing_threshold)

    Returns:
        String classification
    """
    stable_thresh, unstable_thresh = thresholds

    if ddG_value < stable_thresh:
        return "Highly Stabilizing"
    elif ddG_value < 0:
        return "Stabilizing"
    elif ddG_value == 0:
        return "Neutral"
    elif ddG_value < unstable_thresh:
        return "Destabilizing"
    else:
        return "Highly Destabilizing"


def save_results(data: pd.DataFrame, file_path: Path) -> None:
    """Save results to CSV file."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(file_path, index=False)


def save_report(results_df: pd.DataFrame, file_path: Path, metadata: Dict[str, Any]) -> None:
    """Save detailed analysis report."""
    stability_counts = results_df['Stability_Class'].value_counts()
    correlation = np.corrcoef(results_df['ddG'], results_df['dTm'])[0, 1]

    with open(file_path, 'w') as f:
        f.write("COMPARATIVE STABILITY ANALYSIS REPORT\n")
        f.write("=====================================\n\n")
        f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input File: {metadata['input_file']}\n")
        f.write(f"Wild-type FASTA: {metadata['wt_fasta_file']}\n")
        f.write(f"Total Variants: {len(results_df)}\n\n")

        f.write("STABILITY STATISTICS\n")
        f.write("-------------------\n")
        f.write(f"ΔΔG (kcal/mol):\n")
        f.write(f"  Mean: {results_df['ddG'].mean():.4f}\n")
        f.write(f"  Std Dev: {results_df['ddG'].std():.4f}\n")
        f.write(f"  Median: {results_df['ddG'].median():.4f}\n")
        f.write(f"  Range: {results_df['ddG'].min():.4f} to {results_df['ddG'].max():.4f}\n\n")

        f.write(f"ΔTm (°C):\n")
        f.write(f"  Mean: {results_df['dTm'].mean():.4f}\n")
        f.write(f"  Std Dev: {results_df['dTm'].std():.4f}\n")
        f.write(f"  Median: {results_df['dTm'].median():.4f}\n")
        f.write(f"  Range: {results_df['dTm'].min():.4f} to {results_df['dTm'].max():.4f}\n\n")

        f.write("STABILITY CLASSIFICATION\n")
        f.write("-----------------------\n")
        for stability_class, count in stability_counts.items():
            percentage = (count / len(results_df)) * 100
            f.write(f"{stability_class}: {count} ({percentage:.1f}%)\n")

        f.write(f"\nCorrelation between ΔΔG and ΔTm: {correlation:.4f}\n")


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
def run_comparative_stability_analysis(
    input_file: Union[str, Path],
    wt_fasta_file: Union[str, Path],
    output_file: Optional[Union[str, Path]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Comparative stability analysis using SPIRED-Stab with rankings and classifications.

    Args:
        input_file: Path to CSV file with 'seq' column containing protein sequences
        wt_fasta_file: Path to wild-type sequence FASTA file
        output_file: Path to save output CSV (optional, auto-generated if not provided)
        config: Configuration dict (uses DEFAULT_CONFIG if not provided)
        **kwargs: Override specific config parameters

    Returns:
        Dict containing:
            - result: DataFrame with predictions and analysis (id, seq, ddG, dTm, classifications, ranks)
            - output_file: Path to output file (if saved)
            - enhanced_file: Path to enhanced output file (if saved)
            - report_file: Path to analysis report (if saved)
            - metadata: Execution metadata

    Example:
        >>> result = run_comparative_stability_analysis("variants.csv", "wt.fasta", "output.csv")
        >>> print(result['metadata']['stability_counts'])
    """
    # Setup
    input_file = Path(input_file)
    wt_fasta_file = Path(wt_fasta_file)
    config = {**DEFAULT_CONFIG, **(config or {}), **kwargs}

    # Load input data
    df = load_csv_input(input_file)
    wt_sequence = load_wt_fasta(wt_fasta_file)

    print(f"Loaded {len(df)} sequences from {input_file}")
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
    iterator = tqdm.tqdm(df.iterrows(), total=len(df), desc="Comparative analysis") if config['progress_bar'] else df.iterrows()

    for idx, row in iterator:
        mut_seq = str(row['seq'])
        seq_id = f"seq_{idx}"
        if 'id' in df.columns:
            seq_id = str(row['id'])

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

    # Add stability classifications
    thresholds = (config['stable_threshold'], config['unstable_threshold'])
    results_df['Stability_Class'] = results_df['ddG'].apply(
        lambda x: classify_stability(x, thresholds)
    )

    # Add rankings
    results_df['ddG_Rank'] = results_df['ddG'].rank(ascending=True)  # Lower is better (more stable)
    results_df['dTm_Rank'] = results_df['dTm'].rank(ascending=False)  # Higher is better

    # Calculate percentiles
    results_df['ddG_Percentile'] = results_df['ddG'].rank(pct=True) * 100
    results_df['dTm_Percentile'] = results_df['dTm'].rank(pct=True) * 100

    # Save basic output if requested
    output_path = None
    if output_file:
        output_path = Path(output_file)
        # Save basic results (original format)
        basic_df = results_df[['id', 'seq', 'ddG', 'dTm']].copy()
        save_results(basic_df, output_path)
        print(f"Basic results saved to: {output_path}")

    # Save enhanced results if enabled
    enhanced_path = None
    if config['save_enhanced'] and output_file:
        enhanced_path = Path(str(output_path).replace('.csv', '_enhanced.csv'))
        save_results(results_df, enhanced_path)
        print(f"Enhanced results saved to: {enhanced_path}")

    # Analysis
    stability_counts = results_df['Stability_Class'].value_counts()
    correlation = np.corrcoef(results_df['ddG'], results_df['dTm'])[0, 1]

    # Show analysis if enabled
    if config['show_analysis']:
        print(f"\n=== COMPARATIVE STABILITY ANALYSIS ===")
        print(f"Total variants analyzed: {len(results_df)}")

        print(f"\nStability Statistics:")
        print(f"ΔΔG: {results_df['ddG'].mean():.4f} ± {results_df['ddG'].std():.4f}")
        print(f"     Range: {results_df['ddG'].min():.4f} to {results_df['ddG'].max():.4f}")
        print(f"ΔTm: {results_df['dTm'].mean():.4f} ± {results_df['dTm'].std():.4f}")
        print(f"     Range: {results_df['dTm'].min():.4f} to {results_df['dTm'].max():.4f}")

        print(f"\nStability Classification:")
        for stability_class, count in stability_counts.items():
            percentage = (count / len(results_df)) * 100
            print(f"  {stability_class}: {count} ({percentage:.1f}%)")

        print(f"\nTop 5 Most Stable Variants (lowest ΔΔG):")
        top_stable = results_df.nsmallest(5, 'ddG')[['id', 'ddG', 'dTm', 'Stability_Class']]
        print(top_stable.to_string(index=False))

        print(f"\nTop 5 Least Stable Variants (highest ΔΔG):")
        least_stable = results_df.nlargest(5, 'ddG')[['id', 'ddG', 'dTm', 'Stability_Class']]
        print(least_stable.to_string(index=False))

        print(f"\nCorrelation between ΔΔG and ΔTm: {correlation:.4f}")

    # Save detailed analysis report if enabled
    report_path = None
    if config['save_report'] and output_file:
        report_path = Path(str(output_path).replace('.csv', '_report.txt'))
        save_report(results_df, report_path, {
            'input_file': str(input_file),
            'wt_fasta_file': str(wt_fasta_file)
        })
        print(f"Detailed report saved to: {report_path}")

    return {
        "result": results_df,
        "output_file": str(output_path) if output_path else None,
        "enhanced_file": str(enhanced_path) if enhanced_path else None,
        "report_file": str(report_path) if report_path else None,
        "metadata": {
            "input_file": str(input_file),
            "wt_fasta_file": str(wt_fasta_file),
            "num_variants": len(results_df),
            "device": device,
            "config": config,
            "mean_ddG": results_df['ddG'].mean(),
            "std_ddG": results_df['ddG'].std(),
            "mean_dTm": results_df['dTm'].mean(),
            "std_dTm": results_df['dTm'].std(),
            "correlation_ddG_dTm": correlation,
            "stability_counts": stability_counts.to_dict()
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
    parser.add_argument('--input', '-i', required=True, help='Input CSV file with sequences')
    parser.add_argument('--wt_fasta', '-w', required=True, help='Wild-type FASTA file')
    parser.add_argument('--output', '-o', help='Output CSV file path')
    parser.add_argument('--device', '-d', default='cpu', help='Device (cuda:0, cpu)')
    parser.add_argument('--config', '-c', help='Config file (JSON)')
    parser.add_argument('--stable_threshold', type=float, default=-1.0, help='Threshold for highly stabilizing mutations')
    parser.add_argument('--unstable_threshold', type=float, default=1.0, help='Threshold for highly destabilizing mutations')
    parser.add_argument('--no_progress', action='store_true', help='Disable progress bar')
    parser.add_argument('--no_analysis', action='store_true', help='Disable analysis output')
    parser.add_argument('--no_enhanced', action='store_true', help='Disable enhanced output file')
    parser.add_argument('--no_report', action='store_true', help='Disable report file')

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
        'stable_threshold': args.stable_threshold,
        'unstable_threshold': args.unstable_threshold,
        'progress_bar': not args.no_progress,
        'show_analysis': not args.no_analysis,
        'save_enhanced': not args.no_enhanced,
        'save_report': not args.no_report
    }

    # Auto-generate output file if not provided
    output_file = args.output
    if not output_file:
        input_path = Path(args.input)
        output_file = input_path.parent / f"{input_path.stem}_comparative_analysis.csv"

    # Run analysis
    try:
        result = run_comparative_stability_analysis(
            input_file=args.input,
            wt_fasta_file=args.wt_fasta,
            output_file=output_file,
            config=config,
            **config_overrides
        )

        print(f"\n✅ Success: {result['output_file']}")
        print(f"Processed {result['metadata']['num_variants']} variants")
        print(f"Mean ΔΔG: {result['metadata']['mean_ddG']:.4f} ± {result['metadata']['std_ddG']:.4f}")
        print(f"Correlation (ΔΔG vs ΔTm): {result['metadata']['correlation_ddG_dTm']:.4f}")
        if result['enhanced_file']:
            print(f"Enhanced results: {result['enhanced_file']}")
        if result['report_file']:
            print(f"Detailed report: {result['report_file']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()