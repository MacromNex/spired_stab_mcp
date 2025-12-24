#!/usr/bin/env python3
"""
Use Case 5: Comparative Stability Analysis using SPIRED-Stab

This script compares the stability predictions of multiple protein variants
and provides detailed comparative analysis including rankings and classifications.

Usage:
    python examples/use_case_5_comparative_stability_analysis.py --input examples/data.csv --output comparative_analysis.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
# Optional plotting dependencies
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib and/or seaborn not available for plotting")

# Add the project root and scripts to the Python path
project_root = Path(__file__).parent.parent
scripts_dir = project_root / "scripts"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(scripts_dir))

from src.spired_stab_mcp import predict_stability_direct


def classify_stability(ddG_value, thresholds=(-1.0, 1.0)):
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


def create_stability_plots(results_df, output_dir):
    """
    Create visualization plots for stability analysis.

    Args:
        results_df: DataFrame with prediction results
        output_dir: Directory to save plots
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available - skipping plot creation")
        return None

    os.makedirs(output_dir, exist_ok=True)

    plt.style.use('default')
    sns.set_palette("husl")

    # Plot 1: ΔΔG distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 2, 1)
    plt.hist(results_df['ddG'], bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral (ΔΔG=0)')
    plt.xlabel('ΔΔG (kcal/mol)')
    plt.ylabel('Frequency')
    plt.title('Distribution of ΔΔG Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: ΔTm distribution
    plt.subplot(2, 2, 2)
    plt.hist(results_df['dTm'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Neutral (ΔTm=0)')
    plt.xlabel('ΔTm (°C)')
    plt.ylabel('Frequency')
    plt.title('Distribution of ΔTm Values')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: ΔΔG vs ΔTm correlation
    plt.subplot(2, 2, 3)
    plt.scatter(results_df['ddG'], results_df['dTm'], alpha=0.6)
    plt.xlabel('ΔΔG (kcal/mol)')
    plt.ylabel('ΔTm (°C)')
    plt.title('ΔΔG vs ΔTm Correlation')
    plt.grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = np.corrcoef(results_df['ddG'], results_df['dTm'])[0, 1]
    plt.text(0.05, 0.95, f'R = {correlation:.3f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle="round", facecolor='white', alpha=0.8))

    # Plot 4: Stability classification pie chart
    plt.subplot(2, 2, 4)
    stability_counts = results_df['Stability_Class'].value_counts()
    colors = ['green', 'lightgreen', 'gray', 'orange', 'red']
    plt.pie(stability_counts.values, labels=stability_counts.index, autopct='%1.1f%%',
            colors=colors[:len(stability_counts)])
    plt.title('Stability Classification Distribution')

    plt.tight_layout()
    plot_file = os.path.join(output_dir, 'stability_analysis_plots.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_file


def main():
    parser = argparse.ArgumentParser(
        description="Comparative stability analysis using SPIRED-Stab"
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
        default="comparative_stability_analysis.csv",
        help="Path to output CSV file (default: comparative_stability_analysis.csv)"
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda:0",
        help="Device to use for computation (default: cuda:0, use 'cpu' if no GPU)"
    )
    parser.add_argument(
        "--create_plots",
        action="store_true",
        help="Create visualization plots (requires matplotlib and seaborn)"
    )
    parser.add_argument(
        "--stable_threshold",
        type=float,
        default=-1.0,
        help="Threshold for highly stabilizing mutations (default: -1.0)"
    )
    parser.add_argument(
        "--unstable_threshold",
        type=float,
        default=1.0,
        help="Threshold for highly destabilizing mutations (default: 1.0)"
    )

    args = parser.parse_args()

    # Convert to absolute paths
    input_file = os.path.abspath(args.input)
    output_file = os.path.abspath(args.output)

    # Validate input file
    if not os.path.exists(input_file):
        print(f"Error: Input file not found: {input_file}")
        sys.exit(1)

    # Check input file format
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

    if not os.path.exists(wt_fasta_file):
        print(f"Error: Wild-type FASTA file not found: {wt_fasta_file}")
        sys.exit(1)

    print(f"Input file: {input_file}")
    print(f"Wild-type FASTA: {wt_fasta_file}")
    print(f"Output file: {output_file}")
    print(f"Device: {args.device}")
    print()

    try:
        # Run SPIRED-Stab prediction
        print("Starting SPIRED-Stab prediction...")
        result = predict_stability_direct(
            variant_file=input_file,
            wt_fasta_file=wt_fasta_file,
            output_file=output_file,
            device=args.device
        )

        print("Prediction completed successfully!")
        print(result)

        # Load and analyze results
        if os.path.exists(output_file):
            results_df = pd.read_csv(output_file)

            # Add stability classifications
            thresholds = (args.stable_threshold, args.unstable_threshold)
            results_df['Stability_Class'] = results_df['ddG'].apply(
                lambda x: classify_stability(x, thresholds)
            )

            # Add rankings
            results_df['ddG_Rank'] = results_df['ddG'].rank(ascending=True)  # Lower is better (more stable)
            results_df['dTm_Rank'] = results_df['dTm'].rank(ascending=False)  # Higher is better

            # Calculate percentiles
            results_df['ddG_Percentile'] = results_df['ddG'].rank(pct=True) * 100
            results_df['dTm_Percentile'] = results_df['dTm'].rank(pct=True) * 100

            # Save enhanced results
            enhanced_output = output_file.replace('.csv', '_enhanced.csv')
            results_df.to_csv(enhanced_output, index=False)

            print(f"\n=== COMPARATIVE STABILITY ANALYSIS ===")
            print(f"Total variants analyzed: {len(results_df)}")

            print(f"\nStability Statistics:")
            print(f"ΔΔG: {results_df['ddG'].mean():.4f} ± {results_df['ddG'].std():.4f}")
            print(f"     Range: {results_df['ddG'].min():.4f} to {results_df['ddG'].max():.4f}")
            print(f"ΔTm: {results_df['dTm'].mean():.4f} ± {results_df['dTm'].std():.4f}")
            print(f"     Range: {results_df['dTm'].min():.4f} to {results_df['dTm'].max():.4f}")

            print(f"\nStability Classification:")
            stability_counts = results_df['Stability_Class'].value_counts()
            for stability_class, count in stability_counts.items():
                percentage = (count / len(results_df)) * 100
                print(f"  {stability_class}: {count} ({percentage:.1f}%)")

            print(f"\nTop 5 Most Stable Variants (lowest ΔΔG):")
            top_stable = results_df.nsmallest(5, 'ddG')[['id', 'ddG', 'dTm', 'Stability_Class']]
            print(top_stable.to_string(index=False))

            print(f"\nTop 5 Least Stable Variants (highest ΔΔG):")
            least_stable = results_df.nlargest(5, 'ddG')[['id', 'ddG', 'dTm', 'Stability_Class']]
            print(least_stable.to_string(index=False))

            # Correlation analysis
            correlation = np.corrcoef(results_df['ddG'], results_df['dTm'])[0, 1]
            print(f"\nCorrelation between ΔΔG and ΔTm: {correlation:.4f}")

            # Create visualizations if requested
            if args.create_plots:
                try:
                    output_dir = Path(output_file).parent / "stability_plots"
                    plot_file = create_stability_plots(results_df, output_dir)
                    print(f"\nVisualization plots saved to: {plot_file}")
                except ImportError:
                    print("\nWarning: matplotlib and/or seaborn not available for plotting")
                except Exception as e:
                    print(f"\nWarning: Could not create plots: {e}")

            # Save detailed analysis report
            report_file = output_file.replace('.csv', '_report.txt')
            with open(report_file, 'w') as f:
                f.write("COMPARATIVE STABILITY ANALYSIS REPORT\n")
                f.write("=====================================\n\n")
                f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Input File: {input_file}\n")
                f.write(f"Wild-type FASTA: {wt_fasta_file}\n")
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

            print(f"\nDetailed report saved to: {report_file}")
            print(f"Enhanced results saved to: {enhanced_output}")

    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()