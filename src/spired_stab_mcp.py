#!/usr/bin/env python3
"""
SPIRED-Stab MCP Server

This MCP server provides protein stability prediction using SPIRED-Stab.
It supports both CSV and FASTA format inputs for variants.
"""

import os
import sys
import tempfile
import pandas as pd
from pathlib import Path
from typing import Optional
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from fastmcp import FastMCP
import tqdm

# Add scripts directory to path to import the model
script_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(script_dir))

# Change working directory to scripts for imports
original_cwd = os.getcwd()
os.chdir(str(script_dir))

from src.model import SPIRED_Stab
from src.utils_train_valid import getStabDataTest

# Restore original working directory
os.chdir(original_cwd)
import torch
import numpy as np
from loguru import logger

# Initialize FastMCP server
mcp = FastMCP("spired_stab_mcp")

# Global variables for model caching
_model_cache = {
    'spired_stab': None,
    'esm2_650M': None,
    'esm2_3B': None,
    'esm2_batch_converter': None,
    'device': None
}


def _initialize_models(device: str = 'cuda:0'):
    """
    Initialize and cache the SPIRED-Stab and ESM models.

    Args:
        device: Device to load models on (e.g., 'cuda:0', 'cpu')
    """
    global _model_cache

    if _model_cache['spired_stab'] is not None and _model_cache['device'] == device:
        logger.info("Models already loaded, using cached versions")
        return

    logger.info(f"Loading SPIRED-Stab model on device: {device}")

    # Load SPIRED-Stab model
    model = SPIRED_Stab(device_list=[device, device, device, device])
    model_path = script_dir / "data" / "model" / "SPIRED-Stab.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(str(model_path)))
    model.to(device)
    model.eval()

    # Load ESM-2 650M model
    logger.info("Loading ESM-2 650M model...")
    esm2_650M, _ = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
    esm2_650M.to(device)
    esm2_650M.eval()

    # Load ESM-2 3B model
    logger.info("Loading ESM-2 3B model...")
    esm2_3B, esm2_alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D')
    esm2_3B.to(device)
    esm2_3B.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()

    # Cache models
    _model_cache['spired_stab'] = model
    _model_cache['esm2_650M'] = esm2_650M
    _model_cache['esm2_3B'] = esm2_3B
    _model_cache['esm2_batch_converter'] = esm2_batch_converter
    _model_cache['device'] = device

    logger.info("All models loaded successfully")


def _convert_csv_to_fasta(csv_file: str, seq_column: str = 'seq') -> str:
    """
    Convert a CSV file with sequences to a FASTA file.

    Args:
        csv_file: Path to the CSV file
        seq_column: Name of the column containing sequences

    Returns:
        Path to the temporary FASTA file
    """
    df = pd.read_csv(csv_file)

    if seq_column not in df.columns:
        raise ValueError(f"Column '{seq_column}' not found in CSV. Available columns: {list(df.columns)}")

    # Create temporary FASTA file
    temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)

    records = []
    for idx, row in df.iterrows():
        seq = str(row[seq_column])
        # Use index as ID if no ID column exists
        seq_id = f"seq_{idx}"
        if 'id' in df.columns:
            seq_id = str(row['id'])
        elif 'Mutations' in df.columns:
            seq_id = str(row['Mutations']).replace('/', '_').replace(' ', '_')

        record = SeqRecord(Seq(seq), id=seq_id, description="")
        records.append(record)

    SeqIO.write(records, temp_fasta, "fasta")
    temp_fasta.close()

    logger.info(f"Converted CSV to FASTA: {len(records)} sequences")
    return temp_fasta.name


def _predict_stability(fasta_file: str, wt_fasta_file: str, device: str) -> pd.DataFrame:
    """
    Run SPIRED-Stab prediction on a FASTA file.

    Args:
        fasta_file: Path to the variant sequences FASTA file
        wt_fasta_file: Path to the wild-type FASTA file
        device: Device to run inference on

    Returns:
        DataFrame with predictions
    """
    _initialize_models(device)

    model = _model_cache['spired_stab']
    esm2_650M = _model_cache['esm2_650M']
    esm2_3B = _model_cache['esm2_3B']
    esm2_batch_converter = _model_cache['esm2_batch_converter']

    # Load wild-type sequence
    if not os.path.exists(wt_fasta_file):
        raise FileNotFoundError(f"Wild-type FASTA file not found: {wt_fasta_file}")

    wt_seq = str(list(SeqIO.parse(wt_fasta_file, 'fasta'))[0].seq)
    logger.info(f"Loaded wild-type sequence (length: {len(wt_seq)})")

    # Embed wt_seq
    f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(
        wt_seq, esm2_3B, esm2_650M, esm2_batch_converter, device=device
    )
    wt_data = {
        'target_tokens': target_tokens,
        'esm2-3B': f1d_esm2_3B,
        'embedding': f1d_esm2_650M
    }

    # Load variant sequences
    id_list = []
    seq_list = []
    for record in SeqIO.parse(fasta_file, 'fasta'):
        id_list.append(record.id)
        seq_list.append(str(record.seq))

    logger.info(f"Processing {len(seq_list)} variant sequences")

    # Run predictions with progress bar
    results = []
    for seq_id, mut_seq in tqdm.tqdm(zip(id_list, seq_list), total=len(id_list), ncols=80):
        mut_pos_torch_list = torch.tensor(
            (np.array(list(wt_seq)) != np.array(list(mut_seq))).astype(int).tolist()
        )

        # Embed mut_seq
        f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(
            mut_seq, esm2_3B, esm2_650M, esm2_batch_converter, device=device
        )
        mut_data = {
            'target_tokens': target_tokens,
            'esm2-3B': f1d_esm2_3B,
            'embedding': f1d_esm2_650M
        }

        with torch.no_grad():
            ddG, dTm, wt_features, mut_features = model(wt_data, mut_data, mut_pos_torch_list)
            results.append({
                'id': seq_id,
                'seq': mut_seq,
                'ddG': ddG.item(),
                'dTm': dTm.item()
            })

    return pd.DataFrame(results)


@mcp.tool()
def predict_stability(
    variant_file: str,
    wt_fasta_file: Optional[str] = None,
    output_file: Optional[str] = None,
    device: str = 'cuda:0'
) -> str:
    """
    Predict protein stability using SPIRED-Stab.

    This tool predicts the stability changes (ddG and dTm) for protein variants
    compared to a wild-type sequence. It supports both CSV and FASTA input formats.

    Args:
        variant_file: Path to the variant sequences file (CSV or FASTA format).
                     For CSV: must contain a 'seq' column with protein sequences.
                     For FASTA: standard FASTA format with sequence IDs and sequences.
        wt_fasta_file: Path to the wild-type sequence FASTA file.
                      If not provided, will look for 'wt.fasta' in the same directory as variant_file.
        output_file: Path to save the prediction results (CSV format).
                    If not provided, will save as '{variant_file}_spired_stab_pred.csv'
        device: Device to run inference on (default: 'cuda:0').
               Use 'cpu' if no GPU is available, or 'cuda:1' for a different GPU.

    Returns:
        A message indicating success and the path to the output file.
        The output CSV contains: id, seq, ddG (stability change), dTm (melting temperature change)

    Example usage:
        - CSV input: predict_stability('/path/to/variants.csv', '/path/to/wt.fasta')
        - FASTA input: predict_stability('/path/to/variants.fasta', '/path/to/wt.fasta')
        - Default wt: predict_stability('/path/to/variants.fasta')  # uses wt.fasta in same directory
    """
    return predict_stability_direct(variant_file, wt_fasta_file, output_file, device)


def predict_stability_direct(
    variant_file: str,
    wt_fasta_file: Optional[str] = None,
    output_file: Optional[str] = None,
    device: str = 'cuda:0'
) -> str:
    """
    Direct function interface for predict_stability (not an MCP tool).

    This is the same function as the MCP tool but can be imported and called directly.
    """
    try:
        logger.info(f"Starting SPIRED-Stab prediction for: {variant_file}")

        # Validate variant file exists
        if not os.path.exists(variant_file):
            return f"Error: Variant file not found: {variant_file}"

        # Determine input format and convert if needed
        variant_file_path = Path(variant_file)
        is_csv = variant_file_path.suffix.lower() == '.csv'

        if is_csv:
            logger.info("Input format: CSV - converting to FASTA")
            fasta_file = _convert_csv_to_fasta(variant_file)
            cleanup_fasta = True
        else:
            fasta_file = variant_file
            cleanup_fasta = False

        # Determine wild-type FASTA file path
        if wt_fasta_file is None:
            wt_fasta_file = str(variant_file_path.parent / 'wt.fasta')
            logger.info(f"Using default wild-type file: {wt_fasta_file}")

        # Determine output file path
        if output_file is None:
            if is_csv:
                output_file = str(variant_file_path).replace('.csv', '_spired_stab_pred.csv')
            else:
                output_file = f"{variant_file}_spired_stab_pred.csv"

        # Run prediction
        results_df = _predict_stability(fasta_file, wt_fasta_file, device)

        # Save results
        results_df.to_csv(output_file, index=False)
        logger.info(f"Predictions saved to: {output_file}")

        # Clean up temporary FASTA file if created
        if cleanup_fasta:
            os.unlink(fasta_file)

        # Generate summary
        summary = f"""
SPIRED-Stab prediction completed successfully!

Input file: {variant_file}
Wild-type file: {wt_fasta_file}
Output file: {output_file}

Processed {len(results_df)} variants.

Prediction summary:
- Mean ddG: {results_df['ddG'].mean():.4f} ± {results_df['ddG'].std():.4f}
- Mean dTm: {results_df['dTm'].mean():.4f} ± {results_df['dTm'].std():.4f}

Results saved to: {output_file}
"""
        return summary

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return f"Error during SPIRED-Stab prediction: {str(e)}"


if __name__ == "__main__":
    mcp.run()
