#!/usr/bin/env python3
"""
Standalone SpiredStab stability prediction runner.
Run from the scripts/ directory.
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import pandas as pd
import torch
import numpy as np
import tqdm
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Import from local src directory
from src.model import SPIRED_Stab
from src.utils_train_valid import getStabDataTest


def convert_csv_to_fasta(csv_file: str, seq_column: str = 'seq') -> str:
    """Convert a CSV file with sequences to a FASTA file."""
    df = pd.read_csv(csv_file)

    if seq_column not in df.columns:
        raise ValueError(f"Column '{seq_column}' not found in CSV. Available columns: {list(df.columns)}")

    temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)

    records = []
    for idx, row in df.iterrows():
        seq = str(row[seq_column])
        seq_id = f"seq_{idx}"
        if 'id' in df.columns:
            seq_id = str(row['id'])
        elif 'mutations' in df.columns:
            seq_id = str(row['mutations']).replace('/', '_').replace(' ', '_')

        record = SeqRecord(Seq(seq), id=seq_id, description="")
        records.append(record)

    SeqIO.write(records, temp_fasta, "fasta")
    temp_fasta.close()

    print(f"Converted CSV to FASTA: {len(records)} sequences")
    return temp_fasta.name


def predict_stability(fasta_file: str, wt_fasta_file: str, device: str,
                     esm2_3B, esm2_650M, esm2_batch_converter, model) -> pd.DataFrame:
    """Run SPIRED-Stab prediction on a FASTA file."""

    # Load wild-type sequence
    if not os.path.exists(wt_fasta_file):
        raise FileNotFoundError(f"Wild-type FASTA file not found: {wt_fasta_file}")

    wt_seq = str(list(SeqIO.parse(wt_fasta_file, 'fasta'))[0].seq)
    print(f"Loaded wild-type sequence (length: {len(wt_seq)})")

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

    print(f"Processing {len(seq_list)} variant sequences")

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


def main():
    parser = argparse.ArgumentParser(description="Run SPIRED-Stab stability prediction")
    parser.add_argument("-i", "--input", required=True, help="Input CSV with variant sequences")
    parser.add_argument("-w", "--wt_fasta", required=True, help="Wild-type reference FASTA")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument("-d", "--device", default="cuda:0", help="Device (cuda:0 or cpu)")
    args = parser.parse_args()

    device = args.device
    script_dir = Path(__file__).parent

    print(f"Input file: {args.input}")
    print(f"Wild-type FASTA: {args.wt_fasta}")
    print(f"Output file: {args.output}")
    print(f"Device: {device}")
    print()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.wt_fasta):
        print(f"Error: Wild-type FASTA file not found: {args.wt_fasta}")
        sys.exit(1)

    # Load models
    print("Loading SPIRED-Stab model...")
    model = SPIRED_Stab(device_list=[device, device, device, device])
    model_path = script_dir / "data" / "model" / "SPIRED-Stab.pth"
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    print("  Model loaded")

    # Load ESM-2 650M model
    print("Loading ESM-2 650M model...")
    esm2_650M, _ = torch.hub.load('facebookresearch/esm:main', 'esm2_t33_650M_UR50D')
    esm2_650M.to(device)
    esm2_650M.eval()
    print("  ESM-2 650M loaded")

    # Load ESM-2 3B model
    print("Loading ESM-2 3B model...")
    esm2_3B, esm2_alphabet = torch.hub.load('facebookresearch/esm:main', 'esm2_t36_3B_UR50D')
    esm2_3B.to(device)
    esm2_3B.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()
    print("  ESM-2 3B loaded")

    print()

    # Convert CSV to FASTA if needed
    input_path = Path(args.input)
    is_csv = input_path.suffix.lower() == '.csv'

    if is_csv:
        print("Input format: CSV - converting to FASTA")
        fasta_file = convert_csv_to_fasta(args.input)
        cleanup_fasta = True
    else:
        fasta_file = args.input
        cleanup_fasta = False

    # Run prediction
    results_df = predict_stability(
        fasta_file, args.wt_fasta, device,
        esm2_3B, esm2_650M, esm2_batch_converter, model
    )

    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nPredictions saved to: {args.output}")

    # Clean up temporary FASTA file if created
    if cleanup_fasta:
        os.unlink(fasta_file)

    # Print summary
    print(f"\nSummary:")
    print(f"  Processed {len(results_df)} variants")
    print(f"  Mean ddG: {results_df['ddG'].mean():.4f} +/- {results_df['ddG'].std():.4f}")
    print(f"  Mean dTm: {results_df['dTm'].mean():.4f} +/- {results_df['dTm'].std():.4f}")
    print(f"  Stabilizing (ddG < 0): {(results_df['ddG'] < 0).sum()}")
    print(f"  Destabilizing (ddG > 0): {(results_df['ddG'] > 0).sum()}")


if __name__ == "__main__":
    main()
