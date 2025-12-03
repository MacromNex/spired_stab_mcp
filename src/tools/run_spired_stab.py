"""
Protein stability prediction using SPIRED-Stab for mutation analysis from sequences, FASTA, or CSV files.

This MCP Server provides 1 tool:
1. spired_predict_mutation_stability: Predicts stability changes (ddG, dTm) and generates 3D structures for protein mutations

All tools extracted from https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED-Stab.py
"""

# Standard imports
from typing import Annotated, Any
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
from fastmcp import FastMCP
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import sys
from loguru import logger

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("RUN_SPIRED_STAB_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("RUN_SPIRED_STAB_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Repository path for imports
REPO_PATH = PROJECT_ROOT / "repo" / "SPIRED-Fitness"

# Add repository to sys.path if not already present
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# Import SPIRED-Fitness modules
from scripts.model import SPIRED_Stab

# Import shared ESM model loader
from tools.esm_model_loader import esm_manager

# MCP server instance
run_spired_stab_mcp = FastMCP(name="run_spired_stab")

# Amino acid dictionary for PDB file generation
AA_DICT = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "ALA"
}


def create_fasta_from_pair(wt_seq: str, mut_seq: str, wt_id: str = "WT", mut_id: str = "MUT") -> Path:
    """Create a temporary FASTA file from wild-type and mutant sequences."""
    fasta_file = INPUT_DIR / f"temp_stab_{timestamp}.fasta"
    records = [
        SeqRecord(Seq(wt_seq), id=wt_id, description=""),
        SeqRecord(Seq(mut_seq), id=mut_id, description="")
    ]
    SeqIO.write(records, str(fasta_file), "fasta")
    return fasta_file


def load_sequence_pairs_from_csv(csv_path: str, id_column: str = "id", wt_column: str = "wt_sequence", mut_column: str = "mut_sequence") -> list[tuple[str, str, str]]:
    """Load wild-type and mutant sequence pairs from CSV file."""
    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in CSV. Available columns: {list(df.columns)}")

    if wt_column not in df.columns:
        raise ValueError(f"Wild-type sequence column '{wt_column}' not found in CSV. Available columns: {list(df.columns)}")

    if mut_column not in df.columns:
        raise ValueError(f"Mutant sequence column '{mut_column}' not found in CSV. Available columns: {list(df.columns)}")

    pairs = []
    for _, row in df.iterrows():
        pair_id = str(row[id_column])
        wt_seq = str(row[wt_column]).strip()
        mut_seq = str(row[mut_column]).strip()
        pairs.append((pair_id, wt_seq, mut_seq))

    return pairs


def getStabDataTestWithDevice(seq, ESM2_3B, ESM2_650M, esm2_batch_converter, device='cuda:0'):
    """
    Get stability data test embeddings with proper device handling.

    This is a device-aware version of getStabDataTest that ensures all tensors
    are on the correct device to avoid CUDA device mismatch errors.
    """
    logger.debug(f"Generating stability embeddings for sequence of length {len(seq)} on device {device}")

    with torch.no_grad():
        _, _, target_tokens = esm2_batch_converter([('', seq)])
        # Move target_tokens to the correct device
        target_tokens = target_tokens.to(device)

        logger.debug("Running ESM2-3B model...")
        results = ESM2_3B(target_tokens, repr_layers=range(37), need_head_weights=False, return_contacts=False)
        f1d_esm2_3B = torch.stack([v for _, v in sorted(results["representations"].items())], dim=2)
        f1d_esm2_3B = f1d_esm2_3B[:, 1:-1]
        f1d_esm2_3B = f1d_esm2_3B.to(dtype=torch.float32)

        logger.debug("Running ESM2-650M model...")
        result_esm2_650m = ESM2_650M(target_tokens, repr_layers=[33], return_contacts=False)
        f1d_esm2_650M = result_esm2_650m['representations'][33][0, 1:-1, :].unsqueeze(0)

        logger.debug("Stability embeddings generated successfully")

    return f1d_esm2_3B, f1d_esm2_650M, target_tokens[:, 1:-1]


@run_spired_stab_mcp.tool
def spired_predict_mutation_stability(
    wt_sequence: Annotated[str | None, "Wild-type protein sequence as string. Must be used with mut_sequence. Mutually exclusive with fasta_path and csv_path."] = None,
    mut_sequence: Annotated[str | None, "Mutant protein sequence as string. Must be used with wt_sequence. Mutually exclusive with fasta_path and csv_path."] = None,
    pair_id: Annotated[str, "Identifier for the sequence pair when using wt_sequence and mut_sequence parameters"] = "protein",
    fasta_path: Annotated[str | None, "Path to input FASTA file containing two sequences: wild-type (first) and mutant (second). Mutually exclusive with sequences and csv_path."] = None,
    csv_path: Annotated[str | None, "Path to CSV file containing wild-type and mutant sequence pairs. Mutually exclusive with sequences and fasta_path."] = None,
    csv_id_column: Annotated[str, "Column name for pair IDs in CSV file"] = "id",
    csv_wt_column: Annotated[str, "Column name for wild-type sequences in CSV file"] = "wt_sequence",
    csv_mut_column: Annotated[str, "Column name for mutant sequences in CSV file"] = "mut_sequence",
    out_prefix: Annotated[str | None, "Output file prefix for stability predictions and structures"] = None,
    device: Annotated[str, "Device to use for computation (cpu or cuda:0)"] = "cuda:0",
) -> dict:
    """
    Predict protein stability changes from mutations using SPIRED-Stab deep learning model.

    Input can be:
    - A pair of sequences (via 'wt_sequence' and 'mut_sequence' parameters)
    - A FASTA file with 2 sequences (via 'fasta_path' parameter)
    - A CSV file with columns for pairs (via 'csv_path' parameter with column specifications)

    Output is stability metrics (ddG, dTm) and 3D protein structures.
    """
    # Input validation - exactly one input method must be provided
    has_sequences = (wt_sequence is not None and mut_sequence is not None)
    has_fasta = fasta_path is not None
    has_csv = csv_path is not None

    input_methods = sum([has_sequences, has_fasta, has_csv])
    if input_methods == 0:
        raise ValueError("Must provide one of: (wt_sequence + mut_sequence), fasta_path, or csv_path")
    if input_methods > 1:
        raise ValueError("Only one input method allowed: (wt_sequence + mut_sequence), fasta_path, or csv_path")

    # Validate sequences are provided together
    if (wt_sequence is None) != (mut_sequence is None):
        raise ValueError("Both wt_sequence and mut_sequence must be provided together")

    # Convert all inputs to list of (id, wt_seq, mut_seq) tuples
    temp_fasta = None
    if has_sequences:
        # Sequence pair provided
        sequence_pairs = [(pair_id, wt_sequence, mut_sequence)]
    elif has_csv:
        # CSV file provided
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        sequence_pairs = load_sequence_pairs_from_csv(csv_path, csv_id_column, csv_wt_column, csv_mut_column)
    else:
        # FASTA file provided
        fasta_file = Path(fasta_path)
        if not fasta_file.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
        sequences = list(SeqIO.parse(str(fasta_file), "fasta"))
        if len(sequences) < 2:
            raise ValueError(f"FASTA file must contain exactly 2 sequences (wild-type and mutant), found {len(sequences)}")
        sequence_pairs = [("pair", str(sequences[0].seq), str(sequences[1].seq))]

    # Setup output paths
    if out_prefix is None:
        out_prefix = f"spired_stab_{timestamp}"

    output_base_dir = OUTPUT_DIR / out_prefix
    output_base_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting mutation stability prediction on device: {device}")

    # Load model weights path
    model_path = REPO_PATH / "model" / "SPIRED-Stab.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"SPIRED-Stab model weights not found: {model_path}")

    # Load SPIRED-Stab model
    logger.info("Loading SPIRED-Stab model...")
    model = SPIRED_Stab(device_list=[device, device, device, device])
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    logger.info("SPIRED-Stab model loaded")

    # Load ESM-2 models using shared loader (avoids redundant loading)
    logger.info("Loading ESM models via shared loader...")
    esm2_3B, esm2_650M, esm2_batch_converter = esm_manager.get_models_for_spired_stab(device)
    logger.info("ESM models loaded")

    # Track outputs
    all_artifacts = []
    all_results = []

    # Process each sequence pair
    for seq_pair_id, wt_seq, mut_seq in sequence_pairs:
        # Validate sequence lengths
        if len(wt_seq) != len(mut_seq):
            raise ValueError(f"Wild-type and mutant sequences for '{seq_pair_id}' must have the same length (WT: {len(wt_seq)}, Mut: {len(mut_seq)})")

        # Create output directory for this pair
        output_dir = output_base_dir / seq_pair_id if len(sequence_pairs) > 1 else output_base_dir
        (output_dir / "wt" / "CA_structure").mkdir(parents=True, exist_ok=True)
        (output_dir / "wt" / "GDFold2").mkdir(parents=True, exist_ok=True)
        (output_dir / "mut" / "CA_structure").mkdir(parents=True, exist_ok=True)
        (output_dir / "mut" / "GDFold2").mkdir(parents=True, exist_ok=True)

        # Identify mutation positions
        mut_pos_torch_list = torch.tensor((np.array(list(wt_seq)) != np.array(list(mut_seq))).astype(int).tolist(), device=device)
        num_mutations = mut_pos_torch_list.sum().item()

        # Identify specific mutations for reporting
        mutations = []
        for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)):
            if wt != mut:
                mutations.append(f"{wt}{i+1}{mut}")

        # Run stability predictions
        logger.info(f"Processing sequence pair: {seq_pair_id}")
        with torch.no_grad():
            # Generate embeddings for wild-type (with device-aware handling)
            logger.debug(f"Generating wild-type embeddings for {seq_pair_id}")
            f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTestWithDevice(wt_seq, esm2_3B, esm2_650M, esm2_batch_converter, device=device)
            wt_data = {"target_tokens": target_tokens, "esm2-3B": f1d_esm2_3B, "embedding": f1d_esm2_650M}

            # Generate embeddings for mutant (with device-aware handling)
            logger.debug(f"Generating mutant embeddings for {seq_pair_id}")
            f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTestWithDevice(mut_seq, esm2_3B, esm2_650M, esm2_batch_converter, device=device)
            mut_data = {"target_tokens": target_tokens, "esm2-3B": f1d_esm2_3B, "embedding": f1d_esm2_650M}

            # Run SPIRED-Stab model
            ddG, dTm, wt_features, mut_features = model(wt_data, mut_data, mut_pos_torch_list)

            # Save stability predictions
            stability_df = pd.DataFrame({"ddG": ddG.item(), "dTm": dTm.item()}, index=[0])
            pred_csv_path = output_dir / "pred.csv"
            stability_df.to_csv(pred_csv_path, index=False)

            # Process wild-type structure
            Predxyz_4thBlock_6thlayer = wt_features["Predxyz"]["4th"][-1]
            plddt_value = wt_features["Plddt"]["4th"][-1][0]
            plddt_value_L = torch.mean(plddt_value, dim=1)
            plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
            plddt_value_top8 = plddt_value[plddt_top8_idx, :]
            xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
            xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
            phi_psi_1D = wt_features["phi_psi_1D"][0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
            plddt_top8_idx_np = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
            plddt_value_top8_np = plddt_value_top8.cpu().detach().numpy().astype(np.float32)

            # Save wild-type NPZ
            wt_npz_path = output_dir / "wt" / "GDFold2" / "input.npz"
            np.savez(
                str(wt_npz_path),
                reference=plddt_top8_idx_np,
                translation=xyz_top8,
                dihedrals=phi_psi_1D,
                plddt=plddt_value_top8_np
            )

            # Write wild-type PDB files
            N, L, _ = xyz_top8.shape
            if N > 8:
                N = 8

            wt_pdb_paths = []
            for n in range(N):
                xyz_L = xyz_top8[n, ...]
                pdb_path = output_dir / "wt" / "CA_structure" / f"{n}.pdb"
                with open(pdb_path, "w") as f:
                    for i in range(L):
                        amino_acid = AA_DICT[wt_seq[i]]
                        xyz_ca = xyz_L[i, ...]
                        x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]), 3), round(float(xyz_ca[2]), 3))
                        f.write("ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n".format(
                            int(i + 1), "CA", amino_acid, int(i + 1), x, y, z, 1.0, 0.0, "C"
                        ))
                wt_pdb_paths.append(str(pdb_path.resolve()))

            # Process mutant structure
            Predxyz_4thBlock_6thlayer = mut_features["Predxyz"]["4th"][-1]
            plddt_value = mut_features["Plddt"]["4th"][-1][0]
            plddt_value_L = torch.mean(plddt_value, dim=1)
            plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
            plddt_value_top8 = plddt_value[plddt_top8_idx, :]
            xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
            xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
            phi_psi_1D = mut_features["phi_psi_1D"][0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
            plddt_top8_idx_np = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
            plddt_value_top8_np = plddt_value_top8.cpu().detach().numpy().astype(np.float32)

            # Save mutant NPZ
            mut_npz_path = output_dir / "mut" / "GDFold2" / "input.npz"
            np.savez(
                str(mut_npz_path),
                reference=plddt_top8_idx_np,
                translation=xyz_top8,
                dihedrals=phi_psi_1D,
                plddt=plddt_value_top8_np
            )

            # Write mutant PDB files
            N, L, _ = xyz_top8.shape
            if N > 8:
                N = 8

            mut_pdb_paths = []
            for n in range(N):
                xyz_L = xyz_top8[n, ...]
                pdb_path = output_dir / "mut" / "CA_structure" / f"{n}.pdb"
                with open(pdb_path, "w") as f:
                    for i in range(L):
                        amino_acid = AA_DICT[mut_seq[i]]
                        xyz_ca = xyz_L[i, ...]
                        x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]), 3), round(float(xyz_ca[2]), 3))
                        f.write("ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n".format(
                            int(i + 1), "CA", amino_acid, int(i + 1), x, y, z, 1.0, 0.0, "C"
                        ))
                mut_pdb_paths.append(str(pdb_path.resolve()))

        # Prepare artifacts list for this pair
        artifacts = [
            {
                "description": f"Stability predictions for {seq_pair_id} (ddG, dTm)",
                "path": str(pred_csv_path.resolve())
            },
            {
                "description": f"Wild-type GDFold2 input for {seq_pair_id}",
                "path": str(wt_npz_path.resolve())
            },
            {
                "description": f"Mutant GDFold2 input for {seq_pair_id}",
                "path": str(mut_npz_path.resolve())
            }
        ]

        # Add first WT and mutant PDB as examples
        if wt_pdb_paths:
            artifacts.append({
                "description": f"Wild-type structure for {seq_pair_id} (top confidence)",
                "path": wt_pdb_paths[0]
            })

        if mut_pdb_paths:
            artifacts.append({
                "description": f"Mutant structure for {seq_pair_id} (top confidence)",
                "path": mut_pdb_paths[0]
            })

        all_artifacts.extend(artifacts)

        # Store results
        mutation_str = ", ".join(mutations) if mutations else "None"
        all_results.append({
            "pair_id": seq_pair_id,
            "num_mutations": num_mutations,
            "mutations": mutation_str,
            "ddG": ddG.item(),
            "dTm": dTm.item()
        })

    # Save summary CSV if multiple pairs
    if len(sequence_pairs) > 1:
        summary_df = pd.DataFrame(all_results)
        summary_path = output_base_dir / "stability_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        all_artifacts.append({
            "description": "Stability predictions summary",
            "path": str(summary_path.resolve())
        })

    # Create summary message
    if len(sequence_pairs) == 1:
        result = all_results[0]
        message = f"Predicted stability for {result['num_mutations']} mutation(s): {result['mutations']}. ddG={result['ddG']:.3f} kcal/mol, dTm={result['dTm']:.3f} °C"
        logger.info(f"Stability prediction completed: {message}")
    else:
        avg_ddG = np.mean([r['ddG'] for r in all_results])
        avg_dTm = np.mean([r['dTm'] for r in all_results])
        message = f"Predicted stability for {len(sequence_pairs)} sequence pairs. Average ddG={avg_ddG:.3f} kcal/mol, Average dTm={avg_dTm:.3f} °C"
        logger.info(f"Stability prediction completed for {len(sequence_pairs)} pairs")
        logger.info(f"Average ddG: {avg_ddG:.3f} kcal/mol, Average dTm: {avg_dTm:.3f} °C")

    logger.info(f"Output directory: {output_base_dir}")

    return {
        "message": message,
        "reference": "https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED-Stab.py",
        "artifacts": all_artifacts
    }
