"""
SPIRED-Fitness protein structure prediction from sequences, FASTA, or CSV files.

This MCP Server provides 1 tool:
1. spired_predict_protein_structures: Predict 3D protein structures from amino acid sequences using SPIRED-Fitness model

All tools extracted from https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED.py
"""

# Standard imports
from typing import Annotated, Any
import os
import sys
import torch
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from fastmcp import FastMCP
import tempfile
from loguru import logger

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
DEFAULT_INPUT_DIR = PROJECT_ROOT / "tmp" / "inputs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tmp" / "outputs"

INPUT_DIR = Path(os.environ.get("SPIRED_INPUT_DIR", DEFAULT_INPUT_DIR))
OUTPUT_DIR = Path(os.environ.get("SPIRED_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

# Ensure directories exist
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Repository path for imports
REPO_PATH = PROJECT_ROOT / "repo" / "SPIRED-Fitness"

# Add repository path to sys.path for imports
if str(REPO_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_PATH))

# Import SPIRED-Fitness modules
from scripts.model import SPIRED_Fitness_Union

# Import shared ESM model loader
from tools.esm_model_loader import esm_manager

# Timestamp for unique outputs
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# MCP server instance
run_spired_mcp = FastMCP(name="run_spired")


def create_fasta_from_sequences(sequences: dict[str, str]) -> Path:
    """Create a temporary FASTA file from sequence dictionary."""
    fasta_file = INPUT_DIR / f"temp_{timestamp}.fasta"
    records = [SeqRecord(Seq(seq), id=name, description="") for name, seq in sequences.items()]
    SeqIO.write(records, str(fasta_file), "fasta")
    return fasta_file


def load_sequences_from_csv(csv_path: str, id_column: str = "id", sequence_column: str = "sequence") -> dict[str, str]:
    """Load sequences from CSV file."""
    df = pd.read_csv(csv_path)

    if id_column not in df.columns:
        raise ValueError(f"ID column '{id_column}' not found in CSV. Available columns: {list(df.columns)}")

    if sequence_column not in df.columns:
        raise ValueError(f"Sequence column '{sequence_column}' not found in CSV. Available columns: {list(df.columns)}")

    sequences = {}
    for _, row in df.iterrows():
        seq_id = str(row[id_column])
        seq = str(row[sequence_column]).strip()
        sequences[seq_id] = seq

    return sequences


def generate_esm1v_logits_with_device(model, batch_converter, alphabet, seq_data, device='cuda:0'):
    """
    Generate ESM-1v logits with proper device handling.

    This is a device-aware version that ensures batch_tokens are on the correct device.
    """
    with torch.no_grad():
        batch_labels, batch_strs, batch_tokens = batch_converter(seq_data)
        seq = seq_data[0][1]

        # Move batch_tokens to the correct device before passing to model
        batch_tokens = batch_tokens.to(device)

        token_probs = torch.log_softmax(model(batch_tokens)['logits'], dim=-1)
        logits_33 = token_probs[0, 1:-1, :].detach().cpu().clone()

        # logits 33 dim -> 20 dim
        amino_acid_list = list('ARNDCQEGHILKMFPSTWYV')
        esm1v_amino_acid_dict = {}
        for i in amino_acid_list:
            esm1v_amino_acid_dict[i] = alphabet.get_idx(i)

        logits_20_single = torch.zeros((logits_33.shape[0], 20))
        for wt_pos, wt_amino_acid in enumerate(seq):
            for mut_pos, mut_amino_acid in enumerate(amino_acid_list):
                logits_20_single[wt_pos, mut_pos] = logits_33[wt_pos, esm1v_amino_acid_dict[mut_amino_acid]] - logits_33[wt_pos, esm1v_amino_acid_dict[wt_amino_acid]]

        logits_400_double = torch.zeros((logits_33.shape[0], logits_33.shape[0], 400))
        for wt_pos_1, wt_amino_acid_1 in enumerate(seq):
            for wt_pos_2, wt_amino_acid_2 in enumerate(seq):
                for mut_pos_1, mut_amino_acid_1 in enumerate(amino_acid_list):
                    for mut_pos_2, mut_amino_acid_2 in enumerate(amino_acid_list):
                        logits_400_double[wt_pos_1, wt_pos_2, mut_pos_1 * 20 + mut_pos_2] = (
                            logits_33[wt_pos_1, esm1v_amino_acid_dict[mut_amino_acid_1]]
                            - logits_33[wt_pos_1, esm1v_amino_acid_dict[wt_amino_acid_1]]
                            + logits_33[wt_pos_2, esm1v_amino_acid_dict[mut_amino_acid_2]]
                            - logits_33[wt_pos_2, esm1v_amino_acid_dict[wt_amino_acid_2]]
                        )

    return logits_20_single, logits_400_double


def getDataTestWithDevice(seq, ESM2_3B, ESM2_650M, ESM1v_1, ESM1v_2, ESM1v_3, ESM1v_4, ESM1v_5,
                          esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter, device='cuda:0'):
    """
    Get ESM embeddings for a sequence with proper device handling.

    This is a device-aware version of getDataTest that ensures all tensors
    are on the correct device to avoid CUDA device mismatch errors.
    """
    logger.debug(f"Generating ESM embeddings for sequence of length {len(seq)} on device {device}")

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

        logger.debug("Running ESM1v models (1-5)...")
        esm1v_single_1, esm1v_double_1 = generate_esm1v_logits_with_device(ESM1v_1, esm1v_batch_converter, esm1v_alphabet, [('', seq)], device=device)
        esm1v_single_2, esm1v_double_2 = generate_esm1v_logits_with_device(ESM1v_2, esm1v_batch_converter, esm1v_alphabet, [('', seq)], device=device)
        esm1v_single_3, esm1v_double_3 = generate_esm1v_logits_with_device(ESM1v_3, esm1v_batch_converter, esm1v_alphabet, [('', seq)], device=device)
        esm1v_single_4, esm1v_double_4 = generate_esm1v_logits_with_device(ESM1v_4, esm1v_batch_converter, esm1v_alphabet, [('', seq)], device=device)
        esm1v_single_5, esm1v_double_5 = generate_esm1v_logits_with_device(ESM1v_5, esm1v_batch_converter, esm1v_alphabet, [('', seq)], device=device)
        esm1v_single_logits = torch.cat([esm1v_single_1, esm1v_single_2, esm1v_single_3, esm1v_single_4, esm1v_single_5], dim=0).unsqueeze(0)
        esm1v_double_logits = torch.cat([esm1v_double_1, esm1v_double_2, esm1v_double_3, esm1v_double_4, esm1v_double_5], dim=0).unsqueeze(0)

        logger.debug("ESM embeddings generated successfully")

    return f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens[:, 1:-1]


@run_spired_mcp.tool
def spired_predict_protein_structures(
    sequence: Annotated[str | None, "Single protein sequence as string (e.g., 'MKLLLLLL...'). Mutually exclusive with fasta_path and csv_path."] = None,
    sequence_id: Annotated[str, "Identifier for the sequence when using 'sequence' parameter"] = "protein",
    fasta_path: Annotated[str | None, "Path to input FASTA file containing protein sequences. Each sequence should have a unique identifier. Mutually exclusive with sequence and csv_path."] = None,
    csv_path: Annotated[str | None, "Path to CSV file containing protein sequences. Must have columns for ID and sequence. Mutually exclusive with sequence and fasta_path."] = None,
    csv_id_column: Annotated[str, "Column name for sequence IDs in CSV file"] = "id",
    csv_sequence_column: Annotated[str, "Column name for sequences in CSV file"] = "sequence",
    model_path: Annotated[str | None, "Path to SPIRED-Fitness model weights file (.pth). If None, uses default model from repository."] = None,
    num_structures: Annotated[int, "Number of top structures to extract based on pLDDT score"] = 8,
    out_prefix: Annotated[str | None, "Output file prefix for organizing results"] = None,
    device: Annotated[str, "Device to use for computation (cpu or cuda:0)"] = "cuda:0",
) -> dict:
    """
    Predict 3D protein structures from amino acid sequences using the SPIRED-Fitness deep learning model.

    Input can be:
    - A single sequence string (via 'sequence' parameter)
    - A FASTA file (via 'fasta_path' parameter)
    - A CSV file (via 'csv_path' parameter with 'csv_id_column' and 'csv_sequence_column')

    Output is top-ranked PDB structure files with confidence scores and NPZ files for GDFold2.
    """
    # Input validation - exactly one input method must be provided
    input_methods = sum([sequence is not None, fasta_path is not None, csv_path is not None])
    if input_methods == 0:
        raise ValueError("Must provide one of: sequence, fasta_path, or csv_path")
    if input_methods > 1:
        raise ValueError("Only one input method allowed: sequence, fasta_path, or csv_path")

    # Convert all inputs to FASTA format
    temp_fasta = None
    if sequence is not None:
        # Single sequence provided
        sequences = {sequence_id: sequence}
        fasta_file = create_fasta_from_sequences(sequences)
        temp_fasta = fasta_file
    elif csv_path is not None:
        # CSV file provided
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        sequences = load_sequences_from_csv(csv_path, csv_id_column, csv_sequence_column)
        fasta_file = create_fasta_from_sequences(sequences)
        temp_fasta = fasta_file
    else:
        # FASTA file provided
        fasta_file = Path(fasta_path)
        if not fasta_file.exists():
            raise FileNotFoundError(f"Input FASTA file not found: {fasta_path}")

    # Setup model path
    if model_path is None:
        model_path = REPO_PATH / "model" / "SPIRED-Fitness.pth"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")

    # Setup output directory
    if out_prefix is None:
        out_prefix = f"spired_prediction_{timestamp}"

    saved_folder = OUTPUT_DIR / out_prefix
    saved_folder.mkdir(parents=True, exist_ok=True)

    # Define amino acid dictionaries (from tutorial)
    amino_acid_list = list("ARNDCQEGHILKMFPSTWYV")
    amino_acid_dict = {}
    for index, value in enumerate(amino_acid_list):
        amino_acid_dict[index] = value

    double_mut_list = list(itertools.product(amino_acid_list, amino_acid_list, repeat=1))
    double_mut_dict = {}
    double_mut_dict_inverse = {}
    for index, value in enumerate(double_mut_list):
        double_mut_dict[index] = "".join(value)
        double_mut_dict_inverse["".join(value)] = index

    aa_dict = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
        "X": "ALA"
    }

    logger.info(f"Starting protein structure prediction on device: {device}")

    # Load SPIRED-Fitness model
    logger.info("Loading SPIRED-Fitness model...")
    model = SPIRED_Fitness_Union(device_list=[device, device, device, device])
    model.load_state_dict(torch.load(str(model_path), map_location=device))
    model.to(device)
    model.eval()
    logger.info("SPIRED-Fitness model loaded")

    # Load ESM models using shared loader (avoids redundant loading)
    logger.info("Loading ESM models via shared loader...")
    esm2_3B, esm2_650M, esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5, esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter = esm_manager.get_models_for_spired(device)
    logger.info("ESM models loaded")

    # Load FASTA sequences
    fasta_dict = {}
    for i in SeqIO.parse(str(fasta_file), "fasta"):
        fasta_dict[i.description] = str(i.seq)

    # Track outputs
    artifacts = []
    results_summary = []

    # Run predictions
    with torch.no_grad():
        for protein_name in fasta_dict.keys():
            # Create output directories
            protein_output_dir = saved_folder / protein_name
            ca_structure_dir = protein_output_dir / "CA_structure"
            gdfold2_dir = protein_output_dir / "GDFold2"
            ca_structure_dir.mkdir(parents=True, exist_ok=True)
            gdfold2_dir.mkdir(parents=True, exist_ok=True)

            seq = fasta_dict[protein_name]

            # Get ESM embeddings and predictions (with device-aware handling)
            logger.info(f"Processing protein: {protein_name} (length: {len(seq)})")
            f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens = getDataTestWithDevice(
                seq, esm2_3B, esm2_650M, esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5,
                esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter, device=device
            )

            # Run SPIRED model
            single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = model(
                target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits
            )

            # Extract top structures based on pLDDT
            Predxyz_4thBlock_6thlayer = Predxyz["4th"][-1]
            plddt_value = Plddt["4th"][-1][0]
            plddt_value_L = torch.mean(plddt_value, dim=1)
            plddt_top_idx = torch.topk(plddt_value_L, num_structures)[-1]
            plddt_value_top = plddt_value[plddt_top_idx, :]
            xyz_top = Predxyz_4thBlock_6thlayer[0, :, plddt_top_idx, :]
            xyz_top = xyz_top.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
            phi_psi_1D_np = phi_psi_1D[0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
            plddt_top_idx_np = plddt_top_idx.cpu().detach().numpy().astype(np.int32)
            plddt_value_top_np = plddt_value_top.cpu().detach().numpy().astype(np.float32)

            # Save NPZ file for GDFold2
            npz_path = gdfold2_dir / "input.npz"
            np.savez(
                str(npz_path),
                reference=plddt_top_idx_np,
                translation=xyz_top,
                dihedrals=phi_psi_1D_np,
                plddt=plddt_value_top_np
            )

            artifacts.append({
                "description": f"GDFold2 input for {protein_name}",
                "path": str(npz_path.resolve())
            })

            # Write PDB files
            N, L, _ = xyz_top.shape
            if N > num_structures:
                N = num_structures

            pdb_files = []
            for n in range(N):
                xyz_L = xyz_top[n, ...]
                pdb_path = ca_structure_dir / f"{n}.pdb"
                with open(pdb_path, "w") as f:
                    for i in range(L):
                        amino_acid = aa_dict[seq[i]]
                        xyz_ca = xyz_L[i, ...]
                        x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]), 3), round(float(xyz_ca[2]), 3))
                        f.write("ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n".format(
                            int(i + 1), "CA", amino_acid, int(i + 1), x, y, z, 1.0, 0.0, "C"
                        ))

                pdb_files.append(str(pdb_path.resolve()))
                artifacts.append({
                    "description": f"{protein_name} structure {n} (pLDDT: {plddt_value_L[plddt_top_idx[n]]:.3f})",
                    "path": str(pdb_path.resolve())
                })

            results_summary.append({
                "protein_name": protein_name,
                "sequence_length": len(seq),
                "num_structures": N,
                "avg_plddt": float(plddt_value_L[plddt_top_idx].mean()),
                "max_plddt": float(plddt_value_L[plddt_top_idx].max()),
                "output_dir": str(protein_output_dir.resolve())
            })

    # Save summary CSV
    summary_df = pd.DataFrame(results_summary)
    summary_path = saved_folder / "prediction_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    artifacts.append({
        "description": "Prediction summary",
        "path": str(summary_path.resolve())
    })

    # Clean up temporary FASTA if created
    if temp_fasta and temp_fasta.exists():
        temp_fasta.unlink()

    logger.info(f"Structure prediction completed for {len(fasta_dict)} protein(s)")
    logger.info(f"Average pLDDT: {summary_df['avg_plddt'].mean():.3f}")
    logger.info(f"Output directory: {saved_folder}")

    return {
        "message": f"Predicted structures for {len(fasta_dict)} protein(s) with avg pLDDT {summary_df['avg_plddt'].mean():.3f}",
        "reference": "https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED.py",
        "artifacts": artifacts
    }
