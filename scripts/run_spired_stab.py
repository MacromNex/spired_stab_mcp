#!/usr/bin/env python3
"""
SPIRED Stability Prediction Script

This script runs SPIRED-Stab for predicting protein stability changes (ddG and dTm).
It compares wild-type and mutant protein sequences to predict stability effects.
"""

import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from Bio import SeqIO


def setup_aa_dict():
    """Setup amino acid dictionary."""
    aa_dict = {
        "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
        "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
        "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
        "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
        "X": "ALA"
    }
    return aa_dict


def load_models(repo_path, model_dir, device="cpu"):
    """Load all required models."""
    # Add repository to path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from scripts.model import SPIRED_Stab
    from scripts.utils_train_valid import getStabDataTest

    print("Loading SPIRED-Stab model...")
    model = SPIRED_Stab(device_list=[device, device, device, device])
    model_path = os.path.join(model_dir, "SPIRED-Stab.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    print("Loading ESM-2 650M model...")
    esm2_650M, _ = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
    esm2_650M.to(device)
    esm2_650M.eval()
    print("ESM-2 650M loaded!")

    print("Loading ESM-2 3B model...")
    esm2_3B, esm2_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t36_3B_UR50D")
    esm2_3B.to(device)
    esm2_3B.eval()
    esm2_batch_converter = esm2_alphabet.get_batch_converter()
    print("ESM-2 3B loaded!")

    return model, esm2_3B, esm2_650M, esm2_batch_converter, getStabDataTest


def write_pdb(pdb_path, seq, xyz_coords, aa_dict):
    """Write PDB file for CA-only structure."""
    with open(pdb_path, "w") as f:
        for i in range(len(seq)):
            amino_acid = aa_dict[seq[i]]
            xyz_ca = xyz_coords[i, ...]
            x, y, z = (round(float(xyz_ca[0]), 3), round(float(xyz_ca[1]), 3), round(float(xyz_ca[2]), 3))
            f.write("ATOM  {:>5} {:<4} {} A{:>4}    {:>8.3f}{:>8.3f}{:>8.3f}{:>6.2f}{:>6.2f}          {:>2}  \n".format(
                int(i + 1), "CA", amino_acid, int(i + 1), x, y, z, 1.0, 0.0, "C"
            ))


def identify_mutations(wt_seq, mut_seq, device='cpu'):
    """Identify mutation positions and details."""
    mut_pos_torch_list = torch.tensor((np.array(list(wt_seq)) != np.array(list(mut_seq))).astype(int).tolist(), device=device)
    mutations = []
    for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)):
        if wt != mut:
            mutations.append(f"{wt}{i+1}{mut}")
    return mut_pos_torch_list, mutations


def predict_stability(fasta_file, output_folder, repo_path, model_dir, device="cpu"):
    """Run SPIRED stability predictions."""
    # Setup dictionaries
    aa_dict = setup_aa_dict()

    # Load models
    model, esm2_3B, esm2_650M, esm2_batch_converter, getStabDataTest = load_models(repo_path, model_dir, device)

    # Load wild-type and mutant sequences
    print(f"\nReading paired sequences from: {fasta_file}")
    sequences = list(SeqIO.parse(fasta_file, "fasta"))

    if len(sequences) != 2:
        print(f"Error: Expected exactly 2 sequences (wild-type and mutant), but found {len(sequences)}")
        sys.exit(1)

    wt_seq = str(sequences[0].seq)
    mut_seq = str(sequences[1].seq)

    print(f"Wild-type sequence (length: {len(wt_seq)}):")
    print(f"  {wt_seq}")
    print(f"\nMutant sequence (length: {len(mut_seq)}):")
    print(f"  {mut_seq}")

    # Identify mutations
    mut_pos_torch_list, mutations = identify_mutations(wt_seq, mut_seq, device)
    num_mutations = mut_pos_torch_list.sum().item()
    print(f"\nNumber of mutations: {num_mutations}")
    for i, (wt, mut) in enumerate(zip(wt_seq, mut_seq)):
        if wt != mut:
            print(f"  Position {i+1}: {wt} -> {mut}")

    # Run stability predictions
    print("\nStarting stability predictions...")
    with torch.no_grad():
        # Create output directories
        os.makedirs(f"{output_folder}/wt/CA_structure", exist_ok=True)
        os.makedirs(f"{output_folder}/wt/GDFold2", exist_ok=True)
        os.makedirs(f"{output_folder}/mut/CA_structure", exist_ok=True)
        os.makedirs(f"{output_folder}/mut/GDFold2", exist_ok=True)

        # Generate embeddings for wild-type
        print("  - Generating wild-type embeddings...")
        f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(wt_seq, esm2_3B, esm2_650M, esm2_batch_converter, device=device)
        wt_data = {"target_tokens": target_tokens, "esm2-3B": f1d_esm2_3B, "embedding": f1d_esm2_650M}

        # Generate embeddings for mutant
        print("  - Generating mutant embeddings...")
        f1d_esm2_3B, f1d_esm2_650M, target_tokens = getStabDataTest(mut_seq, esm2_3B, esm2_650M, esm2_batch_converter, device=device)
        mut_data = {"target_tokens": target_tokens, "esm2-3B": f1d_esm2_3B, "embedding": f1d_esm2_650M}

        # Run SPIRED-Stab model
        print("  - Running SPIRED-Stab stability prediction...")
        ddG, dTm, wt_features, mut_features = model(wt_data, mut_data, mut_pos_torch_list)

        # Save stability predictions
        print("  - Saving stability predictions...")
        stability_df = pd.DataFrame({"ddG": ddG.item(), "dTm": dTm.item()}, index=[0])
        stability_df.to_csv(f"{output_folder}/pred.csv", index=False)
        print(f"    - ddG (Gibbs free energy change): {ddG.item():.3f} kcal/mol")
        print(f"    - dTm (Melting temperature change): {dTm.item():.3f} °C")

        # Process wild-type structure
        print("\n  - Processing wild-type structures...")
        Predxyz_4thBlock_6thlayer = wt_features["Predxyz"]["4th"][-1]
        plddt_value = wt_features["Plddt"]["4th"][-1][0]
        plddt_value_L = torch.mean(plddt_value, dim=1)
        plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
        plddt_value_top8 = plddt_value[plddt_top8_idx, :]
        xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
        xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
        phi_psi_1D = wt_features["phi_psi_1D"][0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
        plddt_top8_idx = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
        plddt_value_top8 = plddt_value_top8.cpu().detach().numpy().astype(np.float32)

        # Save wild-type NPZ
        np.savez(
            f"{output_folder}/wt/GDFold2/input.npz",
            reference=plddt_top8_idx,
            translation=xyz_top8,
            dihedrals=phi_psi_1D,
            plddt=plddt_value_top8
        )

        # Write wild-type PDB files
        N, L, _ = xyz_top8.shape
        if N > 8:
            N = 8

        print(f"    - Writing {N} wild-type PDB files...")
        for n in range(N):
            pdb_path = f"{output_folder}/wt/CA_structure/{n}.pdb"
            write_pdb(pdb_path, wt_seq, xyz_top8[n, ...], aa_dict)
            print(f"      - Saved {pdb_path} (pLDDT: {plddt_value_L[plddt_top8_idx[n]]:.3f})")

        # Process mutant structure
        print("\n  - Processing mutant structures...")
        Predxyz_4thBlock_6thlayer = mut_features["Predxyz"]["4th"][-1]
        plddt_value = mut_features["Plddt"]["4th"][-1][0]
        plddt_value_L = torch.mean(plddt_value, dim=1)
        plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
        plddt_value_top8 = plddt_value[plddt_top8_idx, :]
        xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
        xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
        phi_psi_1D = mut_features["phi_psi_1D"][0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
        plddt_top8_idx = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
        plddt_value_top8 = plddt_value_top8.cpu().detach().numpy().astype(np.float32)

        # Save mutant NPZ
        np.savez(
            f"{output_folder}/mut/GDFold2/input.npz",
            reference=plddt_top8_idx,
            translation=xyz_top8,
            dihedrals=phi_psi_1D,
            plddt=plddt_value_top8
        )

        # Write mutant PDB files
        N, L, _ = xyz_top8.shape
        if N > 8:
            N = 8

        print(f"    - Writing {N} mutant PDB files...")
        for n in range(N):
            pdb_path = f"{output_folder}/mut/CA_structure/{n}.pdb"
            write_pdb(pdb_path, mut_seq, xyz_top8[n, ...], aa_dict)
            print(f"      - Saved {pdb_path} (pLDDT: {plddt_value_L[plddt_top8_idx[n]]:.3f})")

    print("\nAll stability predictions completed successfully!")

    # Display summary
    print("\n" + "="*70)
    print("STABILITY PREDICTION SUMMARY")
    print("="*70)
    print(f"\nMutations: {', '.join(mutations)}")
    print(f"\nStability Changes:")
    stability_results = pd.read_csv(f"{output_folder}/pred.csv")
    print(f"  ddG (Gibbs free energy): {stability_results['ddG'].values[0]:.3f} kcal/mol")
    print(f"  dTm (Melting temperature): {stability_results['dTm'].values[0]:.3f} °C")
    print(f"\nWild-Type Structures:")
    print(f"  Sequence length: {len(wt_seq)} residues")
    print(f"  CA structures: {output_folder}/wt/CA_structure/{{0-7}}.pdb")
    print(f"  GDFold2 input: {output_folder}/wt/GDFold2/input.npz")
    print(f"\nMutant Structures:")
    print(f"  Sequence length: {len(mut_seq)} residues")
    print(f"  CA structures: {output_folder}/mut/CA_structure/{{0-7}}.pdb")
    print(f"  GDFold2 input: {output_folder}/mut/GDFold2/input.npz")
    print(f"\nStability predictions: {output_folder}/pred.csv")


def main():
    # Get default model directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "..", "data", "model")

    parser = argparse.ArgumentParser(description="SPIRED Stability Prediction")
    parser.add_argument("--fasta", required=True, help="Input FASTA file with wild-type and mutant sequences")
    parser.add_argument("--output", required=True, help="Output directory for predictions")
    parser.add_argument("--repo", required=True, help="Path to SPIRED-Fitness repository")
    parser.add_argument("--model-dir", default=default_model_dir, help=f"Directory containing model files (default: data/model)")
    parser.add_argument("--device", default="cuda:0", help="Device to use (cpu or cuda) [default: cuda:0]")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.fasta):
        print(f"Error: FASTA file not found: {args.fasta}")
        sys.exit(1)

    if not os.path.exists(args.repo):
        print(f"Error: Repository path not found: {args.repo}")
        sys.exit(1)

    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Run predictions
    predict_stability(args.fasta, args.output, args.repo, args.model_dir, args.device)


if __name__ == "__main__":
    main()
