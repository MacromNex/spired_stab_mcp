#!/usr/bin/env python3
"""
SPIRED Structure Prediction Script

This script runs SPIRED-Fitness structure prediction for protein sequences.
It generates 3D structures (CA-only) and prepares input for GDFold2.
"""

import os
import sys
import argparse
import torch
import itertools
import numpy as np
import pandas as pd
from Bio import SeqIO


def setup_dictionaries():
    """Setup amino acid and mutation dictionaries."""
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

    return amino_acid_list, amino_acid_dict, double_mut_dict, double_mut_dict_inverse, aa_dict


def load_models(repo_path, model_dir, device="cpu"):
    """Load all required models."""
    # Add repository to path
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)

    from scripts.model import SPIRED_Fitness_Union
    from scripts.utils_train_valid import getDataTest

    print("Loading SPIRED-Fitness model...")
    model = SPIRED_Fitness_Union(device_list=[device, device, device, device])
    model_path = os.path.join(model_dir, "SPIRED-Fitness.pth")
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

    print("Loading ESM-1v models...")
    esm1v_1, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_1")
    esm1v_2, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_2")
    esm1v_3, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_3")
    esm1v_4, _ = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_4")
    esm1v_5, esm1v_alphabet = torch.hub.load("facebookresearch/esm:main", "esm1v_t33_650M_UR90S_5")
    esm1v_1.to(device)
    esm1v_2.to(device)
    esm1v_3.to(device)
    esm1v_4.to(device)
    esm1v_5.to(device)
    esm1v_1.eval()
    esm1v_2.eval()
    esm1v_3.eval()
    esm1v_4.eval()
    esm1v_5.eval()
    esm1v_batch_converter = esm1v_alphabet.get_batch_converter()
    print("All ESM-1v models loaded!")

    return (model, esm2_3B, esm2_650M, esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5,
            esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter, getDataTest)


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


def predict_structures(fasta_file, output_folder, repo_path, model_dir, device="cpu"):
    """Run SPIRED structure predictions."""
    # Setup dictionaries
    amino_acid_list, amino_acid_dict, double_mut_dict, double_mut_dict_inverse, aa_dict = setup_dictionaries()

    # Load models
    (model, esm2_3B, esm2_650M, esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5,
     esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter, getDataTest) = load_models(repo_path, model_dir, device)

    # Load FASTA sequences
    print(f"\nReading FASTA file: {fasta_file}")
    fasta_dict = {}
    for i in SeqIO.parse(fasta_file, "fasta"):
        fasta_dict[i.description] = str(i.seq)
        print(f"Loaded sequence: {i.description} (length: {len(str(i.seq))})")
    print(f"Total sequences loaded: {len(fasta_dict)}")

    # Run predictions
    print("\nStarting structure predictions...")
    with torch.no_grad():
        for protein_name in fasta_dict.keys():
            print(f"\nProcessing protein: {protein_name}")

            # Create output directories
            os.makedirs(f"{output_folder}/{protein_name}/CA_structure", exist_ok=True)
            os.makedirs(f"{output_folder}/{protein_name}/GDFold2", exist_ok=True)

            seq = fasta_dict[protein_name]

            # Get ESM embeddings and predictions
            print("  - Generating ESM embeddings...")
            f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits, target_tokens = getDataTest(
                seq, esm2_3B, esm2_650M, esm1v_1, esm1v_2, esm1v_3, esm1v_4, esm1v_5,
                esm1v_batch_converter, esm1v_alphabet, esm2_batch_converter
            )

            # Run SPIRED model
            print("  - Running SPIRED structure prediction...")
            single_pred, double_pred, Predxyz, PredCadistavg, Plddt, ca, cb, omega, theta, phi, phi_psi_1D, f1d_cycle, f2d_cycle = model(
                target_tokens, f1d_esm2_3B, f1d_esm2_650M, esm1v_single_logits, esm1v_double_logits
            )

            # Extract top 8 predictions based on pLDDT
            print("  - Extracting top 8 structures by pLDDT...")
            Predxyz_4thBlock_6thlayer = Predxyz["4th"][-1]
            plddt_value = Plddt["4th"][-1][0]
            plddt_value_L = torch.mean(plddt_value, dim=1)
            plddt_top8_idx = torch.topk(plddt_value_L, 8)[-1]
            plddt_value_top8 = plddt_value[plddt_top8_idx, :]
            xyz_top8 = Predxyz_4thBlock_6thlayer[0, :, plddt_top8_idx, :]
            xyz_top8 = xyz_top8.permute(1, 2, 0).cpu().detach().numpy().astype(np.float32)
            phi_psi_1D = phi_psi_1D[0].permute(1, 0).cpu().detach().numpy().astype(np.float32)
            plddt_top8_idx = plddt_top8_idx.cpu().detach().numpy().astype(np.int32)
            plddt_value_top8 = plddt_value_top8.cpu().detach().numpy().astype(np.float32)

            # Save NPZ file for GDFold2
            print("  - Saving NPZ file for GDFold2...")
            np.savez(
                f"{output_folder}/{protein_name}/GDFold2/input.npz",
                reference=plddt_top8_idx,
                translation=xyz_top8,
                dihedrals=phi_psi_1D,
                plddt=plddt_value_top8
            )

            # Write PDB files
            N, L, _ = xyz_top8.shape
            if N > 8:
                N = 8

            print(f"  - Writing {N} PDB structure files...")
            for n in range(N):
                pdb_path = f"{output_folder}/{protein_name}/CA_structure/{n}.pdb"
                write_pdb(pdb_path, seq, xyz_top8[n, ...], aa_dict)
                print(f"    - Saved {pdb_path} (pLDDT: {plddt_value_L[plddt_top8_idx[n]]:.3f})")

            print(f"  - Completed predictions for {protein_name}")

    print("\nAll predictions completed successfully!")

    # Display summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    for protein_name in fasta_dict.keys():
        print(f"\nProtein: {protein_name}")
        print(f"  Sequence length: {len(fasta_dict[protein_name])} residues")
        print(f"  Output directory: {output_folder}/{protein_name}")
        print(f"  CA structures: {output_folder}/{protein_name}/CA_structure/{{0-7}}.pdb")
        print(f"  GDFold2 input: {output_folder}/{protein_name}/GDFold2/input.npz")


def main():
    # Get default model directory relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_dir = os.path.join(script_dir, "..", "data", "model")

    parser = argparse.ArgumentParser(description="SPIRED Structure Prediction")
    parser.add_argument("--fasta", required=True, help="Input FASTA file with protein sequences")
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
    predict_structures(args.fasta, args.output, args.repo, args.model_dir, args.device)


if __name__ == "__main__":
    main()
