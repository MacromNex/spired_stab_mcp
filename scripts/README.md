# SPIRED-Fitness Python Scripts

This directory contains Python scripts adapted from the Jupyter notebooks for running SPIRED-Fitness predictions.

## Scripts Overview

### 1. run_spired.py
Structure prediction for protein sequences. Generates 3D CA-only structures and prepares input for GDFold2.

**Usage:**
```bash
# Using default GPU (cuda:0)
python run_spired.py \
  --fasta <input.fasta> \
  --output <output_dir> \
  --repo <path_to_SPIRED-Fitness_repo>

# Using CPU instead
python run_spired.py \
  --fasta <input.fasta> \
  --output <output_dir> \
  --repo <path_to_SPIRED-Fitness_repo> \
  --device cpu
```

**Arguments:**
- `--fasta`: Input FASTA file with protein sequences
- `--output`: Output directory for predictions
- `--repo`: Path to SPIRED-Fitness repository
- `--model-dir`: Directory containing model files [default: data/model]
- `--device`: Device to use (cpu or cuda:0) [default: cuda:0]

**Outputs:**
- `{output}/{protein_name}/CA_structure/{0-7}.pdb`: Top 8 predicted structures
- `{output}/{protein_name}/GDFold2/input.npz`: Input for GDFold2

### 2. run_spired_fitness.py
Fitness landscape prediction for protein sequences. Predicts single and double mutation effects.

**Usage:**
```bash
# Using default GPU (cuda:0)
python run_spired_fitness.py \
  --fasta <input.fasta> \
  --output <output_dir> \
  --repo <path_to_SPIRED-Fitness_repo> \
  --top-k 2000

# Using CPU instead
python run_spired_fitness.py \
  --fasta <input.fasta> \
  --output <output_dir> \
  --repo <path_to_SPIRED-Fitness_repo> \
  --device cpu \
  --top-k 2000
```

**Arguments:**
- `--fasta`: Input FASTA file with protein sequences
- `--output`: Output directory for predictions
- `--repo`: Path to SPIRED-Fitness repository
- `--model-dir`: Directory containing model files [default: data/model]
- `--device`: Device to use (cpu or cuda:0) [default: cuda:0]
- `--top-k`: Number of top/bottom double mutations to save [default: 2000]

**Outputs:**
- `{output}/{protein_name}/CA_structure/{0-7}.pdb`: Top 8 predicted structures
- `{output}/{protein_name}/GDFold2/input.npz`: Input for GDFold2
- `{output}/{protein_name}/single_mut_pred_for_heatmap.csv`: Single mutation heatmap data
- `{output}/{protein_name}/single_mut_pred.csv`: Single mutation predictions
- `{output}/{protein_name}/double_mut_pred_top_k.csv`: Top k double mutations
- `{output}/{protein_name}/double_mut_pred_last_k.csv`: Bottom k double mutations
- `{output}/{protein_name}/features_for_downstream/`: Structural features (3d.pt, plddt.pt)
- `{output}/{protein_name}/double_mut_pred.pt`: Full double mutation tensor

### 3. run_spired_stab.py
Stability prediction for protein mutations. Predicts ddG and dTm changes.

**Usage:**
```bash
# Using default GPU (cuda:0)
python run_spired_stab.py \
  --fasta <input.fasta> \
  --output <output_dir> \
  --repo <path_to_SPIRED-Fitness_repo>

# Using CPU instead
python run_spired_stab.py \
  --fasta <input.fasta> \
  --output <output_dir> \
  --repo <path_to_SPIRED-Fitness_repo> \
  --device cpu

# With specific GPU using CUDA_VISIBLE_DEVICES
CUDA_VISIBLE_DEVICES=1 python run_spired_stab.py \
  --fasta <input.fasta> \
  --output <output_dir> \
  --repo <path_to_SPIRED-Fitness_repo>
```

**Arguments:**
- `--fasta`: Input FASTA file with wild-type and mutant sequences (exactly 2 sequences)
- `--output`: Output directory for predictions
- `--repo`: Path to SPIRED-Fitness repository
- `--model-dir`: Directory containing model files [default: data/model]
- `--device`: Device to use (cpu or cuda:0) [default: cuda:0]

**Outputs:**
- `{output}/pred.csv`: Stability predictions (ddG and dTm)
- `{output}/wt/CA_structure/{0-7}.pdb`: Wild-type structures
- `{output}/wt/GDFold2/input.npz`: Wild-type GDFold2 input
- `{output}/mut/CA_structure/{0-7}.pdb`: Mutant structures
- `{output}/mut/GDFold2/input.npz`: Mutant GDFold2 input

## Requirements

All scripts require:
- PyTorch
- Biopython
- NumPy
- Pandas
- ESM models (downloaded automatically via torch.hub)
- SPIRED-Fitness repository (for code imports)
- Model files (SPIRED-Fitness.pth, SPIRED-Stab.pth) in the model directory

## Examples

### Structure Prediction
```bash
# Using default GPU (cuda:0)
python run_spired.py \
  --fasta /path/to/sequences.fasta \
  --output ./predictions \
  --repo /path/to/SPIRED-Fitness

# Using specific GPU
CUDA_VISIBLE_DEVICES=1 python run_spired.py \
  --fasta /path/to/sequences.fasta \
  --output ./predictions \
  --repo /path/to/SPIRED-Fitness

# Using CPU
python run_spired.py \
  --fasta /path/to/sequences.fasta \
  --output ./predictions \
  --repo /path/to/SPIRED-Fitness \
  --device cpu
```

### Fitness Landscape
```bash
# Using default GPU (cuda:0)
python run_spired_fitness.py \
  --fasta /path/to/sequences.fasta \
  --output ./fitness_predictions \
  --repo /path/to/SPIRED-Fitness \
  --top-k 5000

# Using CPU
python run_spired_fitness.py \
  --fasta /path/to/sequences.fasta \
  --output ./fitness_predictions \
  --repo /path/to/SPIRED-Fitness \
  --device cpu \
  --top-k 5000
```

### Stability Prediction
```bash
# Using default GPU (cuda:0)
python run_spired_stab.py \
  --fasta /path/to/wt_mut_pair.fasta \
  --output ./stability_predictions \
  --repo /path/to/SPIRED-Fitness

# Using specific GPU
CUDA_VISIBLE_DEVICES=1 python run_spired_stab.py \
  --fasta /path/to/wt_mut_pair.fasta \
  --output ./stability_predictions \
  --repo /path/to/SPIRED-Fitness
```

## Notes

- **GPU Acceleration**: All scripts default to `cuda:0` for GPU acceleration
  - Use `--device cpu` to force CPU usage if needed
  - Use `CUDA_VISIBLE_DEVICES=N` to select a specific GPU (where N is the GPU index)
  - When using `CUDA_VISIBLE_DEVICES`, the specified GPU becomes `cuda:0` in PyTorch
- ESM models are downloaded automatically on first run and cached in `~/.cache/torch/hub`
- By default, model files are loaded from `data/model` directory (relative to the scripts location)
- You can specify a custom model directory using `--model-dir` argument
- The `--repo` argument should point to the SPIRED-Fitness repository for code imports
- For stability predictions, the FASTA file must contain exactly 2 sequences: wild-type followed by mutant
- All tensors and models are automatically moved to the specified device for efficient computation
