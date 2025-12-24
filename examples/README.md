# SPIRED-Stab Use Cases and Examples

This directory contains comprehensive examples demonstrating different use cases for protein stability prediction using SPIRED-Stab.

## Use Cases Overview

| Script | Description | Input Format | Example Command |
|--------|-------------|--------------|-----------------|
| `use_case_1_csv_stability_prediction.py` | Basic CSV-based stability prediction | CSV with 'seq' column | `python use_case_1_csv_stability_prediction.py --input data/subtilisin/variants.csv` |
| `use_case_2_fasta_stability_prediction.py` | FASTA-based stability prediction | FASTA sequences | `python use_case_2_fasta_stability_prediction.py --input data/subtilisin/sequences.fasta` |
| `use_case_3_single_variant_prediction.py` | Single variant analysis | Individual sequence or mutations | `python use_case_3_single_variant_prediction.py --mutation "I31L,S3T"` |
| `use_case_4_batch_mutation_analysis.py` | Systematic mutation scanning | Position lists or specific mutations | `python use_case_4_batch_mutation_analysis.py --positions 31,67,124` |
| `use_case_5_comparative_stability_analysis.py` | Comprehensive analysis with rankings | CSV with sequences | `python use_case_5_comparative_stability_analysis.py --input data/subtilisin/variants.csv --create_plots` |

## Example Data

### Subtilisin Dataset
Located in `data/subtilisin/`:
- `variants.csv`: CSV file with 1000+ protein variants and experimental data
- `sequences.fasta`: FASTA format of the same sequences
- `wt.fasta`: Wild-type subtilisin sequence for comparison

### Model Files
Located in `data/model/` (symbolic link to `../../scripts/data/model/`):
- `SPIRED-Stab.pth`: Pre-trained SPIRED-Stab model weights (507MB)
- `SPIRED-Fitness.pth`: SPIRED-Fitness model weights (510MB)

**Note**: Model files must be downloaded separately from [Zenodo](https://zenodo.org/records/10675405).

## Quick Start Examples

### 1. Basic CSV Prediction
```bash
# Predict stability for all variants in CSV file
python use_case_1_csv_stability_prediction.py --input data/subtilisin/variants.csv
```

### 2. FASTA Prediction
```bash
# Predict stability for FASTA sequences
python use_case_2_fasta_stability_prediction.py --input data/subtilisin/sequences.fasta
```

### 3. Single Mutation Analysis
```bash
# Analyze a specific mutation
python use_case_3_single_variant_prediction.py --mutation "I31L"

# Analyze multiple mutations
python use_case_3_single_variant_prediction.py --mutation "I31L,S3T"
```

### 4. Mutation Scanning
```bash
# Scan all possible mutations at specific positions
python use_case_4_batch_mutation_analysis.py --positions 31,67,124

# Analyze specific mutations
python use_case_4_batch_mutation_analysis.py --mutations "I31L,H67Y,M124L"
```

### 5. Comprehensive Analysis
```bash
# Complete analysis with visualization
python use_case_5_comparative_stability_analysis.py --input data/subtilisin/variants.csv --create_plots
```

## Output Formats

All scripts generate CSV files with the following columns:
- `id`: Sequence/variant identifier
- `seq`: Protein sequence
- `ddG`: Stability change (ΔΔG) in kcal/mol
- `dTm`: Melting temperature change (ΔTm) in °C

Additional outputs may include:
- `*_enhanced.csv`: Results with stability classifications and rankings
- `*_report.txt`: Detailed analysis reports
- `*_summary.txt`: Summary statistics
- `stability_plots/`: Visualization plots (if matplotlib/seaborn available)

## Interpretation Guide

### ΔΔG (Stability Change)
- **ΔΔG < -1.0**: Highly Stabilizing (much more stable than wild-type)
- **-1.0 ≤ ΔΔG < 0**: Stabilizing (more stable than wild-type)
- **ΔΔG = 0**: Neutral (similar stability to wild-type)
- **0 < ΔΔG ≤ 1.0**: Destabilizing (less stable than wild-type)
- **ΔΔG > 1.0**: Highly Destabilizing (much less stable than wild-type)

### ΔTm (Melting Temperature Change)
- **Positive ΔTm**: Higher melting temperature (more thermostable)
- **Negative ΔTm**: Lower melting temperature (less thermostable)

## Requirements

### Environment
- Python 3.11+ environment with PyTorch, BioPython, pandas
- CUDA-capable GPU recommended (can use CPU with `--device cpu`)

### Model Files
Download from [Zenodo](https://zenodo.org/records/10675405) and place in `data/model/`:
- SPIRED-Stab.pth
- SPIRED-Fitness.pth

### Optional Dependencies
For visualization (use_case_5):
```bash
pip install matplotlib seaborn
```

## Performance Notes

- **GPU vs CPU**: GPU is ~10-20x faster than CPU
- **Batch Size**: Larger batches are more efficient but use more memory
- **Memory**: Large protein variants (>1000 residues) require more GPU memory
- **Speed**: ~1-5 sequences per second on modern GPU

## Troubleshooting

### Common Issues
1. **Model file not found**: Download model files from Zenodo
2. **CUDA out of memory**: Use `--device cpu` or reduce batch size
3. **Import errors**: Ensure environment is activated and dependencies installed
4. **Sequence length mismatch**: Check that variant and wild-type sequences have similar lengths

### Debug Options
Add these flags for debugging:
- `--device cpu`: Use CPU instead of GPU
- Reduce input file size for testing

## Citation

If you use SPIRED-Stab in your research, please cite:

```
@article{SPIRED-Stab,
    title={SPIRED-Stab: A Deep Learning Approach for Protein Stability Prediction},
    author={[Authors]},
    journal={[Journal]},
    year={[Year]}
}
```