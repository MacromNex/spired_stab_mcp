# MCP Scripts

Clean, self-contained scripts extracted from use cases for MCP tool wrapping.

## Design Principles

1. **Minimal Dependencies**: Only essential packages imported
2. **Self-Contained**: Functions inlined where possible
3. **Configurable**: Parameters in config files, not hardcoded
4. **MCP-Ready**: Each script has a main function ready for MCP wrapping

## Scripts

| Script | Description | Repo Dependent | Config |
|--------|-------------|----------------|--------|
| `csv_stability_prediction.py` | Predict stability from CSV | Yes (models) | `configs/csv_stability_config.json` |
| `fasta_stability_prediction.py` | Predict stability from FASTA | Yes (models) | `configs/fasta_stability_config.json` |
| `single_variant_prediction.py` | Single variant analysis | Yes (models) | `configs/single_variant_config.json` |
| `batch_mutation_analysis.py` | Batch mutation analysis | Yes (models) | `configs/batch_mutation_config.json` |
| `comparative_stability_analysis.py` | Comparative analysis with rankings | Yes (models) | `configs/comparative_analysis_config.json` |

## Usage

```bash
# Activate environment (prefer mamba over conda)
mamba activate ./env  # or: conda activate ./env

# Run a script
python mcp_scripts/csv_stability_prediction.py --input examples/data/subtilisin/variants.csv --wt_fasta examples/data/subtilisin/wt.fasta --output results/output.csv

# With custom config
python mcp_scripts/csv_stability_prediction.py --input FILE --wt_fasta FILE --output FILE --config configs/custom.json
```

## Shared Library

Common functions are in `lib/`:
- `io.py`: File loading/saving (CSV, FASTA, JSON)
- `models.py`: Model loading and caching
- `utils.py`: Mutation handling, stability classification

## Configuration

Each script has a corresponding config file in `configs/`:

- `csv_stability_config.json`: CSV prediction settings
- `fasta_stability_config.json`: FASTA prediction settings
- `single_variant_config.json`: Single variant settings
- `batch_mutation_config.json`: Batch analysis settings
- `comparative_analysis_config.json`: Comparative analysis settings
- `default_config.json`: Default settings for all scripts

## For MCP Wrapping (Step 6)

Each script exports a main function that can be wrapped:

```python
from mcp_scripts.csv_stability_prediction import run_csv_stability_prediction

# In MCP tool:
@mcp.tool()
def predict_csv_stability(input_file: str, wt_fasta_file: str, output_file: str = None):
    return run_csv_stability_prediction(input_file, wt_fasta_file, output_file)
```

## Dependencies

### Essential (always required):
- pandas
- numpy
- torch
- biopython

### Model-specific (lazy loaded):
- SPIRED-Stab model code from `scripts/src/`
- ESM-2 models (downloaded automatically)

### Optional:
- matplotlib, seaborn (for plotting in comparative analysis - disabled by default)

## Scripts Overview

### 1. CSV Stability Prediction
- **Path**: `csv_stability_prediction.py`
- **Function**: `run_csv_stability_prediction()`
- **Input**: CSV file with 'seq' column
- **Output**: CSV with ddG and dTm predictions

```bash
python mcp_scripts/csv_stability_prediction.py \
  --input examples/data/subtilisin/variants.csv \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/csv_predictions.csv
```

### 2. FASTA Stability Prediction
- **Path**: `fasta_stability_prediction.py`
- **Function**: `run_fasta_stability_prediction()`
- **Input**: FASTA file with variant sequences
- **Output**: CSV with ddG and dTm predictions + analysis

```bash
python mcp_scripts/fasta_stability_prediction.py \
  --input examples/data/subtilisin/sequences.fasta \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/fasta_predictions.csv
```

### 3. Single Variant Prediction
- **Path**: `single_variant_prediction.py`
- **Function**: `run_single_variant_prediction()`
- **Input**: Mutation string (e.g., "I31L") or complete sequence
- **Output**: CSV with single prediction + interpretation

```bash
python mcp_scripts/single_variant_prediction.py \
  --mutation "I31L" \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/single_variant.csv
```

### 4. Batch Mutation Analysis
- **Path**: `batch_mutation_analysis.py`
- **Function**: `run_batch_mutation_analysis()`
- **Input**: List of positions or specific mutations
- **Output**: CSV + summary file with statistics

```bash
# Specific mutations
python mcp_scripts/batch_mutation_analysis.py \
  --mutations "I31L,H67Y,M124L" \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/batch_analysis.csv

# Saturation mutagenesis at positions
python mcp_scripts/batch_mutation_analysis.py \
  --positions "31,67,124" \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/saturation_analysis.csv
```

### 5. Comparative Stability Analysis
- **Path**: `comparative_stability_analysis.py`
- **Function**: `run_comparative_stability_analysis()`
- **Input**: CSV file with sequences
- **Output**: Multiple files (basic CSV, enhanced CSV, report)

```bash
python mcp_scripts/comparative_stability_analysis.py \
  --input examples/data/subtilisin/variants.csv \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/comparative_analysis.csv
```

## Error Handling

All scripts include robust error handling:
- Input file validation
- Sequence format validation
- Device availability checking
- Model loading verification
- Graceful degradation for optional features

## Performance Notes

- **Model Loading**: Models are cached after first load (~40-90 seconds initial load)
- **CPU Performance**: ~40-50 seconds per variant on CPU
- **GPU Performance**: 10-20x faster than CPU (if available)
- **Memory Usage**: ~4GB RAM during inference
- **Lazy Loading**: Heavy dependencies loaded only when needed

## Testing

```bash
# Test script help
python mcp_scripts/single_variant_prediction.py --help

# Test with small data
python mcp_scripts/single_variant_prediction.py \
  --mutation "I31L" \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output test_output.csv \
  --device cpu

# Test with config file
python mcp_scripts/csv_stability_prediction.py \
  --input examples/data/subtilisin/small_test.csv \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output test_csv.csv \
  --config configs/csv_stability_config.json
```

## Troubleshooting

### Model Loading Issues
- Ensure `scripts/data/model/SPIRED-Stab.pth` exists
- Check internet connection for ESM-2 model downloads
- Verify PyTorch installation

### Memory Issues
- Use `--device cpu` if GPU memory insufficient
- Process smaller batches for large datasets
- Clear Python cache between runs

### SSL Certificate Issues (ESM model download)
```bash
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
```

### Path Issues
- Run scripts from project root directory
- Use absolute paths for input files
- Check that `scripts/` directory is accessible