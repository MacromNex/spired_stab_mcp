# SPIRED-Stab MCP

> Protein stability prediction using the SPIRED-Stab deep learning model through MCP integration

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Local Usage (Scripts)](#local-usage-scripts)
- [MCP Server Installation](#mcp-server-installation)
- [Using with Claude Code](#using-with-claude-code)
- [Using with Gemini CLI](#using-with-gemini-cli)
- [Available Tools](#available-tools)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)

## Overview

The SPIRED-Stab MCP provides protein stability prediction capabilities using a deep learning model that combines ESM-2 embeddings with specialized training for predicting stability changes (ΔΔG and ΔTm) upon protein mutation. This MCP integration enables seamless protein engineering workflows through both synchronous and asynchronous APIs.

### Features
- Protein stability prediction from CSV and FASTA inputs
- Single variant analysis with mutation specification
- Batch mutation analysis and systematic scanning
- Comparative stability analysis with rankings and statistics
- GPU-accelerated inference with automatic CPU fallback
- Asynchronous job submission for large datasets
- Job management with progress tracking

### Directory Structure
```
./
├── README.md               # This file
├── env/                    # Conda environment
├── src/
│   ├── server.py           # MCP server
│   ├── spired_stab_mcp.py  # Core prediction functions
│   └── jobs/               # Job management system
├── scripts/
│   ├── run_SPIRED-Stab.py  # Original SPIRED-Stab script
│   └── src/                # Model implementation
├── examples/
│   ├── data/               # Demo data
│   ├── use_case_*.py       # Example scripts
│   └── *.csv, *.fasta      # Sample data files
├── reports/                # Documentation from setup steps
└── repo/                   # Original SPIRED-Stab repository
```

---

## Installation

### Prerequisites
- Conda or Mamba (mamba recommended for faster installation)
- Python 3.10+
- CUDA-compatible GPU (optional but recommended for performance)

### Create Environment

Based on the environment setup from `reports/step3_environment.md`:

```bash
# Navigate to the MCP directory
cd /home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/spired_stab_mcp

# Create conda environment (use mamba if available)
mamba create -p ./env python=3.11 -y
# or: conda create -p ./env python=3.11 -y

# Activate environment
mamba activate ./env
# or: conda activate ./env

# Install Dependencies
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 torchaudio==2.7.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html
pip install pandas==2.3.3 numpy==2.3.5 biopython==1.86
pip install tqdm click loguru

# Install MCP dependencies
pip install fastmcp --ignore-installed
```

**Note**: The environment contains pre-trained model weights and will download ESM-2 models (~2-11GB) on first use.

---

## Local Usage (Scripts)

You can use the scripts directly without MCP for local processing.

### Available Scripts

| Script | Description | Example |
|--------|-------------|---------|
| `examples/use_case_1_csv_stability_prediction.py` | Predict stability from CSV file | See below |
| `examples/use_case_2_fasta_stability_prediction.py` | Predict stability from FASTA file | See below |
| `examples/use_case_3_single_variant_prediction.py` | Single variant analysis | See below |
| `examples/use_case_4_batch_mutation_analysis.py` | Systematic mutation scanning | See below |
| `examples/use_case_5_comparative_stability_analysis.py` | Comparative analysis with rankings | See below |

### Script Examples

#### CSV Stability Prediction

```bash
# Activate environment
mamba activate ./env

# Run CSV prediction
python examples/use_case_1_csv_stability_prediction.py \
  --input examples/data/subtilisin/variants.csv \
  --output results/csv_predictions.csv \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --device cuda:0
```

**Parameters:**
- `--input, -i`: CSV file with 'seq' column containing protein sequences (required)
- `--output, -o`: Output CSV file path (default: auto-generated)
- `--wt_fasta, -w`: Wild-type protein sequence FASTA file (optional, auto-detected)
- `--device, -d`: Computing device (cuda:0, cpu) (default: cuda:0)

#### FASTA Stability Prediction

```bash
python examples/use_case_2_fasta_stability_prediction.py \
  --input examples/data/subtilisin/sequences.fasta \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/fasta_predictions.csv
```

#### Single Variant Analysis

```bash
python examples/use_case_3_single_variant_prediction.py \
  --mutation "I31L" \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/single_variant.csv

# Multiple mutations
python examples/use_case_3_single_variant_prediction.py \
  --mutation "I31L,S3T" \
  --wt_fasta examples/data/subtilisin/wt.fasta
```

#### Batch Mutation Analysis

```bash
python examples/use_case_4_batch_mutation_analysis.py \
  --positions "31,67,124" \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --max_variants 100 \
  --output results/batch_analysis.csv
```

#### Comparative Stability Analysis

```bash
python examples/use_case_5_comparative_stability_analysis.py \
  --input examples/data/subtilisin/variants.csv \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --output results/comparative_analysis.csv
```

---

## MCP Server Installation

### Option 1: Using fastmcp (Recommended)

```bash
# Activate environment
mamba activate ./env

# Install MCP server for Claude Code
fastmcp install src/server.py --name spired_stab
```

### Option 2: Manual Installation for Claude Code

```bash
# Add MCP server to Claude Code
claude mcp add spired_stab -- $(pwd)/env/bin/python $(pwd)/src/server.py

# Verify installation
claude mcp list
```

### Option 3: Configure in settings.json

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "spired_stab": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/spired_stab_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/spired_stab_mcp/src/server.py"]
    }
  }
}
```

---

## Using with Claude Code

After installing the MCP server, you can use it directly in Claude Code.

### Quick Start

```bash
# Start Claude Code
claude
```

### Example Prompts

#### Tool Discovery
```
What tools are available from spired_stab?
```

#### Basic Usage
```
Use predict_stability with variant_file @examples/data/subtilisin/variants.csv
```

#### With Configuration
```
Run predict_stability on @examples/data/subtilisin/small_test.csv using wild-type sequence @examples/data/subtilisin/wt.fasta
```

#### Long-Running Tasks (Submit API)
```
Submit stability prediction for @examples/data/subtilisin/variants.csv with job name "test_batch"
Then check the job status
```

#### Single Variant Analysis
```
Analyze single variant with sequence "AQTVPYGVSQI..." and wild-type @examples/data/subtilisin/wt.fasta
```

#### Batch Processing
```
Submit batch mutation analysis for positions "31,67,124" using wild-type @examples/data/subtilisin/wt.fasta
```

### Using @ References

In Claude Code, use `@` to reference files and directories:

| Reference | Description |
|-----------|-------------|
| `@examples/data/subtilisin/variants.csv` | Reference the main variant dataset (150 variants) |
| `@examples/data/subtilisin/wt.fasta` | Reference wild-type sequence |
| `@examples/data/subtilisin/small_test.csv` | Reference small test dataset (5 variants) |
| `@examples/data/subtilisin/sequences.fasta` | Reference FASTA sequences (116 sequences) |
| `@examples/test_small.csv` | Reference small sample dataset in project root |
| `@examples/test_small.fasta` | Reference small sample FASTA dataset |

---

## Using with Gemini CLI

### Configuration

Add to `~/.gemini/settings.json`:

```json
{
  "mcpServers": {
    "spired_stab": {
      "command": "/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/spired_stab_mcp/env/bin/python",
      "args": ["/home/xux/Desktop/ProteinMCP/ProteinMCP/tool-mcps/spired_stab_mcp/src/server.py"]
    }
  }
}
```

### Example Prompts

```bash
# Start Gemini CLI
gemini

# Example prompts (same as Claude Code)
> What tools are available?
> Use predict_stability with file examples/data/subtilisin/variants.csv
> Submit batch mutation analysis for positions "31,67,124"
```

---

## Available Tools

### Quick Operations (Sync API)

These tools return results immediately (< 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `predict_stability` | Protein stability prediction (sync) | `variant_file`, `wt_fasta_file`, `output_file`, `device` |
| `analyze_single_variant` | Single variant analysis | `variant_sequence`, `wt_fasta_file`, `variant_id`, `device` |
| `validate_input_files` | File validation and size estimation | `variant_file`, `wt_fasta_file` |
| `list_example_data` | List available example datasets | None |

### Long-Running Tasks (Submit API)

These tools return a job_id for tracking (> 10 minutes):

| Tool | Description | Parameters |
|------|-------------|------------|
| `submit_stability_prediction` | Large batch stability prediction | `variant_file`, `wt_fasta_file`, `output_dir`, `device`, `job_name` |
| `submit_batch_mutation_analysis` | Systematic mutation analysis | `wt_fasta_file`, `positions` OR `mutations`, `max_variants`, `device`, `job_name` |
| `submit_comparative_stability_analysis` | Compare multiple variant sets | `variant_files`, `wt_fasta_file`, `analysis_name`, `device`, `job_name` |

### Job Management Tools

| Tool | Description |
|------|-------------|
| `get_job_status` | Check job progress and status |
| `get_job_result` | Get results when completed |
| `get_job_log` | View execution logs |
| `cancel_job` | Cancel running job |
| `list_jobs` | List all jobs with optional status filter |

---

## Examples

### Example 1: Quick Stability Prediction (Small Dataset)

**Goal:** Predict stability changes for a small dataset of protein variants

**Using Script:**
```bash
python examples/use_case_1_csv_stability_prediction.py \
  --input examples/data/subtilisin/small_test.csv \
  --output results/quick_predictions.csv
```

**Using MCP (in Claude Code):**
```
Use predict_stability to process @examples/data/subtilisin/small_test.csv and save results to results/quick_predictions.csv
```

**Expected Output:**
- CSV file with ddG and dTm predictions for each variant
- Runtime: ~30 seconds to 2 minutes

### Example 2: Large Batch Processing (Background Job)

**Goal:** Process a large dataset using background job submission

**Using Script:**
```bash
python examples/use_case_1_csv_stability_prediction.py \
  --input examples/data/subtilisin/variants.csv \
  --output results/large_batch.csv \
  --device cuda:0
```

**Using MCP (in Claude Code):**
```
Submit stability prediction for @examples/data/subtilisin/variants.csv with job name "large_batch_prediction"
```

**Check Status:**
```
Get job status for job_id "abc123"
```

### Example 3: Single Variant Engineering

**Goal:** Analyze specific mutations for protein engineering

**Using Script:**
```bash
python examples/use_case_3_single_variant_prediction.py \
  --mutation "I31L" \
  --wt_fasta examples/data/subtilisin/wt.fasta
```

**Using MCP (in Claude Code):**
```
Analyze single variant with mutation "I31L" using wild-type @examples/data/subtilisin/wt.fasta
```

### Example 4: Systematic Mutation Scanning

**Goal:** Explore all possible mutations at specific positions

**Using Script:**
```bash
python examples/use_case_4_batch_mutation_analysis.py \
  --positions "31,67" \
  --max_variants 40 \
  --wt_fasta examples/data/subtilisin/wt.fasta
```

**Using MCP (in Claude Code):**
```
Submit batch mutation analysis for positions "31,67" with max 40 variants using @examples/data/subtilisin/wt.fasta
```

### Example 5: Comparative Analysis

**Goal:** Compare multiple datasets with statistical analysis

**Using Script:**
```bash
python examples/use_case_5_comparative_stability_analysis.py \
  --input examples/data/subtilisin/variants.csv \
  --wt_fasta examples/data/subtilisin/wt.fasta
```

**Using MCP (in Claude Code):**
```
Run comparative stability analysis on @examples/data/subtilisin/variants.csv with rankings and statistics
```

---

## Demo Data

The `examples/data/` directory contains sample data for testing:

| File | Description | Use With | Size |
|------|-------------|----------|------|
| `examples/data/subtilisin/variants.csv` | 150 subtilisin variants with experimental fitness data | `predict_stability`, comparative analysis | 150 sequences |
| `examples/data/subtilisin/sequences.fasta` | Protein variants in FASTA format | FASTA prediction tools | 116 sequences |
| `examples/data/subtilisin/wt.fasta` | Wild-type subtilisin sequence | All prediction tools | Reference sequence |
| `examples/data/subtilisin/small_test.csv` | Small test dataset for quick testing | Quick testing | 5 sequences |
| `examples/test_small.csv` | Small sample dataset in project root | Quick testing | 5 sequences |
| `examples/test_small.fasta` | Small sample FASTA dataset | FASTA testing | Few sequences |

---

## Configuration Files

The environment setup uses specific package versions for reproducibility:

### Key Dependencies
- **PyTorch**: 2.7.1+cu118 (CUDA 11.8 support)
- **BioPython**: 1.86 (FASTA/sequence handling)
- **Pandas**: 2.3.3 (data manipulation)
- **NumPy**: 2.3.5 (numerical operations)
- **FastMCP**: MCP server framework

### Model Configuration
- **SPIRED-Stab**: Located in `scripts/data/model/SPIRED-Stab.pth` (507MB)
- **ESM-2 models**: Auto-downloaded via torch.hub (2-11GB depending on model)

---

## Troubleshooting

### Environment Issues

**Problem:** Environment not found
```bash
# Recreate environment
mamba create -p ./env python=3.11 -y
mamba activate ./env
# Re-run installation commands above
```

**Problem:** Import errors
```bash
# Verify installation
python -c "import torch, pandas, numpy, Bio; print('All imports successful')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### MCP Issues

**Problem:** Server not found in Claude Code
```bash
# Check MCP registration
claude mcp list

# Re-add if needed
claude mcp remove spired_stab
fastmcp install src/server.py --name spired_stab
```

**Problem:** Tools not working
```bash
# Test server directly
python -c "
from src.server import mcp
print(list(mcp.list_tools().keys()))
"
```

### Model Issues

**Problem:** CUDA out of memory
```bash
# Use CPU instead
python examples/use_case_1_csv_stability_prediction.py --device cpu --input examples/data/subtilisin/small_test.csv
```

**Problem:** Model download fails
```bash
# Check internet connection and retry
# Models are downloaded to ~/.cache/torch/hub/
ls -la ~/.cache/torch/hub/
```

### Job Issues

**Problem:** Job stuck in pending
```bash
# Check job directory
ls -la src/jobs/

# View job log
python -c "from src.jobs.manager import job_manager; print(job_manager.get_job_log('JOB_ID'))"
```

**Problem:** Job failed
```
Use get_job_log with job_id "JOB_ID" and tail 100 to see error details
```

### Performance Issues

**Problem:** Very slow predictions
- **GPU**: Ensure CUDA is available and drivers are updated
- **CPU**: Consider reducing batch size or using GPU
- **Memory**: Close other applications using GPU memory

**Problem:** Model loading takes too long
- **First run**: Normal behavior (downloads models)
- **Subsequent runs**: Should be faster with cached models

---

## Development

### Running Tests

```bash
# Activate environment
mamba activate ./env

# Test basic functionality
python -c "from src.spired_stab_mcp import predict_stability_direct; print('Import successful')"

# Test with small data
python examples/use_case_1_csv_stability_prediction.py \
  --input examples/data/subtilisin/small_test.csv \
  --output test_output.csv
```

### Starting Dev Server

```bash
# Run MCP server in development mode
mamba activate ./env
fastmcp dev src/server.py
```

### Performance Characteristics

#### Typical Performance (GPU):
- **Single sequence**: 2-5 seconds
- **Batch of 10 sequences**: 30-60 seconds
- **Batch of 100 sequences**: 5-10 minutes
- **Large dataset (150+ sequences)**: 15-30 minutes

#### Memory Requirements:
- **GPU Memory**: 4-6GB for typical proteins
- **System RAM**: 8GB+ recommended
- **Disk Space**: ~4GB for environment and models

---

## License

This project is based on the SPIRED-Stab model and includes additional MCP integration code.

## Credits

Based on [SPIRED-Stab: Deep learning-based prediction of protein stability changes upon single-point mutation](https://github.com/YoGo-1030/SPIRED-Stab)

- Original SPIRED-Stab model by YoGo-1030
- MCP integration and workflow automation
- Example datasets from subtilisin engineering studies