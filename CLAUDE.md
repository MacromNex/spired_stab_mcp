# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An MCP (Model Context Protocol) server for protein stability prediction using the SPIRED-Stab deep learning model. It predicts stability changes (ΔΔG and ΔTm) for protein variants relative to a wild-type sequence by combining ESM-2 embeddings with a specialized stability prediction head.

## Common Commands

```bash
# Activate environment
mamba activate ./env

# Run MCP server
python src/server.py

# Run dev server with inspector
fastmcp dev src/server.py

# Test imports and server creation
python test_server.py

# Quick smoke test with small dataset (5 variants, ~1 min on GPU)
python examples/use_case_1_csv_stability_prediction.py \
  --input examples/data/subtilisin/small_test.csv \
  --wt_fasta examples/data/subtilisin/wt.fasta \
  --device cuda:0

# Docker build
docker build -t spired-stab-mcp .
```

## Architecture

### Two-Layer Server Design

The MCP server has two entry points that both define a `FastMCP("spired_stab")` instance:

- **`src/server.py`** — The main MCP server entry point. Exposes 11 tools (sync predictions, async job submission, job management, utilities). Imports `predict_stability_direct` from `spired_stab_mcp.py` and delegates all actual computation to it.

- **`src/spired_stab_mcp.py`** — Core prediction engine. Contains model loading/caching logic, CSV-to-FASTA conversion, and the inference loop. Also defines its own `FastMCP` instance and `predict_stability` tool (used when run standalone). The `predict_stability_direct()` function is the primary programmatic interface.

**`server.py` is the canonical entry point** — it wraps `spired_stab_mcp.py` and adds the job management layer.

### Model Pipeline

1. Load SPIRED-Stab weights from `scripts/data/model/SPIRED-Stab.pth`
2. Load ESM-2 650M and ESM-2 3B via `torch.hub.load('facebookresearch/esm:main', ...)`
3. For each variant: compute ESM-2 embeddings → feed through SPIRED-Stab → output ddG/dTm
4. Models are cached in a global `_model_cache` dict to avoid reloading between calls

### Async Job System

`src/jobs/manager.py` provides background job execution via `subprocess.Popen` in daemon threads. Jobs are tracked as directories under `jobs/{job_id}/` with `metadata.json` and `job.log`. The submit tools in `server.py` launch the `examples/use_case_*.py` scripts as subprocesses.

### Path Dependencies

`spired_stab_mcp.py` does `os.chdir(script_dir)` temporarily during import to resolve relative imports in `scripts/src/model.py`. The model weights path is resolved relative to `scripts/data/model/`. Keep these paths intact when refactoring.

## Key Paths

- `scripts/src/` — Original SPIRED-Stab model implementation (model.py, module.py, esmfold_openfold/)
- `scripts/data/model/` — Pre-trained weights (~507MB SPIRED-Stab.pth, ~510MB SPIRED-Fitness.pth)
- `examples/data/subtilisin/` — Demo dataset (wt.fasta, variants.csv with 150 variants, small_test.csv with 5)
- `jobs/` — Runtime directory for async job tracking (created at runtime)

## Docker

The Dockerfile bakes in all model checkpoints to avoid runtime downloads:
- SPIRED-Stab/Fitness weights via COPY
- ESM-2 650M (~2.3GB) and ESM-2 3B (~11GB) pre-downloaded via torch.hub

The resulting image is large (~15-20GB). The GitHub Actions workflow at `.github/workflows/docker.yml` builds and pushes to GHCR on push to main.
