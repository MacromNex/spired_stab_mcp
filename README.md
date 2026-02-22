# SPIRED-Stab MCP

**Protein stability prediction using SPIRED-Stab deep learning model via Docker**

An MCP (Model Context Protocol) server for protein stability prediction with 6 core tools:
- Predict stability changes (ΔΔG and ΔTm) for protein variants from CSV or FASTA
- Analyze single variant mutations against a wild-type reference
- Submit large batch stability prediction jobs with async tracking
- Submit systematic batch mutation analysis
- Monitor and retrieve prediction results
- List available example datasets

## Quick Start with Docker

### Approach 1: Pull Pre-built Image from GitHub

The fastest way to get started. A pre-built Docker image is automatically published to GitHub Container Registry on every release.

```bash
# Pull the latest image
docker pull ghcr.io/macromnex/spired_stab_mcp:latest

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add spired_stab -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` ghcr.io/macromnex/spired_stab_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support (`nvidia-docker` or Docker with NVIDIA runtime)
- Claude Code installed

That's it! The SPIRED-Stab MCP server is now available in Claude Code.

---

### Approach 2: Build Docker Image Locally

Build the image yourself and install it into Claude Code. Useful for customization or offline environments.

```bash
# Clone the repository
git clone https://github.com/MacromNex/spired_stab_mcp.git
cd spired_stab_mcp

# Build the Docker image
docker build -t spired_stab_mcp:latest .

# Register with Claude Code (runs as current user to avoid permission issues)
claude mcp add spired_stab -- docker run -i --rm --user `id -u`:`id -g` --gpus all --ipc=host -v `pwd`:`pwd` spired_stab_mcp:latest
```

**Note:** Run from your project directory. `` `pwd` `` expands to the current working directory.

**Requirements:**
- Docker with GPU support
- Claude Code installed
- Git (to clone the repository)

**About the Docker Flags:**
- `-i` — Interactive mode for Claude Code
- `--rm` — Automatically remove container after exit
- `` --user `id -u`:`id -g` `` — Runs the container as your current user, so output files are owned by you (not root)
- `--gpus all` — Grants access to all available GPUs
- `--ipc=host` — Uses host IPC namespace for PyTorch shared memory
- `-v` — Mounts your project directory so the container can access your data

---

## Verify Installation

After adding the MCP server, you can verify it's working:

```bash
# List registered MCP servers
claude mcp list

# You should see 'spired_stab' in the output
```

In Claude Code, you can now use all 6 SPIRED-Stab tools:
- `predict_stability`
- `analyze_single_variant`
- `submit_stability_prediction`
- `submit_batch_mutation_analysis`
- `get_job_status`
- `get_job_result`

---

## Next Steps

- **Detailed documentation**: See [detail.md](detail.md) for comprehensive guides on:
  - Available MCP tools and parameters
  - Local Python environment setup (alternative to Docker)
  - Example workflows and use cases
  - Data format requirements
  - Troubleshooting

---

## Usage Examples

Once registered, you can use the SPIRED-Stab tools directly in Claude Code. Here are some common workflows:

### Example 1: Quick Stability Prediction

```
I have protein variants at /path/to/variants.csv with a 'seq' column and wild-type sequence at /path/to/wt.fasta. Can you use predict_stability to predict stability changes for all variants and save results to /path/to/results.csv?
```

### Example 2: Single Variant Analysis

```
I want to analyze the effect of mutation I31L on my protein. The wild-type sequence is at /path/to/wt.fasta. Can you use analyze_single_variant to predict the stability change for this mutation?
```

### Example 3: Systematic Mutation Scanning

```
I want to explore all possible mutations at positions 31, 67, and 124 of my protein at /path/to/wt.fasta. Can you submit a batch mutation analysis job using submit_batch_mutation_analysis with max 100 variants, and monitor progress until completion?
```

---

## Troubleshooting

**Docker not found?**
```bash
docker --version  # Install Docker if missing
```

**GPU not accessible?**
- Ensure NVIDIA Docker runtime is installed
- Check with: `docker run --gpus all ubuntu nvidia-smi`

**Claude Code not found?**
```bash
# Install Claude Code
npm install -g @anthropic-ai/claude-code
```

**Out of GPU memory?**
- SPIRED-Stab requires 4-6 GB VRAM
- Use `device: "cpu"` for CPU inference (significantly slower)
- For very large datasets, use `submit_stability_prediction` for background processing

---

## License

Research use — Based on [SPIRED-Stab](https://github.com/YoGo-1030/SPIRED-Stab) by YoGo-1030.
