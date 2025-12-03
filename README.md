# SPIRED-Fitness MCP

SPIRED-Fitness MCP server for protein modeling, extracted from the [official SPIRED-Fitness tutorials](https://github.com/Gonglab-THU/SPIRED-Fitness).

## Overview
This SPIRED-Fitness MCP server provides three protein analysis tools using ProtTrans models. Here we have 3 main scripts for protein analysis. But due to the large memory requirements, we only use Spired-Stab in the mcp service.

## Installation

```bash
# Create and activate virtual environment
mamba env create -p ./env python=3.11 pip -y
mamba activate ./env

pip install click einops pandas biopython tqdm loguru sniffio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install --ignore-installed fastmcp
```
Download the [model parameters](https://zenodo.org/records/10675405) and move it into the `data/model` folder.

## Local usage
### 1. Structure prediction
```shell
python scripts/run_spired.py --fasta examples/test.fasta --output examples/spired --repo repo/SPIRED-Fitness
```

### 2. Fitness prediction
```shell
python scripts/run_spired_fitness.py --fasta examples/test.fasta --output examples/spired --repo repo/SPIRED-Fitness
```

### 3. Stability prediction
```shell
python scripts/run_spired_stab.py --fasta examples/test.fasta --output examples/spired --repo repo/SPIRED-Fitness
```

## MCP usage
### Install MCP Server
```shell
fastmcp install claude-code mcp-servers/spired_fitness_mcp/src/spired_fitness_mcp.py --python mcp-servers/spired_fitness_mcp/env/bin/python
fastmcp install gemini-cli mcp-servers/spired_fitness_mcp/src/spired_fitness_mcp.py --python mcp-servers/spired_fitness_mcp/env/bin/python
```

### Call MCP service
Test wt path
/home/xux/Desktop/ProteinMCP/ProteinMCP/examples/case2.1_subtilisin/wt.fasta

Test fasta path
/home/xux/Desktop/ProteinMCP/ProteinMCP/mcp-servers/spired-fitness_mcp/examples/test.fasta
Test csv path
/home/xux/Desktop/ProteinMCP/ProteinMCP/examples/case2.1_subtilisin/data.csv

#### Stability prediction
```markdown
Can you predict the stabilities for variants in @examples/case2.1_subtilisin/data.csv and save it as @examples/case2.1_subtilisin/data.csv_spired_stab.csv using the spired_fitness_mcp.

Please convert the relative path to absolution path before calling the MCP servers. 
```