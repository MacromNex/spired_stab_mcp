"""
Model Context Protocol (MCP) for spired-fitness

SPIRED-Fitness is a comprehensive protein fitness landscape prediction and structure modeling toolkit that uses deep learning models to predict protein structures, fitness landscapes, and stability changes from mutations.

This MCP Server contains tools extracted from the following tutorial files:
1. run_spired_fitness
    - spired_fitness_predict_fitness_landscape: Predict protein fitness landscapes with single/double mutations and structure predictions
2. run_spired
    - spired_predict_protein_structures: Predict 3D protein structures from amino acid sequences using SPIRED-Fitness model
3. run_spired_stab
    - spired_predict_mutation_stability: Predicts stability changes (ddG, dTm) and generates 3D structures for protein mutations
"""

import sys
from pathlib import Path
from loguru import logger
from fastmcp import FastMCP

# Configure loguru for MCP server logging
# Remove default handler and add custom one
logger.remove()

# Add stderr handler with custom format for MCP server
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Also add file logging for debugging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(
    LOG_DIR / "spired_fitness_mcp.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)

logger.info("Starting SPIRED-Fitness MCP Server")

# Import statements (alphabetical order)
from tools.run_spired import run_spired_mcp
from tools.run_spired_fitness import run_spired_fitness_mcp
from tools.run_spired_stab import run_spired_stab_mcp

# Server definition and mounting
mcp = FastMCP(name="spired-fitness")
# mcp.mount(run_spired_mcp)
# mcp.mount(run_spired_fitness_mcp)
mcp.mount(run_spired_stab_mcp)

logger.info("SPIRED-Fitness MCP Server initialized with 1 tool")

if __name__ == "__main__":
    logger.info("Running SPIRED-Fitness MCP Server")
    mcp.run()