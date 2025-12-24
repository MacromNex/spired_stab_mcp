"""
Shared library for MCP SPIRED-Stab scripts.

This package contains common utilities, I/O functions, and model loading
functions that are shared across all MCP scripts.
"""

__version__ = "1.0.0"
__author__ = "SPIRED-Stab MCP"

from .io import load_csv_input, load_fasta_input, load_wt_fasta, save_results
from .models import get_spired_models, ModelCache
from .utils import apply_mutations, classify_stability

__all__ = [
    "load_csv_input",
    "load_fasta_input",
    "load_wt_fasta",
    "save_results",
    "get_spired_models",
    "ModelCache",
    "apply_mutations",
    "classify_stability"
]