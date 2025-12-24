"""
Shared I/O functions for MCP scripts.

These are extracted and simplified from repo code to minimize dependencies.
"""

import pandas as pd
from pathlib import Path
from typing import Union, List
from Bio import SeqIO


def load_csv_input(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV input file and validate required columns.

    Args:
        file_path: Path to CSV file

    Returns:
        DataFrame with validated data

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If required columns are missing
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        if 'seq' not in df.columns:
            raise ValueError(f"CSV must contain 'seq' column. Available: {list(df.columns)}")
        return df
    except Exception as e:
        raise ValueError(f"Error reading CSV: {e}")


def load_fasta_input(file_path: Union[str, Path]) -> List:
    """
    Load FASTA input file and validate sequences.

    Args:
        file_path: Path to FASTA file

    Returns:
        List of SeqRecord objects

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no sequences found or invalid format
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    try:
        sequences = list(SeqIO.parse(file_path, "fasta"))
        if not sequences:
            raise ValueError("No sequences found in FASTA file")
        return sequences
    except Exception as e:
        raise ValueError(f"Error reading FASTA: {e}")


def load_wt_fasta(file_path: Union[str, Path]) -> str:
    """
    Load wild-type sequence from FASTA file.

    Args:
        file_path: Path to wild-type FASTA file

    Returns:
        Wild-type sequence string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If no sequences found or invalid format
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Wild-type FASTA not found: {file_path}")

    try:
        sequences = list(SeqIO.parse(file_path, "fasta"))
        if not sequences:
            raise ValueError("No sequences found in wild-type FASTA")
        return str(sequences[0].seq)
    except Exception as e:
        raise ValueError(f"Error reading wild-type FASTA: {e}")


def save_results(data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """
    Save results to CSV file.

    Args:
        data: DataFrame to save
        file_path: Output file path
        **kwargs: Additional arguments for pandas.to_csv()
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Default parameters
    csv_kwargs = {
        'index': False,
        'float_format': '%.4f'
    }
    csv_kwargs.update(kwargs)

    data.to_csv(file_path, **csv_kwargs)


def save_json(data: dict, file_path: Union[str, Path]) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        file_path: Output file path
    """
    import json

    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: Union[str, Path]) -> dict:
    """
    Load JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary from JSON file
    """
    import json

    with open(file_path) as f:
        return json.load(f)