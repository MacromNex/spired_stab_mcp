"""
General utilities for MCP scripts.

Common functions for mutation handling, stability classification, and other utilities.
"""

import re
from typing import List, Tuple, Union


def apply_mutations(wt_sequence: str, mutations_str: str) -> str:
    """
    Apply mutations to wild-type sequence.

    Args:
        wt_sequence: Wild-type protein sequence
        mutations_str: Mutations in format "I31L,S3T" or "I31L/S3T"

    Returns:
        Modified sequence with mutations applied

    Raises:
        ValueError: If mutation format is invalid
    """
    sequence = list(wt_sequence)

    # Parse mutations (handle both comma and slash separators)
    mutations = mutations_str.replace('/', ',').split(',')

    for mutation in mutations:
        mutation = mutation.strip()
        if not mutation:
            continue

        # Parse mutation format like "I31L" (from I to L at position 31)
        if len(mutation) < 3:
            print(f"Warning: Invalid mutation format: {mutation}")
            continue

        # Use regex to parse mutations more robustly
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation.upper())
        if not match:
            print(f"Warning: Invalid mutation format: {mutation}")
            continue

        wt_aa, pos_str, mut_aa = match.groups()

        try:
            position = int(pos_str) - 1  # Convert to 0-based indexing
        except ValueError:
            print(f"Warning: Invalid position in mutation: {mutation}")
            continue

        if position < 0 or position >= len(sequence):
            print(f"Warning: Position {position + 1} is out of range for sequence length {len(sequence)}")
            continue

        if sequence[position] != wt_aa:
            print(f"Warning: Expected {wt_aa} at position {position + 1}, but found {sequence[position]}")

        sequence[position] = mut_aa
        print(f"Applied mutation: {wt_aa}{position + 1}{mut_aa}")

    return ''.join(sequence)


def parse_mutation(mutation_str: str) -> Tuple[str, int, str]:
    """
    Parse a single mutation string.

    Args:
        mutation_str: Mutation string like "I31L"

    Returns:
        Tuple of (wt_aa, position, mut_aa)

    Raises:
        ValueError: If mutation format is invalid
    """
    mutation_str = mutation_str.strip()

    # Use regex to parse mutations
    match = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation_str.upper())
    if not match:
        raise ValueError(f"Invalid mutation format: {mutation_str}")

    wt_aa, pos_str, mut_aa = match.groups()
    position = int(pos_str)

    return wt_aa, position, mut_aa


def classify_stability(ddG_value: float, thresholds: Tuple[float, float] = (-1.0, 1.0)) -> str:
    """
    Classify stability based on ΔΔG value.

    Args:
        ddG_value: The ΔΔG value
        thresholds: Tuple of (stabilizing_threshold, destabilizing_threshold)

    Returns:
        String classification
    """
    stable_thresh, unstable_thresh = thresholds

    if ddG_value < stable_thresh:
        return "Highly Stabilizing"
    elif ddG_value < 0:
        return "Stabilizing"
    elif ddG_value == 0:
        return "Neutral"
    elif ddG_value < unstable_thresh:
        return "Destabilizing"
    else:
        return "Highly Destabilizing"


def validate_sequence(sequence: str, allow_gaps: bool = False) -> bool:
    """
    Validate protein sequence.

    Args:
        sequence: Protein sequence to validate
        allow_gaps: Whether to allow gap characters

    Returns:
        True if sequence is valid
    """
    standard_aa = set("ACDEFGHIKLMNPQRSTVWY")
    gap_chars = set("-X")

    allowed_chars = standard_aa
    if allow_gaps:
        allowed_chars = allowed_chars.union(gap_chars)

    sequence_chars = set(sequence.upper())
    return sequence_chars.issubset(allowed_chars)


def count_mutations(wt_sequence: str, mut_sequence: str) -> int:
    """
    Count the number of differences between two sequences.

    Args:
        wt_sequence: Wild-type sequence
        mut_sequence: Mutant sequence

    Returns:
        Number of amino acid differences
    """
    if len(wt_sequence) != len(mut_sequence):
        print(f"Warning: Sequence lengths differ ({len(wt_sequence)} vs {len(mut_sequence)})")
        # Count differences up to the shorter length
        min_len = min(len(wt_sequence), len(mut_sequence))
        differences = sum(1 for i in range(min_len) if wt_sequence[i] != mut_sequence[i])
        # Add length difference
        differences += abs(len(wt_sequence) - len(mut_sequence))
        return differences

    return sum(1 for a, b in zip(wt_sequence, mut_sequence) if a != b)


def generate_mutation_name(wt_sequence: str, mut_sequence: str, max_mutations: int = 5) -> str:
    """
    Generate a mutation name from sequence differences.

    Args:
        wt_sequence: Wild-type sequence
        mut_sequence: Mutant sequence
        max_mutations: Maximum number of mutations to include in name

    Returns:
        Mutation name string like "I31L,S3T"
    """
    if len(wt_sequence) != len(mut_sequence):
        return f"variant_{len(mut_sequence)}aa"

    mutations = []
    for i, (wt, mut) in enumerate(zip(wt_sequence, mut_sequence)):
        if wt != mut:
            mutations.append(f"{wt}{i+1}{mut}")

    if not mutations:
        return "WT"

    if len(mutations) > max_mutations:
        return f"{len(mutations)}_mutations"

    return ",".join(mutations)


def format_number(value: float, precision: int = 4) -> str:
    """
    Format a number with specified precision.

    Args:
        value: Number to format
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{value:.{precision}f}"


def calculate_statistics(values: List[float]) -> dict:
    """
    Calculate basic statistics for a list of values.

    Args:
        values: List of numerical values

    Returns:
        Dictionary with statistics
    """
    import numpy as np

    if not values:
        return {"count": 0}

    values = np.array(values)

    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75))
    }