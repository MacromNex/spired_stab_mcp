"""
SPIRED-Stab MCP Server

Provides both synchronous and asynchronous APIs for protein stability prediction.
Supports CSV and FASTA inputs, single variants, and batch processing.
"""

from fastmcp import FastMCP
from pathlib import Path
from typing import Optional, List
import sys
import pandas as pd
import os

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MCP_ROOT = SCRIPT_DIR.parent
EXAMPLES_DIR = MCP_ROOT / "examples"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(MCP_ROOT))

from jobs.manager import job_manager
from loguru import logger

# Import the existing SPIRED-Stab functionality
from spired_stab_mcp import predict_stability_direct

# Create MCP server
mcp = FastMCP("spired_stab")

# ==============================================================================
# Job Management Tools (for async operations)
# ==============================================================================

@mcp.tool()
def get_job_status(job_id: str) -> dict:
    """
    Get the status of a submitted job.

    Args:
        job_id: The job ID returned from a submit_* function

    Returns:
        Dictionary with job status, timestamps, and any errors
    """
    return job_manager.get_job_status(job_id)


@mcp.tool()
def get_job_result(job_id: str) -> dict:
    """
    Get the results of a completed job.

    Args:
        job_id: The job ID of a completed job

    Returns:
        Dictionary with the job results or path to output file
    """
    return job_manager.get_job_result(job_id)


@mcp.tool()
def get_job_log(job_id: str, tail: int = 50) -> dict:
    """
    Get log output from a running or completed job.

    Args:
        job_id: The job ID to get logs for
        tail: Number of lines from end (default: 50, use 0 for all)

    Returns:
        Dictionary with log lines and total line count
    """
    return job_manager.get_job_log(job_id, tail)


@mcp.tool()
def cancel_job(job_id: str) -> dict:
    """
    Cancel a running job.

    Args:
        job_id: The job ID to cancel

    Returns:
        Success or error message
    """
    return job_manager.cancel_job(job_id)


@mcp.tool()
def list_jobs(status: Optional[str] = None) -> dict:
    """
    List all submitted jobs.

    Args:
        status: Filter by status (pending, running, completed, failed, cancelled)

    Returns:
        List of jobs with their status
    """
    return job_manager.list_jobs(status)


# ==============================================================================
# Synchronous Tools (for fast operations < 10 min)
# ==============================================================================

@mcp.tool()
def predict_stability(
    variant_file: str,
    wt_fasta_file: Optional[str] = None,
    output_file: Optional[str] = None,
    device: str = 'cuda:0'
) -> dict:
    """
    Predict protein stability using SPIRED-Stab (synchronous API for small batches).

    Use this for quick predictions with fewer than 50 variants. For larger batches,
    use submit_stability_prediction() for background processing.

    This tool predicts the stability changes (ddG and dTm) for protein variants
    compared to a wild-type sequence. It supports both CSV and FASTA input formats.

    Args:
        variant_file: Path to the variant sequences file (CSV or FASTA format).
                     For CSV: must contain a 'seq' column with protein sequences.
                     For FASTA: standard FASTA format with sequence IDs and sequences.
        wt_fasta_file: Path to the wild-type sequence FASTA file.
                      If not provided, will look for 'wt.fasta' in the same directory as variant_file.
        output_file: Path to save the prediction results (CSV format).
                    If not provided, will save as '{variant_file}_spired_stab_pred.csv'
        device: Device to run inference on (default: 'cuda:0').
               Use 'cpu' if no GPU is available, or 'cuda:1' for a different GPU.

    Returns:
        Dictionary with status and results summary.
        The output CSV contains: id, seq, ddG (stability change), dTm (melting temperature change)

    Example usage:
        - CSV input: predict_stability('/path/to/variants.csv', '/path/to/wt.fasta')
        - FASTA input: predict_stability('/path/to/variants.fasta', '/path/to/wt.fasta')
        - Default wt: predict_stability('/path/to/variants.fasta')  # uses wt.fasta in same directory
    """
    try:
        # Check input file size for recommendation
        if variant_file.endswith('.csv'):
            try:
                df = pd.read_csv(variant_file)
                num_variants = len(df)
            except:
                num_variants = 0
        else:
            # Rough estimate for FASTA
            with open(variant_file, 'r') as f:
                content = f.read()
                num_variants = content.count('>')

        if num_variants > 50:
            logger.warning(f"Large batch detected ({num_variants} variants). Consider using submit_stability_prediction() for better performance.")

        # Run the prediction
        result = predict_stability_direct(
            variant_file=variant_file,
            wt_fasta_file=wt_fasta_file,
            output_file=output_file,
            device=device
        )

        # Parse the result to extract key information
        lines = result.strip().split('\n')
        output_path = None
        for line in lines:
            if line.startswith('Results saved to:'):
                output_path = line.split('Results saved to:')[1].strip()
                break

        return {
            "status": "success",
            "message": result,
            "output_file": output_path,
            "num_variants": num_variants
        }

    except FileNotFoundError as e:
        return {"status": "error", "error": f"File not found: {e}"}
    except ValueError as e:
        return {"status": "error", "error": f"Invalid input: {e}"}
    except Exception as e:
        logger.error(f"predict_stability failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def analyze_single_variant(
    variant_sequence: str,
    wt_fasta_file: str,
    variant_id: Optional[str] = None,
    device: str = 'cuda:0'
) -> dict:
    """
    Analyze a single protein variant for stability changes (synchronous API).

    This is a convenience function for analyzing just one variant sequence
    without needing to create input files.

    Args:
        variant_sequence: The variant protein sequence (single letter amino acid codes)
        wt_fasta_file: Path to the wild-type sequence FASTA file
        variant_id: Optional ID for the variant (default: 'variant_1')
        device: Device to run inference on (default: 'cuda:0')

    Returns:
        Dictionary with stability prediction results
    """
    import tempfile
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio import SeqIO

    try:
        # Create temporary FASTA file for the variant
        variant_id = variant_id or "variant_1"
        temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)

        variant_record = SeqRecord(Seq(variant_sequence), id=variant_id, description="")
        SeqIO.write([variant_record], temp_fasta, "fasta")
        temp_fasta.close()

        # Create temporary output file
        temp_output = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        temp_output.close()

        # Run prediction
        result = predict_stability_direct(
            variant_file=temp_fasta.name,
            wt_fasta_file=wt_fasta_file,
            output_file=temp_output.name,
            device=device
        )

        # Read the results
        results_df = pd.read_csv(temp_output.name)

        # Clean up temporary files
        os.unlink(temp_fasta.name)
        os.unlink(temp_output.name)

        if len(results_df) > 0:
            row = results_df.iloc[0]
            return {
                "status": "success",
                "variant_id": row['id'],
                "sequence": row['seq'],
                "ddG": float(row['ddG']),
                "dTm": float(row['dTm']),
                "stability_change": "stabilizing" if row['ddG'] < 0 else "destabilizing",
                "message": f"ΔΔG = {row['ddG']:.4f}, ΔTm = {row['dTm']:.4f}"
            }
        else:
            return {"status": "error", "error": "No results generated"}

    except Exception as e:
        logger.error(f"analyze_single_variant failed: {e}")
        return {"status": "error", "error": str(e)}


# ==============================================================================
# Submit Tools (for long-running operations > 10 min)
# ==============================================================================

@mcp.tool()
def submit_stability_prediction(
    variant_file: str,
    wt_fasta_file: Optional[str] = None,
    output_dir: Optional[str] = None,
    device: str = 'cuda:0',
    job_name: Optional[str] = None
) -> dict:
    """
    Submit protein stability prediction for background processing (large batches).

    This operation is suitable for large batches (>50 variants) that may take more
    than 10 minutes. Returns a job_id for tracking progress.

    Args:
        variant_file: Path to the variant sequences file (CSV or FASTA format)
        wt_fasta_file: Path to the wild-type sequence FASTA file
        output_dir: Directory to save outputs (default: current directory)
        device: Device to run inference on (default: 'cuda:0')
        job_name: Optional name for the job (for easier tracking)

    Returns:
        Dictionary with job_id. Use:
        - get_job_status(job_id) to check progress
        - get_job_result(job_id) to get results
        - get_job_log(job_id) to see execution logs
    """
    script_path = str(EXAMPLES_DIR / "use_case_1_csv_stability_prediction.py")
    if variant_file.endswith('.fasta'):
        script_path = str(EXAMPLES_DIR / "use_case_2_fasta_stability_prediction.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "input": variant_file,
            "wt_fasta": wt_fasta_file,
            "device": device
        },
        job_name=job_name or f"stability_prediction_{Path(variant_file).stem}"
    )


@mcp.tool()
def submit_batch_mutation_analysis(
    wt_fasta_file: str,
    positions: Optional[str] = None,
    mutations: Optional[str] = None,
    max_variants: int = 100,
    device: str = 'cuda:0',
    job_name: Optional[str] = None
) -> dict:
    """
    Submit batch mutation analysis for background processing.

    This generates multiple variants with specific mutations and analyzes their
    stability changes in batch. Choose either positions (for all possible mutations
    at those positions) or specific mutations.

    Args:
        wt_fasta_file: Path to wild-type sequence FASTA file
        positions: Positions to mutate (comma-separated, 1-based), e.g., "31,67,124"
        mutations: Specific mutations (comma-separated), e.g., "I31L,H67Y,M124L"
        max_variants: Maximum number of variants to generate (default: 100)
        device: Device to run inference on (default: 'cuda:0')
        job_name: Optional name for the batch job

    Returns:
        Dictionary with job_id for tracking the batch job
    """
    script_path = str(EXAMPLES_DIR / "use_case_4_batch_mutation_analysis.py")

    # Validate input
    if positions is None and mutations is None:
        return {"status": "error", "error": "Must specify either positions or mutations"}

    args = {
        "wt_fasta": wt_fasta_file,
        "device": device,
        "max_variants": max_variants
    }

    if positions:
        args["positions"] = positions
    elif mutations:
        args["mutations"] = mutations

    return job_manager.submit_job(
        script_path=script_path,
        args=args,
        job_name=job_name or f"batch_mutation_analysis"
    )


@mcp.tool()
def submit_comparative_stability_analysis(
    variant_files: List[str],
    wt_fasta_file: str,
    analysis_name: str,
    device: str = 'cuda:0',
    job_name: Optional[str] = None
) -> dict:
    """
    Submit comparative analysis of multiple variant sets for background processing.

    This analyzes and compares stability predictions across multiple variant sets,
    useful for comparing different experimental conditions or mutation strategies.

    Args:
        variant_files: List of paths to variant files to compare
        wt_fasta_file: Path to wild-type sequence FASTA file
        analysis_name: Name for this comparative analysis
        device: Device to run inference on (default: 'cuda:0')
        job_name: Optional name for the job

    Returns:
        Dictionary with job_id for tracking the comparative analysis
    """
    script_path = str(EXAMPLES_DIR / "use_case_5_comparative_stability_analysis.py")

    return job_manager.submit_job(
        script_path=script_path,
        args={
            "variant_files": ",".join(variant_files),
            "wt_fasta": wt_fasta_file,
            "analysis_name": analysis_name,
            "device": device
        },
        job_name=job_name or f"comparative_analysis_{analysis_name}"
    )


# ==============================================================================
# Utility Tools
# ==============================================================================

@mcp.tool()
def validate_input_files(
    variant_file: str,
    wt_fasta_file: Optional[str] = None
) -> dict:
    """
    Validate input files before running predictions.

    This tool checks that input files exist, are properly formatted,
    and estimates the number of variants for runtime planning.

    Args:
        variant_file: Path to variant sequences file
        wt_fasta_file: Path to wild-type FASTA file (optional)

    Returns:
        Dictionary with validation results and recommendations
    """
    try:
        result = {"status": "success", "checks": []}

        # Check variant file
        if not os.path.exists(variant_file):
            return {"status": "error", "error": f"Variant file not found: {variant_file}"}

        result["checks"].append(f"✓ Variant file exists: {variant_file}")

        # Analyze variant file format and size
        is_csv = variant_file.lower().endswith('.csv')
        is_fasta = variant_file.lower().endswith('.fasta') or variant_file.lower().endswith('.fa')

        if is_csv:
            df = pd.read_csv(variant_file)
            if 'seq' not in df.columns:
                return {"status": "error", "error": "CSV must contain a 'seq' column"}
            num_variants = len(df)
            result["checks"].append(f"✓ CSV format validated ({num_variants} variants)")
        elif is_fasta:
            with open(variant_file, 'r') as f:
                content = f.read()
                num_variants = content.count('>')
            result["checks"].append(f"✓ FASTA format detected ({num_variants} sequences)")
        else:
            return {"status": "error", "error": "File must be CSV or FASTA format"}

        result["num_variants"] = num_variants

        # Check wild-type file
        if wt_fasta_file is None:
            wt_fasta_file = str(Path(variant_file).parent / 'wt.fasta')

        if os.path.exists(wt_fasta_file):
            result["checks"].append(f"✓ Wild-type FASTA found: {wt_fasta_file}")
            result["wt_fasta_file"] = wt_fasta_file
        else:
            return {"status": "error", "error": f"Wild-type FASTA not found: {wt_fasta_file}"}

        # Provide runtime recommendations
        if num_variants <= 10:
            result["recommendation"] = "Use predict_stability() for immediate results"
            result["estimated_time"] = "30 seconds - 2 minutes"
        elif num_variants <= 50:
            result["recommendation"] = "Use predict_stability() or submit_stability_prediction()"
            result["estimated_time"] = "2-10 minutes"
        else:
            result["recommendation"] = "Use submit_stability_prediction() for background processing"
            result["estimated_time"] = f"10-{num_variants//10} minutes (depending on hardware)"

        result["summary"] = f"Ready to process {num_variants} variants"
        return result

    except Exception as e:
        logger.error(f"validate_input_files failed: {e}")
        return {"status": "error", "error": str(e)}


@mcp.tool()
def list_example_data() -> dict:
    """
    List available example datasets for testing SPIRED-Stab predictions.

    Returns:
        Dictionary with available example files and their descriptions
    """
    try:
        examples = []

        # Check for example files
        example_files = [
            ("wt.fasta", "Wild-type sequence for testing"),
            ("data.csv", "Example CSV with variant sequences"),
            ("sequences.fasta", "Example FASTA with variant sequences"),
        ]

        for filename, description in example_files:
            filepath = EXAMPLES_DIR / filename
            if filepath.exists():
                # Get file size info
                if filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath)
                        size_info = f"{len(df)} variants"
                    except:
                        size_info = "unknown"
                elif filename.endswith('.fasta'):
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            size_info = f"{content.count('>')} sequences"
                    except:
                        size_info = "unknown"
                else:
                    size_info = "reference file"

                examples.append({
                    "file": str(filepath),
                    "description": description,
                    "size": size_info
                })

        return {
            "status": "success",
            "examples": examples,
            "usage": "Use these files with predict_stability() or submit_stability_prediction()"
        }

    except Exception as e:
        logger.error(f"list_example_data failed: {e}")
        return {"status": "error", "error": str(e)}


# ==============================================================================
# Entry Point
# ==============================================================================

if __name__ == "__main__":
    mcp.run()