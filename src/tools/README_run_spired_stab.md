# run_spired_stab Tools - Implementation Summary

## Overview
Extracted **1 production-ready tool** from the SPIRED-Stab tutorial for protein stability prediction.

**Source**: https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED-Stab.py

## Tools Extracted

### 1. spired_predict_mutation_stability
**Purpose**: Predict protein stability changes from mutations using SPIRED-Stab deep learning model

**Input**:
- FASTA file containing paired wild-type and mutant protein sequences

**Output**:
- Stability metrics: ddG (Gibbs free energy change), dTm (melting temperature change)
- 3D protein structures: Top 8 confidence CA structures for wild-type and mutant
- GDFold2-compatible NPZ files for structure refinement

**Parameters**:
- `fasta_path` (required): Path to FASTA file with 2 sequences (WT, mutant)
- `out_prefix` (optional): Output file prefix (default: auto-generated with timestamp)

**Use Case**: Protein engineers evaluating mutation effects on thermal stability

**Example**:
```python
result = spired_predict_mutation_stability(
    fasta_path="/path/to/sequences.fasta",
    out_prefix="my_protein_v5g_y54g"
)
# Returns: ddG=-2.269 kcal/mol, dTm=-12.572 °C
```

## Implementation Quality

### Tutorial Fidelity
✅ **Exact reproduction**: All function calls, data structures, and processing steps match tutorial
- Model initialization: `SPIRED_Stab(device_list=["cpu", "cpu", "cpu", "cpu"])`
- ESM-2 models: Exact model names and loading procedures
- Structure selection: `torch.topk(plddt_value_L, 8)` - exactly 8 top structures
- PDB format: Exact column widths and formatting

### Parameter Design
✅ **Conservative parameterization**: Only parameterized user data inputs
- Tutorial-specific paths → `fasta_path` parameter
- Tutorial output location → `out_prefix` parameter
- **NO additional parameters** beyond what tutorial explicitly set

### Real-World Applicability
✅ **Production ready**: Designed for actual protein engineering workflows
- Accepts user's protein sequences
- Produces quantitative stability metrics
- Generates visualization-ready PDB structures
- Compatible with downstream refinement tools (GDFold2)

### Error Handling
✅ **Basic validation only**: Following guidelines for minimal error control
- Input file existence
- Sequence count validation (exactly 2)
- Sequence length matching
- Model weights availability

## Output Structure

```
output_dir/
├── pred.csv                          # Stability predictions (ddG, dTm)
├── wt/
│   ├── CA_structure/
│   │   ├── 0.pdb                    # Highest confidence WT structure
│   │   ├── 1.pdb
│   │   └── ...                      # Up to 7.pdb (8 total)
│   └── GDFold2/
│       └── input.npz                # WT structure features
└── mut/
    ├── CA_structure/
    │   ├── 0.pdb                    # Highest confidence mutant structure
    │   ├── 1.pdb
    │   └── ...                      # Up to 7.pdb (8 total)
    └── GDFold2/
        └── input.npz                # Mutant structure features
```

## Dependencies

### Python Packages
- torch (PyTorch)
- numpy
- pandas
- biopython
- fastmcp

### Pre-trained Models
- ESM-2 650M (downloaded via torch.hub)
- ESM-2 3B (downloaded via torch.hub)
- SPIRED-Stab weights (included in repository)

### System Requirements
- **Memory**: ~10GB RAM (for ESM-2 3B model)
- **Compute**: CPU-based (GPU support not implemented)
- **Storage**: ~5GB for ESM-2 models

## Validation Results

All quality checks passed on first iteration:

### Tool Design Validation ✅
- Clear naming convention (spired_predict_mutation_stability)
- Two-sentence docstring
- Self-explanatory parameters
- Independently usable

### Implementation Validation ✅
- Complete tutorial coverage
- File path as primary input
- Basic input validation only
- Exact tutorial function calls preserved
- No hardcoded tutorial-specific values

### Output Validation ✅
- Standardized return format (dict with message, reference, artifacts)
- Absolute file paths
- Correct GitHub reference link
- All tutorial outputs saved

### Code Quality Validation ✅
- Type annotations with Annotated
- Clear documentation
- Template compliance
- Proper import management

## Known Limitations

1. **CPU Only**: Tutorial uses CPU computation; GPU acceleration not implemented
2. **Fixed Architecture**: Always uses 4-block SPIRED-Stab architecture
3. **Structure Count**: Fixed at 8 structures (cannot customize)
4. **Single Pair**: Processes one WT-mutant pair per call (no batch mode)
5. **Memory Intensive**: Requires ~10GB RAM for ESM-2 3B model

These limitations reflect the tutorial's design and are intentionally preserved for exact reproducibility.

## Scientific Context

**SPIRED-Stab** is a deep learning model for predicting protein stability changes from mutations:
- **ddG**: Change in Gibbs free energy (stability indicator)
  - Negative ddG → decreased stability
  - Positive ddG → increased stability
- **dTm**: Change in melting temperature
  - Negative dTm → lower melting point
  - Positive dTm → higher melting point

**Typical Applications**:
- Rational protein design
- Variant effect prediction
- Therapeutic protein optimization
- Enzyme engineering

## Files Generated

1. **src/tools/run_spired_stab.py** - Production tool implementation
2. **src/tools/run_spired_stab_implementation_log.md** - Detailed extraction log
3. **src/tools/README_run_spired_stab.md** - This summary document

## References

- **Tutorial Source**: https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED-Stab.py
- **Execution Notebook**: notebooks/run_spired_stab/run_spired_stab_execution_final.ipynb
- **SPIRED-Fitness Repository**: repo/SPIRED-Fitness/
- **ESM-2 Paper**: Lin et al., "Language models of protein sequences at the scale of evolution" (2022)
