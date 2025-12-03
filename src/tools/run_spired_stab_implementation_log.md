# Implementation Log: run_spired_stab Tool Extraction

## Tutorial Overview
- **Tutorial**: run_SPIRED-Stab.py
- **Source**: https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED-Stab.py
- **Purpose**: Predict protein stability changes from mutations using deep learning
- **Notebook**: notebooks/run_spired_stab/run_spired_stab_execution_final.ipynb

## Tool Design Decisions

### Tool Classification
The tutorial implements a single complete analytical workflow for protein stability prediction. This was designed as **ONE tool** because:

1. **Single Scientific Task**: Predicts stability changes from protein mutations
2. **Inseparable Components**: All steps (embedding generation, structure prediction, stability calculation) must run together
3. **Unified Output**: Produces coherent set of outputs (ddG, dTm, structures) that belong together
4. **Natural Workflow**: Users want complete stability analysis, not individual sub-components

### Tool Definition
**Tool Name**: `spired_predict_mutation_stability`
- **Format**: Follows `library_action_target` convention (spired = library, predict = action, mutation_stability = target)
- **Rationale**: Name clearly indicates the tool performs stability prediction for mutations using SPIRED
- **Classification**: ✅ Applicable to New Data
  - Accepts user-provided FASTA files
  - Performs repeatable analysis on different protein sequences
  - Produces useful outputs (stability metrics + structures)
  - Implements non-trivial deep learning pipeline

### Parameter Design

#### Primary Input
- **fasta_path**: Path to FASTA file containing wild-type and mutant sequences
  - **Type**: str | None (required, validated in function)
  - **Rationale**: FASTA is standard format for protein sequences; tutorial uses paired sequences
  - **Validation**: File existence check only

#### Analysis Parameters
- **out_prefix**: Output file prefix for organizing results
  - **Type**: str | None
  - **Default**: f"spired_stab_{timestamp}" (auto-generated if not provided)
  - **Rationale**: Tutorial used fixed output path; parameterized for user flexibility

#### Parameters NOT Added (Following Tutorial Exactly)
- **No model configuration**: Tutorial uses fixed model architecture (device_list=["cpu", "cpu", "cpu", "cpu"])
- **No ESM-2 parameters**: Tutorial uses models with default settings
- **No structure count**: Tutorial always generates top 8 structures
- **No pLDDT threshold**: Tutorial uses top 8 by confidence without filtering

This follows the CRITICAL RULE: "NEVER ADD PARAMETERS NOT IN TUTORIAL"

### Implementation Choices

#### Library Integration
- **ESM-2 Models**: Loaded via torch.hub as in tutorial
- **SPIRED-Stab**: Imported from repository's scripts.model module
- **Utilities**: Used getStabDataTest from scripts.utils_train_valid exactly as tutorial

#### Data Flow
1. Load paired sequences from FASTA (WT first, mutant second)
2. Generate ESM-2 embeddings for both sequences
3. Run SPIRED-Stab model to predict ddG and dTm
4. Extract top 8 structural predictions by pLDDT
5. Save CSV (stability), NPZ (GDFold2), and PDB files (structures)

#### Output Structure
Preserves exact tutorial output organization:
```
output_dir/
├── pred.csv                    # Stability predictions
├── wt/
│   ├── CA_structure/          # Wild-type PDB files (0-7.pdb)
│   └── GDFold2/input.npz      # Wild-type structure features
└── mut/
    ├── CA_structure/          # Mutant PDB files (0-7.pdb)
    └── GDFold2/input.npz      # Mutant structure features
```

#### Error Handling
Implemented basic validation only:
- FASTA file existence check
- Sequence count validation (must be exactly 2)
- Sequence length matching (WT and mutant same length)
- Model weights file existence check

No additional error handling beyond what's necessary for input validation, following the guideline: "Basic input file validation only"

### Tutorial Fidelity Checks

#### Exact Function Call Preservation
✅ All library calls match tutorial exactly:
- `SPIRED_Stab(device_list=["cpu", "cpu", "cpu", "cpu"])` - exact parameters
- `torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")` - exact model
- `getStabDataTest(seq, esm2_3B, esm2_650M, esm2_batch_converter)` - exact parameters
- `torch.topk(plddt_value_L, 8)` - exact k=8 from tutorial

#### Data Structure Preservation
✅ All data structures preserved exactly:
- Amino acid dictionary (AA_DICT) - exact 3-letter codes
- PDB format string - exact column widths and formatting
- NPZ keys (reference, translation, dihedrals, plddt) - exact naming
- Tensor operations and permutations - exact order

#### No Tutorial Data Fallbacks
✅ No default tutorial data:
- Requires user-provided FASTA file
- Raises clear error if fasta_path is None
- No fallback to example_stab/test.fasta

#### No Demonstration Code
✅ No convenience variables:
- Uses user's full sequences (wt_seq, mut_seq)
- Processes all mutations found between sequences
- No first_mutation or sample_structure shortcuts

## Quality Review Results

### Iteration 1: Initial Implementation

#### Tool Design Validation
- [✓] Tool name clearly indicates functionality
- [✓] Tool description explains when to use and I/O expectations
- [✓] Parameters are self-explanatory with documented possible values
- [✓] Return format documented in docstring
- [✓] Independently usable with no hidden state
- [✓] Accepts user data inputs and produces specific outputs
- [✓] Discoverable via name and description

#### Input/Output Validation
- [✓] Exactly-one-input rule enforced (raises ValueError otherwise)
- [✓] Primary input parameter uses FASTA format (standard for sequences)
- [✓] Basic input file validation implemented (file existence only)
- [✓] Defaults represent recommended tutorial parameters (N/A - no analysis params)
- [✓] All artifact paths are absolute
- [✓] No hardcoded values that should adapt to user input context
- [✓] Context-dependent identifiers, ranges, and references are parameterized

#### Tutorial Logic Adherence Validation
- [✓] Function parameters are actually used (no convenience substitutions)
- [✓] Processing follows tutorial's exact workflow
- [✓] User-provided parameters drive the analysis
- [✓] No convenience variables that bypass user inputs
- [✓] Implementation matches tutorial's specific logic flow
- [✓] **CRITICAL**: Function calls exactly match tutorial
- [✓] **CRITICAL**: Preserve exact data structures

#### Implementation Validation
- [✓] Function coverage: All tutorial analytical steps have corresponding code
- [✓] Parameter design: File path as primary input, tutorial defaults preserved
- [✓] Input validation: Basic FASTA file validation implemented
- [✓] Tutorial fidelity: When run with tutorial data, produces identical results
- [✓] Real-world focus: Designed for actual mutation analysis use cases
- [✓] No hardcoding: No values tied to tutorial's specific example
- [✓] Library compliance: Uses exact tutorial libraries and patterns
- [✓] **CRITICAL**: Exact function calls (no added parameters)

#### Output Validation
- [✓] Figure generation: N/A (tutorial generates no matplotlib figures)
- [✓] Data outputs: CSV for stability, NPZ for structures, PDB for visualization
- [✓] Return format: Standardized dict with message, reference, artifacts
- [✓] File paths: All artifact paths are absolute
- [✓] Reference links: Correct GitHub URL from executed_notebooks.json

#### Code Quality Validation
- [✓] Error handling: Basic input file validation only
- [✓] Type annotations: All parameters use Annotated types with descriptions
- [✓] Documentation: Clear docstrings with usage guidance
- [✓] Template compliance: Follows implementation template structure
- [✓] Import management: All required imports present
- [✓] Environment setup: Proper directory structure and environment variables

### Summary
**Tools evaluated**: 1 of 1
**Passing all checks**: 1 | **Requiring fixes**: 0
**Current iteration**: 1 of 3 maximum

**Status**: ✅ All quality checks passed on first iteration

## Key Implementation Highlights

### Strengths
1. **Exact Tutorial Preservation**: All model calls, data structures, and processing steps match tutorial exactly
2. **Comprehensive Output**: Saves all tutorial outputs (CSV, NPZ, PDB files)
3. **Clear Documentation**: Two-sentence docstring follows template
4. **Proper Validation**: Basic input checks without over-engineering
5. **Production Ready**: Can be used immediately on user's protein sequences

### Limitations Surfaced
1. **CPU Only**: Tutorial uses CPU; GPU support would require additional parameterization
2. **Fixed Structure Count**: Always generates 8 structures (tutorial default)
3. **No Batch Processing**: Processes one mutation pair at a time
4. **Memory Requirements**: ESM-2 3B model requires significant RAM

### Real-World Applicability
- **Primary Use Case**: Protein engineers evaluating mutation effects on stability
- **Input Requirements**: Paired FASTA with aligned WT and mutant sequences
- **Output Value**: Quantitative stability metrics (ddG, dTm) and structural models
- **Integration**: Outputs compatible with downstream GDFold2 refinement

## Files Generated
1. `/src/tools/run_spired_stab.py` - Production tool implementation (1 tool)
2. `/src/tools/run_spired_stab_implementation_log.md` - This documentation

## References
- Tutorial: https://github.com/Gonglab-THU/SPIRED-Fitness/blob/main/run_SPIRED-Stab.py
- Execution notebook: notebooks/run_spired_stab/run_spired_stab_execution_final.ipynb
- SPIRED-Fitness repository: repo/SPIRED-Fitness/
