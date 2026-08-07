# Bioinformatic Pipeline Command Summary

================================================================================
1. FoldMason (Structural Alignment)
================================================================================

Command:
./foldmason easy-msa inputfile.pdb MSA.fasta tmpFolder --report-mode 1 --refine-iters 10

Description:
* easy-msa: module generating structural (3Di) and amino acid (AA) alignments
* inputfile.pdb: Input PDB protein structures
* MSA.fasta: Path for output alignment files
* tmpFolder: Directory for intermediate temporary processing files.
* --report-mode 1: Generates a detailed report
* --refine-iters 10: Performs 10 iterative refinement cycles to optimize alignment accuracy

================================================================================
2. MAFFT G-INS-i (3Di Secondary Alignment Refinement)
================================================================================

Command:
mafft-ginsi --thread -1 --aamatrix mafft_3di_matrix.txt non_aligned_3di.fasta > MSA_output.afa

Description:
* mafft-ginsi: High-accuracy global alignment strategy with iterative refinement.
* --thread -1: Automatically allocates all available CPU cores for execution zipping.
* --aamatrix mafft_3di_matrix.txt: Enforces a custom substitution matrix specific to the 3Di structural alphabet
* non_aligned_3di.fasta: Input 3Di sequences exported from FoldMason
* > MSA_output.afa: Saves the refined 3Di structural alignment to the specified FASTA file

================================================================================
3. IQ-TREE 3 (3Di Structural Tree Inference)
================================================================================

Command:
./iqtree3 -s MSA_trimmed.afa -m MFP -mset provided_matrix.txt -redo -nt 6 -B 1000

Description:
* -s MSA_trimmed.afa: Input matrix of the trimmed 3Di alignment[cite: 1].
* -m MFP: ModelFinder Plus for automatic substitution model selection under BIC/AICc[cite: 1].
* -mset provided_matrix.txt: Restricts model selection strictly to the empirical 3Di structural matrix[cite: 1].
* -redo: Overwrites prior runs and forces clean calculation.
* -nt 6: Allocates 6 CPU threads for multi-threaded processing.
* -B 1000: Calculates 1,000 Ultrafast Bootstrap (UFBoot2) replicates for node support evaluation[cite: 1].

================================================================================
4. ezASR Pipeline (Ancestral Sequence Reconstruction & Indel Reconciliation)
================================================================================

Command:
python ezASRv8_ADAM.py complete -a AA_trimmed.afa -t rooted.tree -m LG+I+G4/Q.3Di.AF 

Description:
* complete: Executes the full end-to-end ASR workflow (node labeling -> ML ASR -> gap parsimony -> sequence reconciliation -> probability plotting)[cite: 1].
* -a AA_trimmed.afa: Primary amino acid alignment mapped 1:1 to trimmed 3Di positions[cite: 1, 2].
* -t rooted.tree: Pre-calculated, rooted 3Di structural phylogenetic tree topology[cite: 1].
* -m LG+I+G4: Optimal empirical AA substitution model determined via ModelFinder (e.g., LG+I+G4 for GLK; BLOSUM62+F+I+R4 for fbaA).
* --auto: Non-interactive mode that bypasses manual terminal prompts[cite: 1].

================================================================================
SUMMARY
================================================================================
Foldmason
$./foldmason easy-msa inputfile.pdb MSA.fasta tmpFolder --report-mode 1  --refine-iters 10$
Mafft 
$mafft-ginsi --thread -1 --aamatrix mafft_3di_matrix.txt non_aligned_3di.fasta > MSA_output.afa$
IQ TREE 3 
$./iqtree3 -s MSA_trimmed.afa -m MFP -mset provided_matrix.txt -redo -nt 6 -B 1000$
Ancestral sequence reconstruction using Python Script
$python ezASRv8_ADAM.py complete -a AA_trimmed.afa -t rooted.tree -m calculated model(e.g. LG+I+G4)$

