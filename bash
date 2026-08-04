Foldmason

$./foldmason easy-msa inputfile.pdb MSA.fasta tmpFolder --report-mode 1  --refine-iters 10$

Mafft 

$mafft-ginsi --thread -1 --aamatrix mafft_3di_matrix.txt non_aligned_3di.fasta > MSA_output.afa$

IQ TREE 3 

$./iqtree3 -s MSA_trimmed.afa -m MFP -mset provided_matrix.txt -redo -nt 6 -B 1000$

Ancestral sequence reconstruction using Python Script

$python ezASRv8_ADAM.py complete -a AA_trimmed.afa -t rooted.tree -m calculated model(e.g. LG+I+G4)$
