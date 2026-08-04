Foldmason

$./foldmason easy-msa inputfile.pdb MSA.fasta tmpFolder --report-mode 1  --refine-iters 10$


Mafft 

mafft-ginsi --thread -1 --aamatrix mafft_3di_matrix.txt non_aligned_3di.fasta > MSA_output.afa
