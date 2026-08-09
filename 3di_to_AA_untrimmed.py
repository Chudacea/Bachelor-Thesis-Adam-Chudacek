#translate untrimmed 3di realignemnt into corresponding amino acids   
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

def align_amino_acids_to_3di(three_di: str, amino_acids_with_gaps: str) -> str:
    amino_acids = amino_acids_with_gaps.replace('-', '')
    aa_index = 0
    aligned_aa = []

    for symbol in three_di:
        if symbol == '-':
            aligned_aa.append('-')
        else:
            if aa_index >= len(amino_acids):
                raise ValueError("Too few amino acids for the number of non-gap 3Di characters.")
            aligned_aa.append(amino_acids[aa_index])
            aa_index += 1

    if aa_index != len(amino_acids):
        raise ValueError("Too many amino acids for the 3Di alignment.")

    return ''.join(aligned_aa)

def process_fasta_alignments(three_di_file: str, aa_file: str, output_file: str):
    # Load sequences into dictionaries by ID
    three_di_dict = SeqIO.to_dict(SeqIO.parse(three_di_file, "fasta"))
    aa_dict = SeqIO.to_dict(SeqIO.parse(aa_file, "fasta"))

    aligned_records = []

    for seq_id, three_di_record in three_di_dict.items():
        if seq_id not in aa_dict:
            raise KeyError(f"Sequence ID {seq_id} not found in amino acid file.")

        aligned_seq = align_amino_acids_to_3di(
            str(three_di_record.seq),
            str(aa_dict[seq_id].seq)
        )

        aligned_record = SeqRecord(Seq(aligned_seq), id=seq_id, description="")
        aligned_records.append(aligned_record)

    # Write to output FASTA
    SeqIO.write(aligned_records, output_file, "fasta")

# ------------------------------
# 📝 Define Input and Output Files
# ------------------------------
if __name__ == "__main__":
    # Replace these with your actual file paths if needed
    three_di_input_file = "" #untrimmed 3di alignment (mafft output)
    amino_acid_input_file = "" #untrimmed amino acid alignment (foldmason output)
    output_file = "" #output file path, for me its the AA.afa

    process_fasta_alignments(three_di_input_file, amino_acid_input_file, output_file)
    print(f"Aligned sequences written to: {output_file}")
