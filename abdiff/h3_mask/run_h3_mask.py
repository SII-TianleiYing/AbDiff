import os
import torch
import argparse
# from abfold.data_process import get_CDRs_mask_with_anarci, process_fasta

from abfold.data.data_process import get_CDRs_mask_with_anarci, process_fasta

def run_h3_mask(fasta_dir: str, output_dir: str):
    """
    Step 4: Generate CDR-H3 Mask
    ----------------------------
    Identifies the CDR-H3 region in antibody heavy/light sequences and outputs a
    binary tensor mask marking residues in the H3 region.

    Inputs:
        fasta_dir: Path to input FASTA directory (contains paired heavy/light seqs)
        output_dir: Path to save generated mask tensors

    Output:
        Each antibody -> <filename>.pt saved in output_dir
            where each file contains a concatenated tensor of (H3 + L3)
    """

    os.makedirs(output_dir, exist_ok=True)
    fasta_files = [f for f in os.listdir(fasta_dir) if f.endswith(".fasta")]

    for file in fasta_files:
        filename = file.split(".")[0]
        fasta_path = os.path.join(fasta_dir, file)
        print(f"[H3 Mask] Processing {filename}...")

        #  Parse FASTA sequences
        seq_h, seq_l = process_fasta(fasta_path)

        #  Generate H3 and L3 masks using ANARCI-based region parsing
        cdr_mask_h, trunk_mask_h, noise_ratio_h = get_CDRs_mask_with_anarci(
            seq_h, chain_type="H", only_H3=True
        )
        cdr_mask_l, trunk_mask_l, noise_ratio_l = get_CDRs_mask_with_anarci(
            seq_l, chain_type="L", only_H3=True
        )

        #  Concatenate masks for heavy + light chains
        cdr_mask = torch.cat((cdr_mask_h, cdr_mask_l), dim=0)

        #  Save
        output_path = os.path.join(output_dir, f"{filename}.pt")
        torch.save(cdr_mask, output_path)
        print(f"[H3 Mask] Saved: {output_path}")

    print(f"\n✅ All masks generated successfully in {output_dir}")


def run_h3_mask_default():
    """
    Default wrapper for your repo layout:
    examples/example_input/fasta_dir → cdr_mask_H3/
    """
    fasta_dir = "examples/example_input/fasta_dir"
    output_dir = "examples/example_input/cdr_mask_H3"
    run_h3_mask(fasta_dir, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CDR-H3 mask tensors")
    parser.add_argument("--fasta_dir", type=str, default="examples/example_input/fasta_dir")
    parser.add_argument("--output_dir", type=str, default="examples/example_input/cdr_mask_H3")
    args = parser.parse_args()

    run_h3_mask(fasta_dir=args.fasta_dir, output_dir=args.output_dir)
