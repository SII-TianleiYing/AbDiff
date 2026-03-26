import os
import argparse
from pathlib import Path
import torch
from igfold import IgFoldRunner

def run_igfold(
    fasta_dir: str,
    output_dir: str,
    device: str = "cuda:0",
):
    """
    Step 2: Extract IgFold Features.
    This is adapted from the lab's get_features.py. :contentReference[oaicite:3]{index=3}

    What it does:
    - Reads antibody heavy/light sequences from each FASTA file in fasta_dir.
      FASTA format is expected to be:
          >ID_H
          <heavy_chain_sequence>
          >ID_L
          <light_chain_sequence>

    - Runs IgFoldRunner.embed(sequences=...) to get embeddings for that antibody.

    - Saves a .pt file per antibody ID containing:
          {
              "structure_embs": emb.structure_embs.squeeze(0),
              "prmsd":          emb.prmsd.squeeze(0)
          }

    - These .pt files are written to output_dir.
      This directory is what the pipeline calls igfold_embedding/.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize IgFold
    igfold_module = IgFoldRunner()

    # Loop over all FASTA files
    for filename in sorted(os.listdir(fasta_dir)):
        if not filename.endswith(".fasta"):
            continue

        fasta_path = os.path.join(fasta_dir, filename)
        with open(fasta_path, "r") as f:
            data = f.readlines()

        # data[1] is heavy chain seq, data[3] is light chain seq in the lab format. :contentReference[oaicite:4]{index=4}
        sequences = {
            "H": data[1].strip(),
            "L": data[3].strip(),
        }

        print(f"[IgFold] Embedding {filename} ...")

        # Call IgFold to get embeddings
        emb = igfold_module.embed(
            sequences=sequences  # Antibody sequences
        )

        # Prepare the output dictionary
        embedding = {
            "structure_embs": emb.structure_embs.squeeze(0),
            "prmsd": emb.prmsd.squeeze(0),
        }

        out_name = filename.replace(".fasta", ".pt")
        out_path = os.path.join(output_dir, out_name)

        torch.save(embedding, out_path)
        print(f"[IgFold] Saved features for {filename} -> {out_path}")

    print(f"[IgFold] Done. All embeddings in {output_dir}")


def run_igfold_default():
    """
    Convenience wrapper for pipeline.py.

    We assume:
    - FASTA files live in examples/example_input/fasta_dir/
    - Output should go to examples/example_input/igfold_embedding/

    These folder names match how we're organizing the full pipeline
    (fasta_dir is from preparation, igfold_embedding is the IgFold output).
    """

    fasta_dir = "examples/example_input/fasta_dir"
    output_dir = "examples/example_input/igfold_embedding"

    run_igfold(
        fasta_dir=fasta_dir,
        output_dir=output_dir,
        device="cuda:0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run IgFold to extract per-antibody embeddings and save them as .pt files."
    )

    parser.add_argument(
        "--fasta_dir",
        type=str,
        default="examples/example_input/fasta_dir",
        help="Directory of .fasta files (each with heavy+light chain).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/example_input/igfold_embedding",
        help="Where to save IgFold embeddings (.pt per antibody).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for IgFoldRunner()",
    )

    args = parser.parse_args()

    run_igfold(
        fasta_dir=args.fasta_dir,
        output_dir=args.output_dir,
        device=args.device,
    )
