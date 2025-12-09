"""
AbDiff End-to-End Inference Pipeline

This pipeline runs the full antibody generation workflow described in the protocol:

1. AlphaFold2 / ColabFold features           -> AF2_repr/
2. IgFold embeddings                         -> igfold_embedding/
3. AbFold encoder fusion                     -> abfold_embedding/
4. CDR-H3 mask extraction                    -> cdr_mask_H3/
5. AbDiff diffusion sampling (inpainting)    -> gen_abdiff_embeddings/
6. Final 3D structure decoding (PDB output)  -> gen_structures/

Input: antibody sequences (FASTA files + test.csv)
Output: predicted 3D antibody structures (.pdb)
"""

import os
import torch

# --- Stage 1: AF2 / ColabFold ---
from abdiff.af2.run_af2 import run_af2_default

# --- Stage 2: IgFold embeddings ---
from abdiff.igfold.run_igfold import run_igfold_default

# --- Stage 3: AbFold encoder ---
from abdiff.abfold_encoder.run_abfold_encoder import run_abfold_encoder_default

# --- Stage 4: H3 mask ---
from abdiff.h3_mask.run_h3_mask import run_h3_mask_default

# --- Stage 5: AbDiff diffusion sampling ---
#   NOTE: sample_embedding.py is already written by the lab.
#   We'll wrap it lightly here instead of rewriting it.
from abdiff.abdiff_sampling.sample_embedding import main as run_abdiff_sampling_main

# --- Stage 6: Structure decode ---
from abdiff.structure_decode.run_structure_decode import run_structure_decode_default


def run_pipeline(
    device: str = None,
):
    """
    Run all stages in order using default repo paths.
    Assumes you've populated:
      examples/example_input/test.csv
      examples/example_input/fasta_dir/*.fasta
    and you have:
      checkpoints/abfold/checkpoint_ema.pt
      checkpoints/std_mean.pt (if your sampling step needs it)
    """

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Pipeline] Using device: {device}")

    # 1. AlphaFold2 / ColabFold features
    print("\n[Pipeline: Step 1/6] AlphaFold2 feature extraction (AF2_repr/)")
    run_af2_default()

    # 2. IgFold embeddings
    print("\n[Pipeline: Step 2/6] IgFold embedding extraction (igfold_embedding/)")
    run_igfold_default()

    # 3. AbFold encoder fusion
    print("\n[Pipeline: Step 3/6] AbFold encoder fusion (abfold_embedding/)")
    run_abfold_encoder_default()

    # 4. H3 Mask
    print("\n[Pipeline: Step 4/6] Generating CDR-H3 mask (cdr_mask_H3/)")
    run_h3_mask_default()

    # 5. AbDiff diffusion sampling
    print("\n[Pipeline: Step 5/6] Running AbDiff sampling (gen_abdiff_embeddings/)")
    # Your sample_embedding.py expects CLI args normally.
    # We'll mimic that here by just calling `main()` with defaults.
    run_abdiff_sampling_main()

    # 6. Structure decoding
    print("\n[Pipeline: Step 6/6] Decoding final structures to PDB (gen_structures/)")
    run_structure_decode_default()

    print("\n✅ Pipeline complete. Check examples/example_output/gen_structures/ for final PDB files.")


if __name__ == "__main__":
    run_pipeline()
