"""
AbDiff — Antibody Diffusion and Structure Generation Pipeline
-------------------------------------------------------------
This package implements the full inference pipeline for antibody structure
generation based on the AbDiff protocol.

Modules:
    - af2: AlphaFold2/ColabFold feature extraction
    - igfold: IgFold embedding extraction
    - abfold_encoder: Fusion of AF2 and IgFold embeddings
    - h3_mask: CDR-H3 region mask generation
    - abdiff_sampling: Diffusion sampling (AbDiff model)
    - structure_decode: Final structure decoding (PDB generation)
    - pipeline: Unified end-to-end pipeline
"""
""" 
from . import af2
from . import igfold
from . import abfold_encoder
from . import h3_mask
from . import abdiff_sampling
from . import structure_decode
from . import utils

__all__ = [
    "af2",
    "igfold",
    "abfold_encoder",
    "h3_mask",
    "abdiff_sampling",
    "structure_decode",
    "utils",
]
remove comments status after all Okay"""
