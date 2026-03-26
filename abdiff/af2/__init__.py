"""
abdiff.af2
========
This module provides:
- 'run_colabfold()' : run ColabFold and save single/pair representations
- 'merge_representations()' : merge single and pair .npy files into .pkl
"""

from .run_af2 import run_colabfold, merge_representations
__all__ = ["run_colabfold", "merge_representations", "run_af2_default"]
