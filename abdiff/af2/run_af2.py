import os
import sys
import argparse
import pickle
import shutil
from pathlib import Path
from typing import Optional, Set

import numpy as np
from colabfold.batch import main as colabfold_main


def run_colabfold(
    fasta_csv: str,
    repr_dir: str,
    af2_repr_dir: str,
    num_recycle: int = 3,
    model_type: str = "AlphaFold2-multimer-v2",
    template_mode: str = "server",   # none | server | local
    template_dir: Optional[str] = None,
    device: str = "cpu",           # cpu | gpu
    cache_dir: Optional[str] = None,
    num_models: int = 1,
    model_order: int = 3,
    random_seed: int = 42,
):
    """
    Run ColabFold to extract single/pair representations, then merge them into pkl files.

    Parameters
    ----------
    fasta_csv : str
        Path to input CSV for ColabFold.
    repr_dir : str
        Directory where raw ColabFold representation .npy files are written.
    af2_repr_dir : str
        Directory where merged .pkl files are written.
    num_recycle : int
        Number of recycles.
    model_type : str
        ColabFold model type.
    template_mode : str
        One of:
        - "none":  no templates
        - "server": use ColabFold default template retrieval
        - "local": use user-provided local template directory
    template_dir : Optional[str]
        Required when template_mode == "local".
    device : str
        "cpu" or "gpu". Default is "cpu".
    cache_dir : Optional[str]
        Cache directory for ColabFold/JAX-related downloads and cache.
    num_models : int
        Number of models to run. Current merge logic assumes num_models=1.
    model_order : int
        Which model order to use.
    random_seed : int
        Random seed.
    """
    # ---------- basic validation ----------
    fasta_csv = str(Path(fasta_csv).expanduser().resolve())
    repr_dir = str(Path(repr_dir).expanduser().resolve())
    af2_repr_dir = str(Path(af2_repr_dir).expanduser().resolve())

    if not Path(fasta_csv).exists():
        raise FileNotFoundError(f"Input fasta_csv not found: {fasta_csv}")

    if template_mode not in {"none", "server", "local"}:
        raise ValueError(f"template_mode must be one of: none, server, local. Got: {template_mode}")

    if device not in {"cpu", "gpu"}:
        raise ValueError(f"device must be one of: cpu, gpu. Got: {device}")

    if template_mode == "local":
        if not template_dir:
            raise ValueError("template_mode='local' requires --template_dir")
        template_dir = str(Path(template_dir).expanduser().resolve())
        if not Path(template_dir).exists():
            raise FileNotFoundError(f"Local template_dir not found: {template_dir}")

    # templates need hhsearch in your tested setup
    if template_mode in {"server", "local"}:
        if shutil.which("hhsearch") is None:
            raise RuntimeError(
                "Templates are enabled, but 'hhsearch' was not found in PATH. "
                "Please install hhsuite into this environment first."
            )

    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/colabfold")
    cache_dir = str(Path(cache_dir).expanduser().resolve())

    # ---------- env vars ----------
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "4.0"
    os.environ["XDG_CACHE_HOME"] = cache_dir

    Path(repr_dir).mkdir(parents=True, exist_ok=True)
    Path(af2_repr_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # ---------- build argv ----------
    argv = [
        "colabfold_batch",
        "--num-recycle", str(num_recycle),
        "--num-models", str(num_models),
        "--model-order", str(model_order),
        "--random-seed", str(random_seed),
        "--model-type", model_type,
        "--save-single-representations",
        "--save-pair-representations",
    ]

    if template_mode != "none":
        argv.append("--templates")

    if template_mode == "local":
        # If your installed ColabFold 1.3.0 uses a different flag name,
        # run: colabfold_batch --help | grep -i template
        argv += ["--custom-template-path", template_dir]

    if device == "cpu":
        argv.append("--cpu")
    # if device == "gpu", do NOT add --cpu

    argv += [fasta_csv, repr_dir]

    # ---------- run ----------
    print(f"[INFO] Running ColabFold internal main(): {' '.join(argv)}")

    old_argv = sys.argv[:]
    try:
        sys.argv = argv
        colabfold_main()
    finally:
        sys.argv = old_argv

    # ---------- merge ----------
    print("[INFO] Merging single/pair representations...")
    stats = merge_representations(
        input_dir=repr_dir,
        output_dir=af2_repr_dir,
        model_order=model_order,
    )
    print(
        "[INFO] Merge done. "
        f"found_ids={stats['found_ids']}, "
        f"merged={stats['merged']}, "
        f"skipped_missing={stats['skipped_missing']}, "
        f"skipped_existing={stats['skipped_existing']}"
    )
    print(f"[INFO] Merged representations saved to {af2_repr_dir}")


def _collect_repr_ids(input_dir: str, model_order: int) -> Set[str]:
    """
    Collect representation IDs from ColabFold output filenames.

    We only target filenames like:
      {rid}_single_repr_1_model_{model_order}.npy
      {rid}_pair_repr_1_model_{model_order}.npy
    """
    input_dir = str(Path(input_dir).resolve())

    single_suffix = f"_single_repr_1_model_{model_order}.npy"
    pair_suffix = f"_pair_repr_1_model_{model_order}.npy"

    repr_ids: Set[str] = set()

    for fname in os.listdir(input_dir):
        if fname.endswith(single_suffix):
            repr_ids.add(fname[:-len(single_suffix)])
        elif fname.endswith(pair_suffix):
            repr_ids.add(fname[:-len(pair_suffix)])

    return repr_ids


def merge_representations(input_dir: str, output_dir: str, model_order: int = 3):
    """
    Merge ColabFold single/pair .npy files into {id}.pkl

    Returns
    -------
    dict
        {
          "found_ids": int,
          "merged": int,
          "skipped_missing": int,
          "skipped_existing": int,
        }
    """
    input_dir = str(Path(input_dir).expanduser().resolve())
    output_dir = str(Path(output_dir).expanduser().resolve())

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    repr_ids = _collect_repr_ids(input_dir=input_dir, model_order=model_order)

    stats = {
        "found_ids": len(repr_ids),
        "merged": 0,
        "skipped_missing": 0,
        "skipped_existing": 0,
    }

    for rid in sorted(repr_ids):
        single_path = os.path.join(input_dir, f"{rid}_single_repr_1_model_{model_order}.npy")
        pair_path = os.path.join(input_dir, f"{rid}_pair_repr_1_model_{model_order}.npy")
        output_path = os.path.join(output_dir, f"{rid}.pkl")

        if not os.path.exists(single_path) or not os.path.exists(pair_path):
            stats["skipped_missing"] += 1
            continue

        if os.path.exists(output_path):
            stats["skipped_existing"] += 1
            continue

        single = np.load(single_path, allow_pickle=False)
        pair = np.load(pair_path, allow_pickle=False)

        repr_obj = {
            "single": single,
            "pair": pair,
        }

        with open(output_path, "wb") as f:
            pickle.dump(repr_obj, f, protocol=4)

        stats["merged"] += 1

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fasta_csv", required=True)
    parser.add_argument("--repr_dir", required=True)
    parser.add_argument("--af2_repr_dir", required=True)

    parser.add_argument("--num_recycle", type=int, default=3)
    parser.add_argument("--model_type", default="AlphaFold2-multimer-v2")

    # new
    parser.add_argument(
        "--template_mode",
        choices=["none", "server", "local"],
        default="server",
        help="none: no templates; server: default ColabFold template retrieval; local: use local template_dir",
    )
    parser.add_argument(
        "--template_dir",
        default=None,
        help="Required when --template_mode local",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu"],
        default="cpu",
        help="Default cpu. Use gpu only if your JAX/CUDA environment is already working.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        help="Optional cache dir for ColabFold/JAX downloads and cache",
    )

    # optional exposed controls
    parser.add_argument("--num_models", type=int, default=1)
    parser.add_argument("--model_order", type=int, default=3)
    parser.add_argument("--random_seed", type=int, default=42)

    args = parser.parse_args()

    run_colabfold(
        fasta_csv=args.fasta_csv,
        repr_dir=args.repr_dir,
        af2_repr_dir=args.af2_repr_dir,
        num_recycle=args.num_recycle,
        model_type=args.model_type,
        template_mode=args.template_mode,
        template_dir=args.template_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        num_models=args.num_models,
        model_order=args.model_order,
        random_seed=args.random_seed,
    )
