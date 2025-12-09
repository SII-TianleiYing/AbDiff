import os
import sys
import argparse
import pickle
import numpy as np
from pathlib import Path

# ColabFold batch runner (AlphaFold2 multimer wrapper)
from colabfold.batch import main as colabfold_main


def run_colabfold(
    fasta_csv: str,
    repr_dir: str,
    af2_repr_dir: str,
    num_recycle: int = 3,
    model_type: str = "AlphaFold2-multimer-v2",
    use_templates: bool = True,
):
    """
    Step 1 of the pipeline: Extract AlphaFold2 Features.
    This follows the protocol:

    1. Run colabfold_batch on the antibody heavy/light sequences
       (seqs_csv_path) to produce residue-level single and pair
       representations for each antibody ID. Output -> repr_dir/.
       

    2. Merge those into final per-antibody feature pickles in AF2_repr_dir/.
       Each pickle will contain:
           { "single": <single repr>, "pair": <pair repr> }
    """

    # Recreate environment behavior from lab run
    os.environ["TF_FORCE_UNIFIED_MEMORY"] = "1"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "4.0"
    os.environ["XDG_CACHE_HOME"] = os.path.expanduser("~/.cache/colabfold")

    Path(repr_dir).mkdir(parents=True, exist_ok=True)
    Path(af2_repr_dir).mkdir(parents=True, exist_ok=True)

    # Build the same style of command arguments shown in protocol:
    # bin/colabfold_batch --templates --num-recycle 3 --model-type AlphaFold2-multimer-v2 \
    #   seqs_csv_path repr_dir/ \
    #   --save-single-representations --save-pair-representations
    # 
    argv = [
        "colabfold_batch",
        "--num-recycle", str(num_recycle),
        "--num-models", "1",
        "--model-order", "3",
        "--random-seed", "42",
        "--model-type", model_type,
    ]

    if use_templates:
        argv.append("--templates")

    argv += [
        "--save-single-representations",
        "--save-pair-representations",
        fasta_csv,
        repr_dir,
    ]

    print(f"[AF2] Running ColabFold with args:\n  {' '.join(argv)}")
    # We call the colabfold main() directly, same as if we ran the CLI.
    sys.argv = argv
    colabfold_main()

    print(f"[AF2] Merging per-ID single/pair reprs into final .pkl files...")
    merge_representations(input_dir=repr_dir, output_dir=af2_repr_dir)

    print(f"[AF2] Done. Final AF2 features in: {af2_repr_dir}")


def merge_representations(input_dir: str, output_dir: str):
    """
    After colabfold_batch runs, repr_dir/ will contain .npy like:
        <ID>_single_repr_1_model_3.npy
        <ID>_pair_repr_1_model_3.npy
    The protocol's second line:
        python merge_single_pair_repr.py --input_dir repr_dir --output_dir AF2_repr_dir
    does exactly this merge. 
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Collect antibody IDs from the filenames
    repr_ids = {
        f.split('_')[0]
        for f in os.listdir(input_dir)
        if f.endswith('.npy')
    }

    for rid in repr_ids:
        single_path = os.path.join(input_dir, f"{rid}_single_repr_1_model_3.npy")
        pair_path   = os.path.join(input_dir, f"{rid}_pair_repr_1_model_3.npy")
        out_path    = os.path.join(output_dir, f"{rid}.pkl")

        # skip if missing or already merged
        if not os.path.exists(single_path) or not os.path.exists(pair_path):
            print(f"[AF2][WARN] Missing pair/single for {rid}, skipping.")
            continue
        if os.path.exists(out_path):
            print(f"[AF2] {out_path} already exists, skipping.")
            continue

        single = np.load(single_path)
        pair   = np.load(pair_path)

        bundle = {
            "single": single,
            "pair": pair,
        }

        with open(out_path, "wb") as f:
            pickle.dump(bundle, f, protocol=4)

        print(f"[AF2] Wrote {out_path}")

    print(f"[AF2] Merge complete. Processed {len(repr_ids)} IDs.")


def run_af2_default():
    """
    Convenience runner that matches the repo directory layout.

    It assumes:
    - examples/example_input/test.csv         (your seqs_csv_path from protocol)
      format: id,seq_H,seq_L  (columns described under 'Sequence File' in protocol) 
    - examples/example_input/repr_dir/        (intermediate results directory repr_dir; .npy goes here)
    - examples/example_input/AF2_repr/        (final result directory AF2_repr_dir; merged .pkl goes here)
    """

    fasta_csv    = "examples/example_input/test.csv"
    repr_dir     = "examples/example_input/repr_dir"
    af2_repr_dir = "examples/example_input/AF2_repr"

    run_colabfold(
        fasta_csv=fasta_csv,
        repr_dir=repr_dir,
        af2_repr_dir=af2_repr_dir,
        num_recycle=3,
        model_type="AlphaFold2-multimer-v2",
        use_templates=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract AlphaFold2 features (single+pair repr) and merge them into AF2_repr_dir."
    )

    parser.add_argument(
        "--fasta_csv",
        required=False,
        default="examples/example_input/test.csv",
        help="Sequence CSV (id, seq_H, seq_L) as described in the protocol.",
    )
    parser.add_argument(
        "--repr_dir",
        required=False,
        default="examples/example_input/repr_dir",
        help="Intermediate results directory (repr_dir). ColabFold writes .npy here.",
    )
    parser.add_argument(
        "--af2_repr_dir",
        required=False,
        default="examples/example_input/AF2_repr",
        help="Final result directory (AF2_repr_dir). Merged .pkl per ID goes here.",
    )
    parser.add_argument(
        "--num_recycle",
        type=int,
        default=3,
        help="Matches the protocol example: --num-recycle 3",
    )
    parser.add_argument(
        "--model_type",
        default="AlphaFold2-multimer-v2",
        help="Matches protocol: --model-type AlphaFold2-multimer-v2",
    )
    parser.add_argument(
        "--no_templates",
        action="store_true",
        help="If set, do not include --templates flag.",
    )

    args = parser.parse_args()

    run_colabfold(
        fasta_csv=args.fasta_csv,
        repr_dir=args.repr_dir,
        af2_repr_dir=args.af2_repr_dir,
        num_recycle=args.num_recycle,
        model_type=args.model_type,
        use_templates=(not args.no_templates),
    )
