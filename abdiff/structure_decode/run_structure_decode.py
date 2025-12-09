import os
import time
import argparse
import numpy as np
import torch

from abfold.config import config
from abfold.model import AbFold
from abfold.np import protein
from abfold.np.residue_constants import str_sequence_to_aatype
from abfold.utils.tensor_utils import tensor_tree_map
from train_ema import tensor_dict_to_device


def run_structure_decode(
    sample_emb_dir: str,
    output_dir: str,
    checkpoint_name: str,
    device: str = None,
):
    """
    Step 6: Structure Decoding / Final 3D Generation
    ------------------------------------------------
    Takes AbDiff-sampled embeddings (gen_abdiff_embeddings/*.pt),
    runs AbFold's decoder to generate 3D coordinates,
    and writes predicted structures as PDB files.

    Inputs:
        sample_emb_dir: directory containing sampled embeddings (.pt)
            e.g. examples/example_output/gen_abdiff_embeddings
        output_dir: directory to save final .pdb structures
            e.g. examples/example_output/gen_structures
        checkpoint_name: the AbFold checkpoint to load
            e.g. checkpoints/abfold/checkpoint_ema.pt
        device: "cuda:0" or "cpu"

    Output:
        <output_dir>/<sample_name>.pdb
    """

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load AbFold model + checkpoint
    print(f"[StructureDecode] Loading AbFold checkpoint: {checkpoint_name}")
    model = AbFold(config)
    checkpoint = torch.load(checkpoint_name, map_location="cpu")

    # Original logic: checkpoint might store weights under 'student' or 'model'
    if "student" in checkpoint:
        model.load_state_dict(checkpoint["student"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Checkpoint format not recognized (expected 'student' or 'model').")

    model.eval()
    model = model.to(device)

    # 2. Loop over every generated embedding .pt file in sample_emb_dir
    sample_files = [f for f in os.listdir(sample_emb_dir) if f.endswith(".pt")]
    for file in sample_files:
        filepath = os.path.join(sample_emb_dir, file)
        sample_name = os.path.splitext(file)[0]
        pdb_path = os.path.join(output_dir, f"{sample_name}.pdb")

        print(f"[StructureDecode] Decoding {file} -> {pdb_path}")

        # 3. Load generated embedding
        emb = torch.load(filepath, map_location=device)

        # emb is expected to contain:
        #   s, z, seq (with 'H' and 'L'), aatype
        s = emb["s"]
        z = emb["z"]
        seq_h = emb["seq"]["H"][0]
        seq_l = emb["seq"]["L"][0]
        aatype = emb["aatype"]

        # We'll also rebuild per-residue metadata for final PDB
        aatype_h = torch.tensor(str_sequence_to_aatype(seq_h))
        aatype_l = torch.tensor(str_sequence_to_aatype(seq_l))

        # 4. Run structure generation
        t1 = time.time()
        with torch.no_grad():
            result = model.sample(s, z, aatype)
        t2 = time.time()
        print(f"[StructureDecode] {file} decoded in {t2 - t1:.2f}s")

        # Move tensors to CPU and squeeze batch dim
        result = tensor_tree_map(lambda x: x.squeeze(dim=0), result)
        result = tensor_dict_to_device(result, "cpu")

        # 5. Build features needed by abfold.np.protein.to_pdb()
        #    NOTE: this is copied from your original script logic. :contentReference[oaicite:1]{index=1}
        feature = {}
        feature["aatype"] = torch.cat([aatype_h, aatype_l], -1).numpy()
        feature["asym_id"] = np.concatenate(
            [
                np.ones([len(seq_h)], np.int8) - 1,
                np.ones([len(seq_l)], np.int8),
            ],
            -1,
        )
        feature["residue_index"] = torch.cat(
            [
                torch.arange(0, len(seq_h)),
                torch.arange(0, len(seq_l)),
            ],
            dim=-1,
        ).numpy()

        # 6. Convert model output to a protein-like object and then to PDB text
        unrelaxed_protein = protein.from_prediction(feature, result)
        pdb_str = protein.to_pdb(unrelaxed_protein)

        # 7. Write the .pdb file
        with open(pdb_path, "w") as fp:
            fp.write(pdb_str)

        print(f"[StructureDecode] Saved PDB: {pdb_path}")

    print(f"\n✅ Done. All predicted structures are in {output_dir}")


def run_structure_decode_default():
    """
    Default convenience wrapper matching our repo layout.
    """
    sample_emb_dir = "examples/example_output/gen_abdiff_embeddings"
    output_dir = "examples/example_output/gen_structures"
    checkpoint_name = "checkpoints/abfold/checkpoint_ema"

    run_structure_decode(
        sample_emb_dir=sample_emb_dir,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
        device=None,  # will auto-pick cuda or cpu
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Decode AbDiff-sampled embeddings into final antibody PDB structures"
    )

    parser.add_argument(
        "--sample_emb_dir",
        type=str,
        default="examples/example_output/gen_abdiff_embeddings",
        help="Directory containing AbDiff-sampled embeddings (.pt files).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="examples/example_output/gen_structures",
        help="Directory to write final predicted structures (.pdb).",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="checkpoints/abfold/checkpoint_ema",
        help="Path to AbFold checkpoint with decoder weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (e.g. 'cuda:0' or 'cpu'). Leave empty for auto.",
    )

    args = parser.parse_args()

    run_structure_decode(
        sample_emb_dir=args.sample_emb_dir,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )
