import os
import argparse
import time
import torch

# from abfold.config import config
# from abfold.data_process import process_repr, process_fasta
# from abfold.model import AbFold
# from abfold.residue_constants import str_sequence_to_aatype
# from abfold.tensor_utils import tensor_tree_map
# from abfold.train_ema import tensor_dict_to_device


from abfold.config import config
from abfold.data.data_process import process_repr, process_fasta, get_frames_from_pdb, load_mask_for_coord
from abfold.model import AbFold
from abfold.np import protein
from abfold.np.residue_constants import str_sequence_to_aatype
from abfold.utils.tensor_utils import tensor_tree_map
from abfold.train_ema import tensor_dict_to_device

def run_abfold_encoder(
    repr_dir: str,
    fasta_dir: str,
    point_feat_dir: str,
    output_dir: str,
    checkpoint_name: str,
    device: str = "cuda:0",
):
    """
    Step 3: Extract AbFold Features (AbFold Encoder Stage)
    ------------------------------------------------------
    Fuses AlphaFold2 representations, IgFold embeddings, and FASTA sequences
    to generate unified AbFold embeddings for each antibody.

    Inputs:
        repr_dir: Path to directory with AF2 .pkl files (AF2_repr/)
        fasta_dir: Path to directory with FASTA files (fasta_dir/)
        point_feat_dir: Path to directory with IgFold .pt embeddings (igfold_embedding/)
        output_dir: Directory to save AbFold .pt results (abfold_embedding/)
        checkpoint_name: Path to trained AbFold checkpoint (.pt or folder)
        device: "cuda:0" or "cpu"

    Output:
        Saves per-antibody files:
            <ID>_pred.pt  →  dict of extracted embeddings
    """

    os.makedirs(output_dir, exist_ok=True)

    # 1️⃣ Load model
    print(f"[AbFold] Loading model from checkpoint: {checkpoint_name}")
    model = AbFold(config)
    checkpoint = torch.load(checkpoint_name, map_location="cpu")
    if "student" in checkpoint:
        model.load_state_dict(checkpoint["student"])
    elif "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        raise ValueError("Checkpoint format not recognized (expected 'student' or 'model' keys).")

    model.eval().to(device)

    # 2️⃣ Iterate over each antibody ID
    for file in sorted(os.listdir(repr_dir)):
        if not file.endswith(".pkl"):
            continue

        repr_id = os.path.splitext(file)[0]
        repr_path = os.path.join(repr_dir, f"{repr_id}.pkl")
        fasta_path = os.path.join(fasta_dir, f"{repr_id}.fasta")
        point_feat_path = os.path.join(point_feat_dir, f"{repr_id}.pt")
        output_path = os.path.join(output_dir, f"{repr_id}_pred.pt")

        print(f"\n[AbFold] Processing {repr_id}")

        # 3️⃣ Prepare inputs
        seq_h, seq_l = process_fasta(fasta_path)
        s_h, s_l, z = process_repr(repr_path, seq_h)
        z_h, z_l = z.split([s_h.shape[-2], s_l.shape[-2]], -3)
        z_hh, z_hl = z_h.split([s_h.shape[-2], s_l.shape[-2]], -2)
        z_lh, z_ll = z_l.split([s_h.shape[-2], s_l.shape[-2]], -2)
        aatype_h = torch.tensor(str_sequence_to_aatype(seq_h))
        aatype_l = torch.tensor(str_sequence_to_aatype(seq_l))

        # 4️⃣ Load IgFold embeddings
        igfold_emb = torch.load(point_feat_path, map_location="cpu")
        point_feat = igfold_emb["structure_embs"]
        plddt = igfold_emb["prmsd"]
        point_feat.requires_grad = False
        plddt.requires_grad = False
        total_len = len(seq_h) + len(seq_l)
        if total_len != point_feat.shape[0]:
            point_feat = point_feat[:total_len]
            plddt = plddt[:total_len]

        # 5️⃣ Combine into full antibody feature dict
        s = torch.cat([s_h, s_l], dim=0)
        z_h = torch.cat([z_hh, z_hl], dim=1)
        z_l = torch.cat([z_lh, z_ll], dim=1)
        z = torch.cat([z_h, z_l], dim=0)
        aatype = torch.cat([aatype_h, aatype_l], dim=0)
        res_idx = torch.cat([
            torch.arange(0, aatype_h.shape[-1]),
            torch.arange(0, aatype_l.shape[-1]) + 256,
        ], dim=-1)

        feature = {
            "s": s,
            "z": z,
            "aatype": aatype,
            "point_feat": point_feat,
            "plddt": plddt,
            "res_idx": res_idx,
        }

        feature = tensor_tree_map(lambda x: x.unsqueeze(dim=0), feature)
        feature = tensor_dict_to_device(feature, device)

        # 6️⃣ Extract embeddings
        t1 = time.time()
        with torch.no_grad():
            result = model(feature, extract_embedding=True)
        t2 = time.time()
        print(f"[AbFold] {repr_id} done in {t2 - t1:.2f}s")

        # 7️⃣ Save outputs
        result["seq"] = {"H": seq_h, "L": seq_l}
        torch.save(result, output_path)
        print(f"[AbFold] Saved: {output_path}")

    print(f"\n✅ AbFold feature extraction complete → {output_dir}")


def run_abfold_encoder_default():
    """Default path version matching repo layout."""
    repr_dir = "examples/example_input/AF2_repr"
    fasta_dir = "examples/example_input/fasta_dir"
    point_feat_dir = "examples/example_input/igfold_embedding"
    output_dir = "examples/example_input/abfold_embedding"
    checkpoint_name = "checkpoints/abfold/checkpoint_ema.pt"

    run_abfold_encoder(
        repr_dir=repr_dir,
        fasta_dir=fasta_dir,
        point_feat_dir=point_feat_dir,
        output_dir=output_dir,
        checkpoint_name=checkpoint_name,
        device="cuda:0",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run AbFold Encoder to generate antibody embeddings"
    )
    parser.add_argument("--repr_dir", type=str, default="examples/example_input/AF2_repr")
    parser.add_argument("--fasta_dir", type=str, default="examples/example_input/fasta_dir")
    parser.add_argument("--point_feat_dir", type=str, default="examples/example_input/igfold_embedding")
    parser.add_argument("--output_dir", type=str, default="examples/example_input/abfold_embedding")
    parser.add_argument("--checkpoint_name", type=str, default="checkpoints/abfold/checkpoint_ema")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_abfold_encoder(
        repr_dir=args.repr_dir,
        fasta_dir=args.fasta_dir,
        point_feat_dir=args.point_feat_dir,
        output_dir=args.output_dir,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )
