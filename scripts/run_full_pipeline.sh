#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_full_pipeline.sh --input_csv <path> --fasta_dir <dir> [--output_root <dir>] [--abfold_ckpt <path>]

Notes:
  - 请在 AbDiff 仓库根目录运行（abdiff/ scripts/ environments/ 同级）
  - output_root 默认: ./output
  - 所有中间产物与最终结果均写入 output_root 下
EOF
}

INPUT_CSV=""
FASTA_DIR=""
OUTPUT_ROOT="output"
ABFOLD_CKPT="checkpoints/abfold/checkpoint_ema"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input_csv) INPUT_CSV="$2"; shift 2;;
    --fasta_dir) FASTA_DIR="$2"; shift 2;;
    --output_root) OUTPUT_ROOT="$2"; shift 2;;
    --abfold_ckpt) ABFOLD_CKPT="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ -z "$INPUT_CSV" || -z "$FASTA_DIR" ]]; then
  echo "[ERROR] --input_csv and --fasta_dir are required." >&2
  usage
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"

# Stage output layout (all under OUTPUT_ROOT)
AF2_RAW_DIR="$OUTPUT_ROOT/AF2_repr_raw"
AF2_REPR_DIR="$OUTPUT_ROOT/AF2_repr"
IGFOLD_DIR="$OUTPUT_ROOT/igfold_embedding"
ABFOLD_EMB_DIR="$OUTPUT_ROOT/abfold_embedding"
H3_MASK_DIR="$OUTPUT_ROOT/cdr_mask_H3"
SAMPLE_EMB_DIR="$OUTPUT_ROOT/gen_abdiff_embeddings"
STRUCT_DIR="$OUTPUT_ROOT/gen_structures"

mkdir -p \
  "$AF2_RAW_DIR" \
  "$AF2_REPR_DIR" \
  "$IGFOLD_DIR" \
  "$ABFOLD_EMB_DIR" \
  "$H3_MASK_DIR" \
  "$SAMPLE_EMB_DIR" \
  "$STRUCT_DIR"

echo "[pipeline] pre-check (paths only)"
python scripts/check.py --mode pre --input_csv "$INPUT_CSV" --fasta_dir "$FASTA_DIR"

echo "[pipeline] Step 1/6: ColabFold AF2 representations"
conda run -n abdiff_colabfold python -m abdiff.af2.run_af2 \
  --fasta_csv "$INPUT_CSV" \
  --repr_dir "$AF2_RAW_DIR" \
  --af2_repr_dir "$AF2_REPR_DIR"

echo "[pipeline] Step 2/6: IgFold embeddings"
conda run -n abdiff_igfold python -m abdiff.igfold.run_igfold \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$IGFOLD_DIR"

echo "[pipeline] Step 3/6: AbFold encoder fusion"
conda run -n abdiff_abfold python -m abdiff.abfold_encoder.run_abfold_encoder \
  --repr_dir "$AF2_REPR_DIR" \
  --fasta_dir "$FASTA_DIR" \
  --point_feat_dir "$IGFOLD_DIR" \
  --output_dir "$ABFOLD_EMB_DIR" \
  --checkpoint_name "$ABFOLD_CKPT"

echo "[pipeline] Step 4/6: CDR-H3 mask"
conda run -n abdiff_abfold python -m abdiff.h3_mask.run_h3_mask \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$H3_MASK_DIR"

echo "[pipeline] Step 5/6: AbDiff diffusion sampling"
conda run -n abdiff_diffusion python -m abdiff.abdiff_sampling.sample_embedding \
  --cdr_mask_dir "$H3_MASK_DIR" \
  --abfold_embedding_dir "$ABFOLD_EMB_DIR" \
  --output_dir "$SAMPLE_EMB_DIR"

echo "[pipeline] Step 6/6: Structure decode -> PDB"
conda run -n abdiff_abfold python -m abdiff.structure_decode.run_structure_decode \
  --sample_emb_dir "$SAMPLE_EMB_DIR" \
  --output_dir "$STRUCT_DIR" \
  --checkpoint_name "$ABFOLD_CKPT"

echo "[pipeline] done"
echo "[pipeline] final PDBs: $STRUCT_DIR"

