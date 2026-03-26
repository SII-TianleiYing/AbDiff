#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

INPUT_CSV="examples/example_input/test.csv"
FASTA_DIR="examples/example_input/fasta_dir"
OUT_DIR="examples/example_output/run_full_pipeline_out"

mkdir -p "$OUT_DIR"

echo "[example] input_csv: $INPUT_CSV"
echo "[example] fasta_dir : $FASTA_DIR"
echo "[example] output    : $OUT_DIR"

bash scripts/run_full_pipeline.sh \
  --input_csv "$INPUT_CSV" \
  --fasta_dir "$FASTA_DIR" \
  --output_root "$OUT_DIR"

