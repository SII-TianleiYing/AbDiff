## AbDiff Pipeline (Internal Notes / English)

This document is intended for **maintainers/internal collaboration**. It describes the current orchestration strategy, IO conventions, and common failure points. For user-facing quickstart, see:
- `README.md` (English)
- `README.zh-CN.md` (中文)

### Overall strategy

Hard constraints for this repository:
- **Do not reorganize the project layout**: `abdiff/`, `scripts/`, `environments/`, `checkpoints/` stay as sibling directories.
- **Fix issues at orchestration layer whenever possible**: avoid touching core module logic unless path assumptions cannot be addressed via CLI args.
- **Single output root**: when users pass `--output_root`, all intermediate and final artifacts must live under that directory.

### External input interface

The full pipeline exposes two primary inputs:
- `input_csv`: ColabFold input CSV (`--input_csv`)
- `fasta_dir`: directory containing paired `.fasta` files (`--fasta_dir`)

### Output layout (all under output_root)

`scripts/run_full_pipeline.sh` creates the following under `output_root/`:
- `AF2_repr_raw/`: raw `.npy` representations from ColabFold
- `AF2_repr/`: merged `.pkl` per sample
- `igfold_embedding/`: IgFold `.pt`
- `abfold_embedding/`: AbFold encoder output `*_pred.pt`
- `cdr_mask_H3/`: CDR-H3 mask `.pt`
- `gen_abdiff_embeddings/`: diffusion-sampled embedding `.pt`
- `gen_structures/`: final `.pdb`

### Stage order and commands

All commands are run from the repo root, executed via `conda run -n <env>`:

1. **AF2/ColabFold (env: `abdiff_colabfold`)**

```bash
conda run -n abdiff_colabfold python -m abdiff.af2.run_af2 \
  --fasta_csv "$INPUT_CSV" \
  --repr_dir "$OUTPUT_ROOT/AF2_repr_raw" \
  --af2_repr_dir "$OUTPUT_ROOT/AF2_repr"
```

2. **IgFold (env: `abdiff_igfold`)**

```bash
conda run -n abdiff_igfold python -m abdiff.igfold.run_igfold \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/igfold_embedding"
```

3. **AbFold encoder (env: `abdiff_abfold`)**

```bash
conda run -n abdiff_abfold python -m abdiff.abfold_encoder.run_abfold_encoder \
  --repr_dir "$OUTPUT_ROOT/AF2_repr" \
  --fasta_dir "$FASTA_DIR" \
  --point_feat_dir "$OUTPUT_ROOT/igfold_embedding" \
  --output_dir "$OUTPUT_ROOT/abfold_embedding" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

4. **H3 mask (env: `abdiff_abfold`)**

```bash
conda run -n abdiff_abfold python -m abdiff.h3_mask.run_h3_mask \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/cdr_mask_H3"
```

5. **AbDiff sampling (env: `abdiff_diffusion`)**

```bash
conda run -n abdiff_diffusion python -m abdiff.abdiff_sampling.sample_embedding \
  --cdr_mask_dir "$OUTPUT_ROOT/cdr_mask_H3" \
  --abfold_embedding_dir "$OUTPUT_ROOT/abfold_embedding" \
  --output_dir "$OUTPUT_ROOT/gen_abdiff_embeddings"
```

6. **Structure decode (env: `abdiff_abfold`)**

```bash
conda run -n abdiff_abfold python -m abdiff.structure_decode.run_structure_decode \
  --sample_emb_dir "$OUTPUT_ROOT/gen_abdiff_embeddings" \
  --output_dir "$OUTPUT_ROOT/gen_structures" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

### Weights / checkpoints

#### AbFold checkpoint (single file)
- Expected path: `checkpoints/abfold/checkpoint_ema`
- Note: in the current snapshot it has **no `.pt` suffix**, and orchestration scripts follow this convention.

#### AbDiff diffusion checkpoint (diffusers directory)
- Expected directory: `checkpoints/abdiff/20250103_1_a_1/`
- Minimal required files:
  - `model_index.json`
  - `unet/config.json`
  - `scheduler/scheduler_config.json`

`scripts/preparation.sh` assumes the diffusion checkpoint is distributed as a zip and extracts it to this directory.

### Common failure points (pre-flight hints)

- **conda not available**: `scripts/check.py` will fail early.
- **wrong diffusion zip layout / incomplete extraction**:
  - `scripts/check.py --mode post` validates `model_index.json`, `unet/config.json`, `scheduler/scheduler_config.json`.
- **Step5 import has side effects**:
  - `abdiff/abdiff_sampling/diffusion/sample.py` loads the diffusers checkpoint at import time, so the checkpoint directory must exist and you must run from repo root.
- **sampling defaults to CUDA**:
  - on non-GPU machines or insufficient VRAM, Step5 will fail at runtime (core module behavior; orchestration does not change it).

