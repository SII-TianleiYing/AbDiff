# AbDiff Pipeline — Technical Reference

[![English](https://img.shields.io/badge/Language-English-blue)](pipeline.md)
[![中文](https://img.shields.io/badge/语言-中文-green)](pipeline.zh.md)

This document provides a detailed technical reference for the AbDiff pipeline, including stage commands, environment routing, checkpoints, and troubleshooting. For a quickstart, see:

- [`README.md`](../README.md) (English)
- [`README.zh.md`](../README.zh.md) (中文)

---

## 1. Architecture Overview

```
input_csv + fasta_dir
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│ Stage 1  AF2/ColabFold         (env: abdiff_colabfold)       │
│          → AF2_repr_raw/ (.npy) → AF2_repr/ (.pkl)           │
├──────────────────────────────────────────────────────────────┤
│ Stage 2  IgFold                (env: abdiff_igfold)          │
│          → igfold_embedding/ (.pt)                           │
├──────────────────────────────────────────────────────────────┤
│ Stage 3  AbFold Encoder        (env: abdiff_abfold)          │
│          → abfold_embedding/ (*_pred.pt)                     │
├──────────────────────────────────────────────────────────────┤
│ Stage 4  CDR-H3 Mask           (env: abdiff_abfold)          │
│          → cdr_mask_H3/ (.pt)                                │
├──────────────────────────────────────────────────────────────┤
│ Stage 5  AbDiff Sampling       (env: abdiff_diffusion)       │
│          → gen_abdiff_embeddings/ (.pt)                      │
├──────────────────────────────────────────────────────────────┤
│ Stage 6  Structure Decode      (env: abdiff_abfold)          │
│          → gen_structures/ (.pdb)                            │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. Stage Commands

All commands are run from the repository root via `conda run -n <env>`:

### Stage 1 — AF2/ColabFold

```bash
conda run -n abdiff_colabfold python -m abdiff.af2.run_af2 \
  --fasta_csv "$INPUT_CSV" \
  --repr_dir "$OUTPUT_ROOT/AF2_repr_raw" \
  --af2_repr_dir "$OUTPUT_ROOT/AF2_repr"
```

### Stage 2 — IgFold

```bash
conda run -n abdiff_igfold python -m abdiff.igfold.run_igfold \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/igfold_embedding"
```

### Stage 3 — AbFold Encoder

```bash
conda run -n abdiff_abfold python -m abdiff.abfold_encoder.run_abfold_encoder \
  --repr_dir "$OUTPUT_ROOT/AF2_repr" \
  --fasta_dir "$FASTA_DIR" \
  --point_feat_dir "$OUTPUT_ROOT/igfold_embedding" \
  --output_dir "$OUTPUT_ROOT/abfold_embedding" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

### Stage 4 — CDR-H3 Mask

```bash
conda run -n abdiff_abfold python -m abdiff.h3_mask.run_h3_mask \
  --fasta_dir "$FASTA_DIR" \
  --output_dir "$OUTPUT_ROOT/cdr_mask_H3"
```

### Stage 5 — AbDiff Sampling

```bash
conda run -n abdiff_diffusion python -m abdiff.abdiff_sampling.sample_embedding \
  --cdr_mask_dir "$OUTPUT_ROOT/cdr_mask_H3" \
  --abfold_embedding_dir "$OUTPUT_ROOT/abfold_embedding" \
  --output_dir "$OUTPUT_ROOT/gen_abdiff_embeddings"
```
- **Requires CUDA GPU** — `sample.py` defaults to `cuda:0`

### Stage 6 — Structure Decode

```bash
conda run -n abdiff_abfold python -m abdiff.structure_decode.run_structure_decode \
  --sample_emb_dir "$OUTPUT_ROOT/gen_abdiff_embeddings" \
  --output_dir "$OUTPUT_ROOT/gen_structures" \
  --checkpoint_name "checkpoints/abfold/checkpoint_ema"
```

---

## 3. Checkpoints

### AbFold checkpoint

| Item | Value |
|---|---|
| Path | `checkpoints/abfold/checkpoint_ema` |
| Format | PyTorch state dict |

### AbDiff diffusion checkpoint

| Item | Value |
|---|---|
| Path | `checkpoints/abdiff/20250103_1_a_1/` |
| Format | Diffusers pipeline (UNet + scheduler) |
| Distribution | `.tar` archive, auto-extracted by `preparation.sh` |

---

## 4. Troubleshooting

| Symptom | Likely cause | Resolution |
|---|---|---|
| `conda: command not found` | Conda not in PATH | Install Anaconda/Miniconda and restart shell |
| `model_index.json not found` | Diffusion checkpoint missing or extraction failed | Re-run `preparation.sh` or manually extract `.tar` to `checkpoints/abdiff/20250103_1_a_1/` |
| `CUDA out of memory` at Stage 5 | Insufficient GPU VRAM | Use a GPU with more VRAM; Stage 5 requires CUDA |
| Stage 5 fails at import | Checkpoint directory must exist at import time | Ensure `checkpoints/abdiff/20250103_1_a_1/` is populated |
| `hhsearch not found` at Stage 1 | ColabFold templates enabled but HHsuite missing | Install HHsuite in `abdiff_colabfold` env, or disable templates |
| Wrong PDB outputs | AbFold checkpoint version mismatch | Verify `checkpoints/abfold/checkpoint_ema` matches the expected model version |
